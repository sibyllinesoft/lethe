#!/usr/bin/env python3
"""
GitHub Data Crawler for LetheBench Code Genre

Collects license-safe code discussions from public repositories with
permissive licenses (MIT, Apache, BSD) focusing on Issues and PR discussions.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import requests
import time
import json
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

@dataclass
class GitHubSession:
    """Represents a GitHub discussion session (Issue or PR)."""
    session_id: str
    repository: str
    title: str
    body: str
    comments: List[Dict]
    created_at: str
    updated_at: str
    labels: List[str]
    state: str
    license: str
    url: str

class GitHubCrawler:
    """
    Crawls GitHub for code-related discussions from permissive-licensed repositories.
    
    Features:
    - Respects GitHub API rate limits
    - Filters for permissive licenses only (MIT, Apache-2.0, BSD)
    - Collects Issues and PRs with substantial discussion
    - Maintains full licensing information for compliance
    """
    
    PERMISSIVE_LICENSES = {
        'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 
        'ISC', 'Unlicense', 'CC0-1.0'
    }
    
    def __init__(self, github_token: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize GitHub crawler.
        
        Args:
            github_token: GitHub API token (optional, increases rate limits)
            rate_limit_delay: Delay between API requests in seconds
        """
        self.github_token = github_token
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
        
        self.collected_repos = set()
        self.licensing_info = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_api_request(self, url: str) -> Optional[Dict]:
        """Make rate-limited API request to GitHub."""
        try:
            self.logger.debug(f"API Request: {url}")
            response = self.session.get(url)
            
            if response.status_code == 403:
                # Rate limit exceeded
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - int(time.time()), 60)
                self.logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds.")
                time.sleep(wait_time)
                response = self.session.get(url)
            
            if response.status_code == 200:
                time.sleep(self.rate_limit_delay)  # Be respectful
                return response.json()
            else:
                self.logger.error(f"API request failed: {response.status_code} - {url}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return None
    
    def _get_repo_license(self, owner: str, repo: str) -> Optional[str]:
        """Get repository license information."""
        url = f"https://api.github.com/repos/{owner}/{repo}/license"
        response = self._make_api_request(url)
        
        if response and 'license' in response:
            return response['license']['spdx_id']
        return None
    
    def find_target_repositories(self, min_issues: int = 50, max_repos: int = 100) -> List[Dict]:
        """
        Find repositories with substantial discussion activity and permissive licenses.
        
        Args:
            min_issues: Minimum number of issues for inclusion
            max_repos: Maximum repositories to collect
            
        Returns:
            List of repository metadata dictionaries
        """
        target_repos = []
        
        # Search for repositories with good discussion activity
        search_queries = [
            'language:python stars:>100 license:mit',
            'language:javascript stars:>100 license:apache-2.0', 
            'language:java stars:>100 license:bsd',
            'language:go stars:>50 license:mit',
            'language:rust stars:>50 license:apache-2.0'
        ]
        
        for query in search_queries:
            if len(target_repos) >= max_repos:
                break
                
            url = f"https://api.github.com/search/repositories?q={query}&sort=updated&per_page=30"
            response = self._make_api_request(url)
            
            if not response or 'items' not in response:
                continue
            
            for repo in response['items']:
                if len(target_repos) >= max_repos:
                    break
                
                # Verify license and issue count
                license_id = self._get_repo_license(repo['owner']['login'], repo['name'])
                
                if license_id in self.PERMISSIVE_LICENSES and repo['open_issues_count'] >= min_issues:
                    repo_info = {
                        'owner': repo['owner']['login'],
                        'name': repo['name'],
                        'full_name': repo['full_name'],
                        'license': license_id,
                        'stars': repo['stargazers_count'],
                        'issues_count': repo['open_issues_count'],
                        'language': repo['language'],
                        'url': repo['html_url']
                    }
                    target_repos.append(repo_info)
                    
                    # Record licensing info
                    self.licensing_info.append({
                        'source': 'github',
                        'url': repo['html_url'],
                        'license': license_id,
                        'repository': repo['full_name']
                    })
        
        self.logger.info(f"Found {len(target_repos)} target repositories")
        return target_repos
    
    def collect_repository_discussions(self, repo_info: Dict, max_items: int = 50) -> Iterator[GitHubSession]:
        """
        Collect Issues and PRs with discussions from a repository.
        
        Args:
            repo_info: Repository information dictionary
            max_items: Maximum issues/PRs to collect per repository
            
        Yields:
            GitHubSession objects representing discussions
        """
        owner = repo_info['owner']
        repo_name = repo_info['name']
        license_id = repo_info['license']
        
        # Collect Issues
        issues_url = f"https://api.github.com/repos/{owner}/{repo_name}/issues"
        issues_response = self._make_api_request(f"{issues_url}?state=closed&sort=comments&per_page={max_items//2}")
        
        if issues_response:
            for issue in issues_response:
                if issue.get('comments', 0) >= 2:  # Must have discussion
                    comments = self._get_issue_comments(owner, repo_name, issue['number'])
                    
                    if comments:
                        yield GitHubSession(
                            session_id=f"github_issue_{owner}_{repo_name}_{issue['number']}",
                            repository=f"{owner}/{repo_name}",
                            title=issue['title'],
                            body=issue['body'] or "",
                            comments=comments,
                            created_at=issue['created_at'],
                            updated_at=issue['updated_at'],
                            labels=[label['name'] for label in issue.get('labels', [])],
                            state=issue['state'],
                            license=license_id,
                            url=issue['html_url']
                        )
        
        # Collect Pull Requests
        pulls_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
        pulls_response = self._make_api_request(f"{pulls_url}?state=closed&sort=updated&per_page={max_items//2}")
        
        if pulls_response:
            for pr in pulls_response:
                if pr.get('comments', 0) >= 2 or pr.get('review_comments', 0) >= 2:
                    comments = self._get_pr_comments(owner, repo_name, pr['number'])
                    
                    if comments:
                        yield GitHubSession(
                            session_id=f"github_pr_{owner}_{repo_name}_{pr['number']}",
                            repository=f"{owner}/{repo_name}",
                            title=pr['title'],
                            body=pr['body'] or "",
                            comments=comments,
                            created_at=pr['created_at'],
                            updated_at=pr['updated_at'],
                            labels=[label['name'] for label in pr.get('labels', [])],
                            state=pr['state'],
                            license=license_id,
                            url=pr['html_url']
                        )
    
    def _get_issue_comments(self, owner: str, repo: str, issue_number: int) -> List[Dict]:
        """Get all comments for an issue."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = self._make_api_request(url)
        
        if not response:
            return []
        
        comments = []
        for comment in response:
            comments.append({
                'id': comment['id'],
                'author': comment['user']['login'],
                'body': comment['body'],
                'created_at': comment['created_at'],
                'updated_at': comment['updated_at']
            })
        
        return comments
    
    def _get_pr_comments(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get all comments for a pull request (both issue and review comments)."""
        comments = []
        
        # Get issue comments
        issue_comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        issue_response = self._make_api_request(issue_comments_url)
        
        if issue_response:
            for comment in issue_response:
                comments.append({
                    'id': f"issue_{comment['id']}",
                    'type': 'issue_comment',
                    'author': comment['user']['login'],
                    'body': comment['body'],
                    'created_at': comment['created_at'],
                    'updated_at': comment['updated_at']
                })
        
        # Get review comments
        review_comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        review_response = self._make_api_request(review_comments_url)
        
        if review_response:
            for comment in review_response:
                comments.append({
                    'id': f"review_{comment['id']}",
                    'type': 'review_comment',
                    'author': comment['user']['login'],
                    'body': comment['body'],
                    'created_at': comment['created_at'],
                    'updated_at': comment['updated_at'],
                    'path': comment.get('path'),
                    'diff_hunk': comment.get('diff_hunk')
                })
        
        # Sort by creation time
        comments.sort(key=lambda x: x['created_at'])
        return comments
    
    def convert_to_lethebench_format(self, session: GitHubSession) -> List[Dict]:
        """
        Convert GitHub session to LetheBench JSONL format.
        
        Returns:
            List of turn dictionaries in LetheBench format
        """
        turns = []
        turn_id = 0
        
        # Initial post (issue/PR description)
        turns.append({
            'session_id': session.session_id,
            'turn': turn_id,
            'role': 'user',
            'text': f"**{session.title}**\n\n{session.body}",
            'ts': session.created_at,
            'meta': {
                'type': 'initial_post',
                'repository': session.repository,
                'labels': session.labels,
                'license': session.license,
                'url': session.url
            }
        })
        turn_id += 1
        
        # Comments as conversation turns
        for comment in session.comments:
            turns.append({
                'session_id': session.session_id,
                'turn': turn_id,
                'role': 'assistant' if 'bot' in comment['author'].lower() else 'user',
                'text': comment['body'],
                'ts': comment['created_at'],
                'meta': {
                    'type': 'comment',
                    'author': comment['author'],
                    'comment_id': comment['id'],
                    'comment_type': comment.get('type', 'issue_comment'),
                    'path': comment.get('path'),  # For review comments
                    'repository': session.repository,
                    'license': session.license
                }
            })
            turn_id += 1
        
        return turns
    
    def get_licensing_manifest(self) -> List[Dict]:
        """Get comprehensive licensing information for all collected data."""
        return self.licensing_info.copy()

def test_github_crawler():
    """Test the GitHub crawler with a small sample."""
    crawler = GitHubCrawler()
    
    # Find a few target repositories
    repos = crawler.find_target_repositories(min_issues=10, max_repos=2)
    print(f"Found {len(repos)} repositories")
    
    for repo in repos[:1]:  # Test with just one repo
        print(f"\nCrawling {repo['full_name']}...")
        
        session_count = 0
        for session in crawler.collect_repository_discussions(repo, max_items=5):
            session_count += 1
            turns = crawler.convert_to_lethebench_format(session)
            
            print(f"Session {session.session_id}: {len(turns)} turns")
            print(f"Title: {session.title[:100]}...")
            
            if session_count >= 2:  # Limit for testing
                break
        
        print(f"Collected {session_count} sessions from {repo['full_name']}")
    
    # Print licensing info
    manifest = crawler.get_licensing_manifest()
    print(f"\nLicensing manifest: {len(manifest)} entries")
    for entry in manifest:
        print(f"  {entry['repository']}: {entry['license']}")

if __name__ == "__main__":
    test_github_crawler()