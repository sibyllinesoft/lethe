#!/usr/bin/env python3
"""
Stack Overflow Data Crawler for LetheBench Code Genre

Collects CC-licensed Q&A content with substantial code discussion,
focusing on accepted answers and detailed explanations.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline  
"""

import requests
import time
import json
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
import logging
from datetime import datetime
import gzip
from urllib.parse import quote

@dataclass
class StackOverflowSession:
    """Represents a Stack Overflow Q&A session."""
    session_id: str
    question_id: int
    title: str
    question_body: str
    answers: List[Dict]
    tags: List[str]
    score: int
    view_count: int
    creation_date: int
    last_activity_date: int
    has_accepted_answer: bool
    license: str = "CC BY-SA 4.0"

class StackOverflowCrawler:
    """
    Crawls Stack Overflow for code-related Q&A sessions.
    
    Features:
    - Uses Stack Exchange API with proper attribution
    - Focuses on questions with accepted answers and code
    - Respects API rate limits and quotas  
    - Maintains CC licensing compliance
    - Filters for substantial technical discussions
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.1):
        """
        Initialize Stack Overflow crawler.
        
        Args:
            api_key: Stack Exchange API key (increases quotas)
            rate_limit_delay: Delay between API requests
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
        # Stack Exchange API configuration
        self.base_url = "https://api.stackexchange.com/2.3"
        self.site = "stackoverflow"
        
        # Track API usage
        self.requests_made = 0
        self.quota_remaining = 10000  # Default quota
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited API request to Stack Exchange API."""
        if params is None:
            params = {}
        
        # Add required parameters
        params['site'] = self.site
        if self.api_key:
            params['key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            self.logger.debug(f"API Request: {endpoint} with params: {params}")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                # Update quota tracking
                data = response.json()
                if 'quota_remaining' in data:
                    self.quota_remaining = data['quota_remaining']
                
                self.requests_made += 1
                time.sleep(self.rate_limit_delay)  # Be respectful
                
                return data
            else:
                self.logger.error(f"API request failed: {response.status_code} - {endpoint}")
                if response.status_code == 400:
                    self.logger.error(f"Response: {response.text}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return None
    
    def find_quality_questions(self, 
                             tags: List[str] = None, 
                             min_score: int = 5,
                             max_questions: int = 1000,
                             has_accepted_answer: bool = True) -> List[Dict]:
        """
        Find high-quality questions with substantial code discussion.
        
        Args:
            tags: List of tags to search for (e.g., ['python', 'javascript'])
            min_score: Minimum question score
            max_questions: Maximum questions to collect
            has_accepted_answer: Only include questions with accepted answers
            
        Returns:
            List of question metadata
        """
        if tags is None:
            tags = ['python', 'javascript', 'java', 'c++', 'go', 'rust']
        
        questions = []
        
        for tag in tags:
            if len(questions) >= max_questions:
                break
            
            self.logger.info(f"Searching for {tag} questions...")
            
            params = {
                'order': 'desc',
                'sort': 'votes',
                'tagged': tag,
                'filter': '!9YdnSMKKT',  # Include body, answers, and other details
                'pagesize': min(100, max_questions - len(questions))
            }
            
            response = self._make_api_request('questions', params)
            
            if not response or 'items' not in response:
                continue
            
            for question in response['items']:
                # Filter for quality questions
                if (question.get('score', 0) >= min_score and 
                    (not has_accepted_answer or question.get('is_answered', False)) and
                    len(question.get('body', '')) > 200):  # Substantial question
                    
                    questions.append(question)
                    
                    if len(questions) >= max_questions:
                        break
            
            if self.quota_remaining < 100:
                self.logger.warning("API quota running low, stopping early")
                break
        
        self.logger.info(f"Found {len(questions)} quality questions")
        return questions
    
    def get_question_with_answers(self, question_id: int) -> Optional[StackOverflowSession]:
        """
        Get complete question with all answers and comments.
        
        Args:
            question_id: Stack Overflow question ID
            
        Returns:
            StackOverflowSession with complete Q&A data
        """
        # Get question details
        params = {
            'filter': '!9YdnSMKKT'  # Include body and other details
        }
        
        question_response = self._make_api_request(f'questions/{question_id}', params)
        
        if not question_response or 'items' not in question_response or not question_response['items']:
            return None
        
        question = question_response['items'][0]
        
        # Get answers
        answers_params = {
            'order': 'desc',
            'sort': 'votes',
            'filter': '!9YdnSJ3dZ'  # Include body and comments
        }
        
        answers_response = self._make_api_request(f'questions/{question_id}/answers', answers_params)
        
        answers = []
        if answers_response and 'items' in answers_response:
            for answer in answers_response['items']:
                # Only include substantial answers
                if len(answer.get('body', '')) > 100:
                    answers.append({
                        'answer_id': answer['answer_id'],
                        'body': answer['body'],
                        'score': answer['score'],
                        'is_accepted': answer.get('is_accepted', False),
                        'creation_date': answer['creation_date'],
                        'last_activity_date': answer.get('last_activity_date', answer['creation_date']),
                        'owner': answer.get('owner', {}).get('display_name', 'Anonymous')
                    })
        
        # Only proceed if we have good answers
        if not answers:
            return None
        
        return StackOverflowSession(
            session_id=f"stackoverflow_q_{question_id}",
            question_id=question_id,
            title=question['title'],
            question_body=question['body'],
            answers=answers,
            tags=question.get('tags', []),
            score=question['score'],
            view_count=question.get('view_count', 0),
            creation_date=question['creation_date'],
            last_activity_date=question.get('last_activity_date', question['creation_date']),
            has_accepted_answer=question.get('is_answered', False)
        )
    
    def collect_qa_sessions(self, questions: List[Dict]) -> Iterator[StackOverflowSession]:
        """
        Collect complete Q&A sessions from question list.
        
        Args:
            questions: List of question metadata from find_quality_questions
            
        Yields:
            StackOverflowSession objects with complete Q&A data
        """
        for question in questions:
            if self.quota_remaining < 50:
                self.logger.warning("API quota too low, stopping collection")
                break
            
            session = self.get_question_with_answers(question['question_id'])
            
            if session and len(session.answers) >= 1:  # Must have at least one good answer
                yield session
    
    def convert_to_lethebench_format(self, session: StackOverflowSession) -> List[Dict]:
        """
        Convert Stack Overflow session to LetheBench JSONL format.
        
        Returns:
            List of turn dictionaries in LetheBench format
        """
        turns = []
        turn_id = 0
        
        # Question as initial turn
        turns.append({
            'session_id': session.session_id,
            'turn': turn_id,
            'role': 'user',
            'text': f"**{session.title}**\n\n{session.question_body}",
            'ts': datetime.fromtimestamp(session.creation_date).isoformat(),
            'meta': {
                'type': 'question',
                'question_id': session.question_id,
                'tags': session.tags,
                'score': session.score,
                'view_count': session.view_count,
                'license': session.license,
                'source': 'stackoverflow'
            }
        })
        turn_id += 1
        
        # Sort answers by acceptance (accepted first), then by score
        sorted_answers = sorted(session.answers, 
                              key=lambda x: (not x.get('is_accepted', False), -x['score']))
        
        # Answers as assistant turns
        for answer in sorted_answers:
            turns.append({
                'session_id': session.session_id,
                'turn': turn_id,
                'role': 'assistant',
                'text': answer['body'],
                'ts': datetime.fromtimestamp(answer['creation_date']).isoformat(),
                'meta': {
                    'type': 'answer',
                    'answer_id': answer['answer_id'],
                    'is_accepted': answer.get('is_accepted', False),
                    'score': answer['score'],
                    'author': answer['owner'],
                    'license': session.license,
                    'source': 'stackoverflow'
                }
            })
            turn_id += 1
        
        return turns
    
    def get_licensing_info(self, sessions: List[StackOverflowSession]) -> List[Dict]:
        """
        Generate licensing information for collected Stack Overflow data.
        
        Returns:
            List of licensing entries for manifest
        """
        licensing_info = []
        
        for session in sessions:
            licensing_info.append({
                'source': 'stackoverflow',
                'url': f"https://stackoverflow.com/questions/{session.question_id}",
                'license': 'CC BY-SA 4.0',
                'attribution': 'Stack Overflow contributors',
                'question_id': session.question_id,
                'title': session.title[:100] + "..." if len(session.title) > 100 else session.title
            })
        
        return licensing_info
    
    def get_api_usage_stats(self) -> Dict:
        """Get API usage statistics."""
        return {
            'requests_made': self.requests_made,
            'quota_remaining': self.quota_remaining,
            'quota_used_percent': ((10000 - self.quota_remaining) / 10000) * 100
        }

def test_stackoverflow_crawler():
    """Test the Stack Overflow crawler with a small sample."""
    crawler = StackOverflowCrawler()
    
    # Find quality questions
    questions = crawler.find_quality_questions(
        tags=['python'], 
        min_score=10,
        max_questions=5
    )
    
    print(f"Found {len(questions)} questions")
    
    # Collect complete sessions
    sessions = list(crawler.collect_qa_sessions(questions))
    print(f"Collected {len(sessions)} complete sessions")
    
    for session in sessions[:2]:  # Test with first 2 sessions
        turns = crawler.convert_to_lethebench_format(session)
        
        print(f"\nSession {session.session_id}:")
        print(f"  Title: {session.title}")
        print(f"  Tags: {session.tags}")
        print(f"  Turns: {len(turns)}")
        print(f"  Answers: {len(session.answers)}")
        
        # Show first turn (question)
        if turns:
            print(f"  Question preview: {turns[0]['text'][:200]}...")
    
    # Show API usage
    usage_stats = crawler.get_api_usage_stats()
    print(f"\nAPI Usage: {usage_stats}")
    
    # Show licensing info
    licensing_info = crawler.get_licensing_info(sessions)
    print(f"Licensing entries: {len(licensing_info)}")

if __name__ == "__main__":
    test_stackoverflow_crawler()