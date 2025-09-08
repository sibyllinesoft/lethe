"""
LetheBench Source Data Crawlers

Implements license-safe data collection for the three LetheBench genres:
- Code: GitHub repositories and Stack Overflow Q&A
- Tool: CLI documentation and notebook transcripts  
- Prose: Meeting transcripts and Wikipedia discussions

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

from .github_crawler import GitHubCrawler
from .stackoverflow_crawler import StackOverflowCrawler  
from .transcript_crawler import TranscriptCrawler

__all__ = ['GitHubCrawler', 'StackOverflowCrawler', 'TranscriptCrawler']