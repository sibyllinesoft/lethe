"""
LetheBench Gold Annotation System

Implements weak supervision approaches for generating gold annotations
for the three LetheBench genres:

- Code: Extract code symbols, function names, file paths from accepted answers
- Tool: Identify tool outputs and trace dependencies backward  
- Prose: Find supporting spans through entity/temporal overlap

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

from .code_labeler import CodeLabeler
from .tool_labeler import ToolLabeler
from .prose_labeler import ProseLabeler

__all__ = ['CodeLabeler', 'ToolLabeler', 'ProseLabeler']