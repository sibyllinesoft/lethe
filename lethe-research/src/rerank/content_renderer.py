"""
Type-aware content renderer for cross-encoder input.

Renders different atom types into meaningful text for CE scoring,
preventing the "Document {doc_id}" fallback that causes flat scores.
"""

import logging
import re
from typing import Dict, Any, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ContentRenderer:
    """Renders atoms into CE-appropriate text based on type."""
    
    def __init__(self, tokenizer_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize with tokenizer for accurate token counting."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_passage_tokens = 384  # Reserve ~128 for query + special tokens
        self.query_cap_tokens = 96     # Cap query to prevent truncation
        
    def render_for_ce(self, atom: Dict[str, Any]) -> str:
        """
        Render atom content for cross-encoder scoring.
        
        Returns meaningful text based on atom type, properly formatted
        for BERT-style cross-encoder input.
        """
        atom_type = atom.get('type', 'UNKNOWN')
        content = atom.get('text', '') or atom.get('content', '')
        
        if not content and 'id' in atom:
            # Last resort fallback, but should trigger guards
            logger.warning(f"Empty content for atom {atom['id']}, using ID fallback")
            return f"Document {atom['id']}"
        
        # Type-specific rendering
        if atom_type in ['CODE', 'ERROR']:
            return self._render_code_error(atom, content)
        elif atom_type in ['TOOL', 'JSON']:
            return self._render_tool_json(atom, content)
        elif atom_type in ['NL', 'FACT', 'PLAN', 'META']:
            return self._render_natural_language(content)
        else:
            # Generic fallback with content
            return self._render_generic(content)
    
    def _render_code_error(self, atom: Dict[str, Any], content: str) -> str:
        """Render CODE/ERROR atoms with function signature + context."""
        
        # Extract function signature if present
        function_pattern = r'(def\s+\w+\([^)]*\)|function\s+\w+\([^)]*\)|class\s+\w+)'
        signature_match = re.search(function_pattern, content)
        
        if signature_match:
            signature = signature_match.group(1)
            # Find the line and include some context
            lines = content.split('\n')
            sig_line_idx = None
            for i, line in enumerate(lines):
                if signature in line:
                    sig_line_idx = i
                    break
            
            if sig_line_idx is not None:
                # Include signature + 3-5 lines of context
                start_idx = max(0, sig_line_idx - 1)
                end_idx = min(len(lines), sig_line_idx + 5)
                context_lines = lines[start_idx:end_idx]
                rendered = signature + '\n' + '\n'.join(context_lines)
            else:
                rendered = signature + '\n' + content[:200]
        else:
            # No clear function, take first meaningful lines
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            rendered = '\n'.join(lines[:5])
        
        # Ensure minimum content
        if len(rendered) < 50:
            rendered = content[:300]
            
        return self._truncate_to_tokens(rendered, self.max_passage_tokens)
    
    def _render_tool_json(self, atom: Dict[str, Any], content: str) -> str:
        """Render TOOL/JSON atoms as key-value summary."""
        
        try:
            import json
            if content.strip().startswith('{'):
                data = json.loads(content)
                # Create key-value summary
                summary_parts = []
                for k, v in list(data.items())[:5]:  # Top 5 keys
                    if isinstance(v, (str, int, float, bool)):
                        summary_parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        summary_parts.append(f"{k}: [{len(v)} items]")
                    else:
                        summary_parts.append(f"{k}: {type(v).__name__}")
                
                rendered = "JSON: " + "; ".join(summary_parts)
            else:
                # Tool output, use first few lines
                lines = content.split('\n')[:3]
                rendered = "Tool output: " + "; ".join(lines)
                
        except:
            # JSON parsing failed, use raw content
            rendered = "Data: " + content[:200]
        
        return self._truncate_to_tokens(rendered, self.max_passage_tokens)
    
    def _render_natural_language(self, content: str) -> str:
        """Render NL/FACT/PLAN/META as trimmed sentence blocks."""
        
        # Split into sentences (simple)
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, content)
        
        # Take first 2-3 complete sentences
        rendered_sentences = []
        total_length = 0
        
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if sentence and total_length + len(sentence) < 400:
                rendered_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        rendered = '. '.join(rendered_sentences)
        if rendered and not rendered.endswith('.'):
            rendered += '.'
            
        # Ensure minimum content
        if len(rendered) < 30:
            rendered = content[:300]
            
        return self._truncate_to_tokens(rendered, self.max_passage_tokens)
    
    def _render_generic(self, content: str) -> str:
        """Generic content rendering."""
        
        # Clean and truncate
        content = content.strip()
        
        # Take meaningful first portion
        if len(content) > 500:
            # Try to break at sentence boundary
            truncated = content[:500]
            last_sentence = truncated.rfind('.')
            if last_sentence > 100:
                content = truncated[:last_sentence + 1]
            else:
                content = truncated
        
        return self._truncate_to_tokens(content, self.max_passage_tokens)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text
    
    def prepare_ce_query(self, query: str) -> str:
        """Prepare and cap query for CE input."""
        
        # Truncate query to prevent it being cut off
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        
        if len(query_tokens) > self.query_cap_tokens:
            truncated_tokens = query_tokens[:self.query_cap_tokens]
            query = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return query.strip()
    
    def validate_ce_input(self, query: str, passage: str) -> Dict[str, Any]:
        """Validate CE input meets quality requirements."""
        
        # Tokenize both parts
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        passage_tokens = self.tokenizer.encode(passage, add_special_tokens=False)
        
        validation = {
            "valid": True,
            "issues": [],
            "query_tokens": len(query_tokens),
            "passage_tokens": len(passage_tokens),
            "total_tokens": len(query_tokens) + len(passage_tokens) + 3  # +3 for [CLS] and 2x [SEP]
        }
        
        # Check query length
        if len(query_tokens) < 8:
            validation["valid"] = False
            validation["issues"].append(f"Query too short: {len(query_tokens)} tokens < 8")
        
        # Check passage length  
        if len(passage_tokens) < 64:
            validation["valid"] = False
            validation["issues"].append(f"Passage too short: {len(passage_tokens)} tokens < 64")
        
        # Check for placeholder patterns
        if re.match(r'^Document\s+\w+', passage):
            validation["valid"] = False
            validation["issues"].append("Placeholder pattern detected in passage")
        
        # Check total length fits in model
        if validation["total_tokens"] > 512:
            validation["valid"] = False
            validation["issues"].append(f"Total tokens {validation['total_tokens']} > 512")
        
        return validation


class CEGuards:
    """Hard guards to prevent CE regression."""
    
    def __init__(self):
        self.min_std = 0.10
        self.min_range = 0.30
        self.min_passage_tokens = 32
        self.safe_mode_active = False
    
    def validate_batch_input(self, query: str, passages: list) -> Dict[str, Any]:
        """Validate CE batch input before scoring."""
        
        renderer = ContentRenderer()
        issues = []
        
        for i, passage in enumerate(passages):
            # Check for placeholder patterns
            if re.match(r'^Document\s+\w+', passage):
                issues.append(f"Passage {i}: Placeholder pattern detected")
            
            # Check minimum length
            passage_tokens = len(renderer.tokenizer.encode(passage, add_special_tokens=False))
            if passage_tokens < self.min_passage_tokens:
                issues.append(f"Passage {i}: Only {passage_tokens} tokens < {self.min_passage_tokens}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "eval_ok": len(issues) == 0
        }
    
    def validate_score_variance(self, logits: list) -> Dict[str, Any]:
        """Validate CE output shows proper variance."""
        
        import numpy as np
        
        if len(logits) < 2:
            return {"valid": True, "reason": "Insufficient samples for variance check"}
        
        std = np.std(logits)
        score_range = max(logits) - min(logits)
        
        validation = {
            "std": std,
            "range": score_range,
            "min_std": self.min_std,
            "min_range": self.min_range,
            "valid": std >= self.min_std and score_range >= self.min_range,
            "issues": []
        }
        
        if std < self.min_std:
            validation["issues"].append(f"Score std {std:.3f} < {self.min_std}")
        
        if score_range < self.min_range:
            validation["issues"].append(f"Score range {score_range:.3f} < {self.min_range}")
        
        if not validation["valid"]:
            validation["should_use_safe_mode"] = True
            
        return validation
    
    def enable_safe_mode(self):
        """Enable CE safe mode fallback."""
        self.safe_mode_active = True
        logger.warning("CE safe mode enabled due to validation failures")
    
    def is_safe_mode_active(self) -> bool:
        """Check if safe mode is active."""
        return self.safe_mode_active