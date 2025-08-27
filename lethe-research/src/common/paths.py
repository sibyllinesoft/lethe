"""
Common Path Handling Utilities

Provides standardized path handling patterns to eliminate redundant path operations
throughout the codebase.

Usage:
    from common.paths import ensure_dir, safe_path, validate_file_exists

    # Ensure directory exists
    output_dir = ensure_dir("./results/experiment1")

    # Safe path construction
    config_path = safe_path("config", "experiment.yaml")

    # Validate file exists with clear error
    data_file = validate_file_exists("./data/corpus.json", "corpus data")
"""

import os
from pathlib import Path
from typing import Union, Optional, List
import logging

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def ensure_dir(path: PathLike, exist_ok: bool = True) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        exist_ok: If True, don't raise error if directory already exists
        
    Returns:
        Path object pointing to the directory
        
    Raises:
        FileExistsError: If directory exists and exist_ok=False
        PermissionError: If insufficient permissions to create directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=exist_ok)
    logger.debug(f"Ensured directory exists: {path_obj}")
    return path_obj


def safe_path(*parts: str) -> Path:
    """
    Safely construct path from components.
    
    Args:
        *parts: Path components to join
        
    Returns:
        Path object with components joined safely
    """
    return Path(*parts)


def validate_file_exists(path: PathLike, 
                        description: Optional[str] = None,
                        raise_on_missing: bool = True) -> Optional[Path]:
    """
    Validate that a file exists with clear error messages.
    
    Args:
        path: File path to validate
        description: Human-readable description of the file
        raise_on_missing: If True, raise FileNotFoundError when missing
        
    Returns:
        Path object if file exists, None if missing and raise_on_missing=False
        
    Raises:
        FileNotFoundError: If file doesn't exist and raise_on_missing=True
    """
    path_obj = Path(path)
    
    if path_obj.exists() and path_obj.is_file():
        return path_obj
    
    desc = description or f"file {path}"
    error_msg = f"{desc} not found at: {path_obj.absolute()}"
    
    if raise_on_missing:
        raise FileNotFoundError(error_msg)
    else:
        logger.warning(error_msg)
        return None


def validate_dir_exists(path: PathLike,
                       description: Optional[str] = None,
                       raise_on_missing: bool = True) -> Optional[Path]:
    """
    Validate that a directory exists with clear error messages.
    
    Args:
        path: Directory path to validate
        description: Human-readable description of the directory
        raise_on_missing: If True, raise FileNotFoundError when missing
        
    Returns:
        Path object if directory exists, None if missing and raise_on_missing=False
        
    Raises:
        FileNotFoundError: If directory doesn't exist and raise_on_missing=True
    """
    path_obj = Path(path)
    
    if path_obj.exists() and path_obj.is_dir():
        return path_obj
    
    desc = description or f"directory {path}"
    error_msg = f"{desc} not found at: {path_obj.absolute()}"
    
    if raise_on_missing:
        raise FileNotFoundError(error_msg)
    else:
        logger.warning(error_msg)
        return None


def ensure_file_dir(file_path: PathLike) -> Path:
    """
    Ensure parent directory of a file exists.
    
    Args:
        file_path: Path to file (parent directory will be created)
        
    Returns:
        Original file path as Path object
    """
    file_path_obj = Path(file_path)
    if file_path_obj.parent != file_path_obj:  # Not root
        ensure_dir(file_path_obj.parent)
    return file_path_obj


def get_project_root() -> Path:
    """
    Get project root directory (contains src/ directory).
    
    Returns:
        Path to project root
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).parent
    
    # Search upwards for src directory
    for parent in [current] + list(current.parents):
        if (parent / "src").is_dir():
            return parent
    
    # Fallback: look for common project markers
    current = Path(__file__).parent
    for parent in [current] + list(current.parents):
        markers = ["setup.py", "pyproject.toml", ".git", "requirements.txt"]
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    raise RuntimeError("Could not find project root directory")


def relative_to_project(path: PathLike) -> Path:
    """
    Convert path to be relative to project root.
    
    Args:
        path: Path to convert
        
    Returns:
        Path relative to project root
    """
    path_obj = Path(path)
    project_root = get_project_root()
    
    if path_obj.is_absolute():
        try:
            return path_obj.relative_to(project_root)
        except ValueError:
            # Path is not under project root, return as-is
            return path_obj
    else:
        return path_obj


def find_files(directory: PathLike, 
               pattern: str = "*",
               recursive: bool = True,
               file_type: str = "file") -> List[Path]:
    """
    Find files matching pattern in directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match (e.g., "*.py", "test_*.py")
        recursive: If True, search recursively
        file_type: "file", "dir", or "any"
        
    Returns:
        List of matching Path objects
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []
    
    if recursive:
        glob_pattern = f"**/{pattern}"
        matches = dir_path.glob(glob_pattern)
    else:
        matches = dir_path.glob(pattern)
    
    # Filter by type
    if file_type == "file":
        results = [p for p in matches if p.is_file()]
    elif file_type == "dir":
        results = [p for p in matches if p.is_dir()]
    else:  # "any"
        results = list(matches)
    
    return sorted(results)


def safe_write_text(path: PathLike, content: str, encoding: str = "utf-8") -> Path:
    """
    Safely write text to file, creating parent directories if needed.
    
    Args:
        path: File path to write to
        content: Text content to write
        encoding: Text encoding to use
        
    Returns:
        Path object of written file
    """
    file_path = ensure_file_dir(path)
    file_path.write_text(content, encoding=encoding)
    logger.debug(f"Wrote text to: {file_path}")
    return file_path


def safe_read_text(path: PathLike, 
                   default: Optional[str] = None,
                   encoding: str = "utf-8") -> Optional[str]:
    """
    Safely read text from file with error handling.
    
    Args:
        path: File path to read from
        default: Value to return if file doesn't exist or can't be read
        encoding: Text encoding to use
        
    Returns:
        File contents as string, or default value on error
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.debug(f"File does not exist: {path_obj}")
            return default
        return path_obj.read_text(encoding=encoding)
    except (IOError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read file {path}: {e}")
        return default


def clean_filename(filename: str, replacement: str = "_") -> str:
    """
    Clean filename by replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid characters with
        
    Returns:
        Cleaned filename safe for filesystem use
    """
    import re
    # Remove or replace invalid characters for most filesystems
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    cleaned = re.sub(invalid_chars, replacement, filename)
    
    # Remove trailing dots and spaces
    cleaned = cleaned.rstrip('. ')
    
    # Ensure not empty
    if not cleaned:
        cleaned = "unnamed"
    
    return cleaned


def temp_filename(prefix: str = "tmp", suffix: str = "", directory: Optional[PathLike] = None) -> Path:
    """
    Generate temporary filename.
    
    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        directory: Directory for temp file (default: system temp)
        
    Returns:
        Path to temporary file
    """
    import tempfile
    import uuid
    
    if directory:
        temp_dir = Path(directory)
        ensure_dir(temp_dir)
    else:
        temp_dir = Path(tempfile.gettempdir())
    
    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{unique_id}{suffix}"
    
    return temp_dir / filename


def copy_file_safe(src: PathLike, dst: PathLike, overwrite: bool = False) -> Path:
    """
    Safely copy file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: If True, overwrite existing destination
        
    Returns:
        Destination path
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileExistsError: If destination exists and overwrite=False
    """
    import shutil
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    # Validate source
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Check destination
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst_path}")
    
    # Ensure destination directory exists
    ensure_file_dir(dst_path)
    
    # Copy file
    shutil.copy2(src_path, dst_path)
    logger.debug(f"Copied {src_path} to {dst_path}")
    
    return dst_path


# Export commonly used functions
__all__ = [
    'ensure_dir',
    'safe_path',
    'validate_file_exists',
    'validate_dir_exists', 
    'ensure_file_dir',
    'get_project_root',
    'relative_to_project',
    'find_files',
    'safe_write_text',
    'safe_read_text',
    'clean_filename',
    'temp_filename',
    'copy_file_safe'
]