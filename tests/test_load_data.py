"""Tests for load_data module."""

import json
import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from src.autonomous_ta.load_data import chunk_text, get_toc, load_pdf


def test_chunk_text():
    """Test text chunking functionality."""
    pages = [
        {"page_num": 1, "text": "Short text."},
        {"page_num": 2, "text": "Another page with more content here."},
    ]
    
    toc = [
        {"level": 1, "title": "Chapter 1", "page_num": 1},
        {"level": 1, "title": "Chapter 2", "page_num": 2},
    ]
    
    chunks = chunk_text(pages, toc, max_tokens=10)
    
    assert len(chunks) > 0
    assert all("chapter_title" in chunk for chunk in chunks)
    assert all("page_num" in chunk for chunk in chunks)
    assert all("chunk_text" in chunk for chunk in chunks)


def test_get_toc():
    """Test table of contents extraction."""
    mock_doc = MagicMock()
    mock_doc.get_toc.return_value = [
        (1, "Chapter 1", 1),
        (2, "Section 1.1", 5),
        (1, "Chapter 2", 10),
    ]
    
    toc = get_toc(mock_doc)
    
    assert len(toc) == 3
    assert toc[0]["level"] == 1
    assert toc[0]["title"] == "Chapter 1"
    assert toc[0]["page_num"] == 1


def test_chunk_text_with_empty_pages():
    """Test chunking with empty pages."""
    pages = [
        {"page_num": 1, "text": ""},
        {"page_num": 2, "text": "Some content"},
    ]
    
    toc = [{"level": 1, "title": "Chapter 1", "page_num": 1}]
    
    chunks = chunk_text(pages, toc)
    # Should handle empty pages gracefully
    assert isinstance(chunks, list)
