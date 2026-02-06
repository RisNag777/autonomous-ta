"""Tests for VectorDB class."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.autonomous_ta.vector_db import VectorDB


@pytest.fixture
def sample_chunks():
    """Create sample chunk data for testing."""
    return [
        {
            "chapter_title": "Chapter 1: Introduction",
            "page_num": 1,
            "chunk_text": "This is a sample text chunk about statistics.",
        },
        {
            "chapter_title": "Chapter 1: Introduction",
            "page_num": 2,
            "chunk_text": "Another chunk of text about data analysis.",
        },
        {
            "chapter_title": "Chapter 2: Methods",
            "page_num": 10,
            "chunk_text": "This chapter discusses various statistical methods.",
        },
    ]


@pytest.fixture
def temp_json_file(sample_chunks, tmp_path):
    """Create a temporary JSON file with sample chunks."""
    json_file = tmp_path / "test_book.pdf.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sample_chunks, f, ensure_ascii=False, indent=2)
    return json_file


def test_vector_db_initialization():
    """Test VectorDB initialization."""
    db = VectorDB()
    assert db.model is not None
    assert db.index is None
    assert db.texts == []
    assert db.metadata == []


def test_list_chapters(temp_json_file, sample_chunks, monkeypatch):
    """Test listing chapters from the vector database."""
    # Mock the DATA_DIR to point to our temp directory
    import src.autonomous_ta.vector_db as vdb_module
    
    original_data_dir = vdb_module.DATA_DIR
    vdb_module.DATA_DIR = temp_json_file.parent
    
    try:
        db = VectorDB()
        db.build_index()
        
        chapters = db.list_chapters()
        expected_chapters = sorted(set(chunk["chapter_title"] for chunk in sample_chunks))
        assert chapters == expected_chapters
    finally:
        vdb_module.DATA_DIR = original_data_dir


def test_query(temp_json_file, monkeypatch):
    """Test querying the vector database."""
    import src.autonomous_ta.vector_db as vdb_module
    
    original_data_dir = vdb_module.DATA_DIR
    vdb_module.DATA_DIR = temp_json_file.parent
    
    try:
        db = VectorDB()
        db.build_index()
        
        results = db.query("statistics", top_k=2)
        assert len(results) <= 2
        assert all("chunk_text" in result for result in results)
        assert all("chapter" in result for result in results)
        assert all("page" in result for result in results)
    finally:
        vdb_module.DATA_DIR = original_data_dir


def test_query_with_chapter_filter(temp_json_file, monkeypatch):
    """Test querying with chapter keyword filtering."""
    import src.autonomous_ta.vector_db as vdb_module
    
    original_data_dir = vdb_module.DATA_DIR
    vdb_module.DATA_DIR = temp_json_file.parent
    
    try:
        db = VectorDB()
        db.build_index()
        
        results = db.query(
            "statistics",
            top_k=5,
            chapter_keywords=["Chapter 1: Introduction"],
        )
        assert len(results) > 0
        assert all("Introduction" in result["chapter"] for result in results)
    finally:
        vdb_module.DATA_DIR = original_data_dir
