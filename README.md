# Autonomous Teaching Assistant

An AI-powered teaching assistant that answers questions by intelligently retrieving and synthesizing information from textbook PDFs. The system uses retrieval-augmented generation (RAG) with semantic search to provide accurate, context-aware answers based on textbook content.

## Overview

The Autonomous Teaching Assistant processes PDF textbooks, extracts their content and table of contents, creates vector embeddings for semantic search, and uses an intelligent agent to answer questions by:

1. Selecting the most relevant chapters based on the question
2. Retrieving relevant text chunks using semantic search
3. Synthesizing comprehensive answers from the retrieved content
4. Self-evaluating the quality and completeness of answers

## Features

- **PDF Processing**: Automatically extracts text, table of contents, and page information from PDF textbooks
- **Semantic Search**: Uses sentence transformers and FAISS for efficient vector-based similarity search
- **Intelligent Chapter Selection**: Leverages GPT models to identify the most relevant chapters for a given question
- **Context-Aware Answers**: Synthesizes answers using only textbook content, maintaining accuracy and relevance
- **Self-Evaluation**: Automatically evaluates answer quality and completeness before returning results
- **Chapter Filtering**: Filters search results to focus on relevant chapters for improved accuracy

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd autonomous-ta
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
autonomous-ta/
├── data/
│   └── raw/              # Place PDF textbooks here
├── notebooks/
│   └── Demo.ipynb        # Example usage notebook
├── src/
│   └── autonomous_ta/
│       ├── __init__.py
│       ├── agent.py      # Main TextbookAgent class
│       ├── load_data.py  # PDF parsing and chunking utilities
│       └── vector_db.py  # Vector database implementation
├── requirements.txt
└── README.md
```

## Usage

### Step 1: Prepare Your Textbook

Place your PDF textbook in the `data/raw/` directory. The system will automatically process any PDF files found in this directory.

### Step 2: Parse the Textbook

Run the data loading script to extract and chunk the textbook content:

```python
from src.autonomous_ta.load_data import parse_book

parse_book()
```

This will:
- Extract text from all pages
- Extract the table of contents
- Chunk the text into manageable segments
- Save the processed data as a JSON file (e.g., `textbook.pdf.json`)

### Step 3: Initialize the Agent

```python
from src.autonomous_ta.agent import TextbookAgent

agent = TextbookAgent(model="gpt-4o-mini")
```

The agent will automatically:
- Load the processed textbook data
- Build the vector index using sentence transformers
- Prepare for querying

### Step 4: Ask Questions

```python
question = "What is regression and why is it useful? What are the disadvantages of using regression?"
answer, chunks = agent.answer_question(question, top_k=5)
```

The `answer_question` method returns:
- `answer`: The synthesized answer string
- `chunks`: List of retrieved text chunks with metadata (chapter, page, book)

### Example Output

```
Consulting Chapters - ['12.3 The Regression Equation', '12.5 Prediction', 'Bivariate Data, Linear Regression, and Univariate Data']
Self-evaluation: YES
=== ANSWER ===
Regression is a statistical process used to determine the relationship between two or more variables...

=== CHUNKS RETRIEVED ===
12.3 The Regression Equation (Page 638): 
12.3 The Regression Equation (Page 640): 
...
```

## API Reference

### TextbookAgent

Main class for querying textbooks and generating answers.

#### Methods

##### `__init__(model="gpt-4o-mini")`
Initialize the agent with a specified GPT model.

**Parameters:**
- `model` (str): OpenAI model name (default: "gpt-4o-mini")

##### `answer_question(question, top_k=5, model="gpt-4o-mini")`
Answer a question using textbook content.

**Parameters:**
- `question` (str): The question to answer
- `top_k` (int): Number of top chunks to retrieve (default: 5)
- `model` (str): OpenAI model for synthesis (default: "gpt-4o-mini")

**Returns:**
- `tuple`: (answer_string, chunks_list) or ("", "") if no answer found

##### `choose_chapters(question, available_chapters)`
Select relevant chapters for a question using GPT.

**Parameters:**
- `question` (str): The question to answer
- `available_chapters` (list): List of available chapter titles

**Returns:**
- `list`: Selected chapter titles

##### `synthesize_answer(question, chunks)`
Generate an answer from retrieved chunks.

**Parameters:**
- `question` (str): The question to answer
- `chunks` (list): List of text chunks with metadata

**Returns:**
- `str`: Synthesized answer

##### `evaluate_answer(question, answer)`
Evaluate if an answer is complete and well-supported.

**Parameters:**
- `question` (str): The original question
- `answer` (str): The generated answer

**Returns:**
- `str`: "YES" or "NO"

### VectorDB

Vector database for semantic search over textbook content.

#### Methods

##### `__init__(model_name="all-MiniLM-L6-v2")`
Initialize the vector database.

**Parameters:**
- `model_name` (str): Sentence transformer model name

##### `build_index()`
Build the FAISS index from processed textbook JSON files.

##### `query(question, top_k=5, chapter_keywords=None)`
Query the vector database for relevant chunks.

**Parameters:**
- `question` (str): Query text
- `top_k` (int): Number of results to return
- `chapter_keywords` (list, optional): Filter results to specific chapters

**Returns:**
- `list`: List of dictionaries with chunk_text, chapter, page, book, and distance

##### `list_chapters()`
Get a sorted list of all available chapters.

**Returns:**
- `list`: Sorted list of chapter titles

## Configuration

### Model Selection

You can customize the models used:

- **Sentence Transformer**: Change `model_name` in `VectorDB.__init__()` (default: "all-MiniLM-L6-v2")
- **OpenAI Model**: Change `model` parameter in `TextbookAgent.__init__()` (default: "gpt-4o-mini")

### Chunking Parameters

In `load_data.py`, you can adjust:
- `max_tokens`: Maximum tokens per chunk (default: 300)

## Dependencies

- `dotenv`: Environment variable management
- `faiss-cpu`: Vector similarity search
- `numpy`: Numerical operations
- `openai`: OpenAI API client
- `PyMuPDF` (fitz): PDF parsing
- `sentence-transformers`: Text embeddings

## Limitations

- Currently processes one textbook at a time (the first JSON file found)
- Requires PDFs with a proper table of contents for chapter detection
- Answers are limited to content available in the processed textbook
- Requires an active internet connection for OpenAI API calls

## Future Improvements

- Support for multiple textbooks simultaneously
- Improved chapter detection for PDFs without table of contents
- Caching mechanisms for repeated queries
- Support for additional file formats (e.g., EPUB, DOCX)
- Web interface for easier interaction
- Batch question processing
- Export answers to various formats

## License

See LICENSE file for details.

## Acknowledgments

This project uses:
- OpenAI GPT models for intelligent question answering
- Sentence Transformers for semantic embeddings
- FAISS for efficient vector search
- PyMuPDF for PDF processing
