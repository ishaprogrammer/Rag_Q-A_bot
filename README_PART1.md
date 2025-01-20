# README for Part_1 Q/A Bot

## Overview
This project implements a Question-Answering (Q/A) bot that extracts financial insights from Profit & Loss (P&L) tables embedded within PDF documents. The bot uses a Retrieval-Augmented Generation (RAG) model combining retrieval techniques with generative language models. Key functionalities include document parsing, content summarization, and answering queries based on extracted financial data.

## Prerequisites
- Python 3.8 or above
- Dependencies (see `requirements.txt`)
- Internet connection for API calls

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
1. Set up API keys:
   - Create a `.env` file in the project root directory with the following:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     UNSTRUCTURED_API_KEY=your_unstructured_api_key
     GROQ_API_KEY=your_groq_api_key
     ```
2. Ensure the `pdf_path` variable points to the correct PDF file location.

## Usage
### 1. Load PDF Document
The bot uses `unstructured.partition.pdf` to partition and extract elements from the provided PDF file:
```python
from unstructured.partition.pdf import partition_pdf
pdf_path = "path_to_your_pdf"
elements = partition_pdf(filename=pdf_path)
```

### 2. Extract and Classify Elements
The document elements are parsed into structured data (tables, text, etc.) and categorized:
```python
from pydantic import BaseModel
class Element(BaseModel):
    type: str
    page_content: Any
categorized_elements = []
```

### 3. Creating Embeddings
Embeddings are generated using the `langchain` library to store vectorized document content:
```python
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Pinecone(...)
```

### 4. Question Answering
The RAG pipeline retrieves relevant data and generates answers:
```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
response = chain.invoke("What is Total non-current assets for March 31, 2024?")
print(response)
```

## Features
- Extract and parse financial statements in PDF format
- Summarize tables and text data using a custom summarization chain
- Retrieve and answer queries using embedded vectorized data

## API Usage
- **Unstructured.io API**: For document parsing
- **OpenAI API**: For generating text embeddings
- **Pinecone API**: For vector database storage and retrieval

## Error Handling
- API errors are caught using `try-except` blocks with meaningful error messages.

## Sample Output
1. Parsed tables and text data.
2. Responses to queries such as:
   - *"What is the Leasehold improvements cost?"*
   - *"Give me information about the revenue."*

## Future Improvements
- Add more robust error handling.
- Include additional document processing strategies.
- Optimize embeddings for larger financial datasets.


## Author
[TASKIN SHAIKH]
[EMAIL : ishashaikh154@gmail.com]

