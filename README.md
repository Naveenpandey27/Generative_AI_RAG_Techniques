# Generative_AI_RAG_Techniques
This folder will contains all the code to different RAG Techniques.


# RAG Techniques: Implementing 5 Effective Methods

A comprehensive implementation guide for various Retrieval-Augmented Generation (RAG) techniques.

## Introduction

Retrieval-Augmented Generation (RAG) enhances language model outputs by integrating external knowledge, resulting in more accurate and contextually aware responses. This repository demonstrates the implementation of effective RAG techniques to improve generative model performance.

## Table of Contents

- [Installation](#installation)
- [Implementation Guide](#implementation-guide)
  - [Document Processing](#document-processing)
  - [Embedding Creation](#embedding-creation)
  - [Vector Database Integration](#vector-database-integration)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributors](#contributors)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-techniques.git

# Navigate to project directory
cd rag-techniques

# Install dependencies
pip install -r requirements.txt
```

## Implementation Guide

### Document Processing

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class RAGPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.llm = ChatGroq(model="mixtral-8x7b-32768")
    
    def load_documents(self, pdf_path: str):
        """Load documents from PDF file."""
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into overlapping chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, chunks):
        """Create embeddings and store in FAISS vector database."""
        return FAISS.from_documents(chunks, self.embeddings)
    
    def create_qa_chain(self, vectorstore):
        """Create retrieval QA chain."""
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question using only the provided context.
        
        Context:
        {context}
        
        Question: {input}
        
        Instructions:
        1. Use only information from the provided context
        2. If the context doesn't contain enough information, say so
        3. Provide specific references when possible
        4. Structure your response clearly and logically
        """)
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, document_chain)
    
    def process_query(self, chain, query: str):
        """Process a query through the RAG pipeline."""
        response = chain.invoke({"input": query})
        return {
            "query": query,
            "answer": response.get("answer", "No answer found"),
            "source_documents": response.get("source_documents", [])
        }

# Usage Example
def main():
    # Initialize the RAG pipeline
    rag = RAGPipeline()
    
    # Load and process documents
    documents = rag.load_documents("path/to/your/document.pdf")
    chunks = rag.chunk_documents(documents)
    vectorstore = rag.create_vector_store(chunks)
    
    # Create QA chain
    qa_chain = rag.create_qa_chain(vectorstore)
    
    # Process queries
    query = "What are the key components of a RAG system?"
    result = rag.process_query(qa_chain, query)
    print(f"Q: {result['query']}")
    print(f"A: {result['answer']}")

if __name__ == "__main__":
    main()
```

## Features

- Document chunking with configurable size and overlap
- Embedding generation using HuggingFace models
- FAISS vector database integration
- Configurable retrieval chain
- Comprehensive error handling
- Structured response format

## Dependencies

Create a `requirements.txt` file with the following:

```
langchain>=0.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
PyPDF2>=3.0.0
transformers>=4.30.0
torch>=2.0.0
langchain-groq>=0.2.2
```

## Configuration

Set up required environment variables:

```bash
export GROQ_API_KEY='your_groq_api_key'
```

## Best Practices

### Document Chunking
- Recommended chunk size: 500-1000 characters
- Overlap: 10-20% of chunk size
- Adjust based on content type and complexity

### Vector Database
- Regular index maintenance
- Consider GPU acceleration for large datasets
- Implement proper error handling and backup strategies

### Query Processing
- Implement rate limiting for API calls
- Cache frequently accessed results
- Monitor and log response times

## Troubleshooting

### Common Issues and Solutions

1. Memory Issues
   - Reduce chunk size
   - Implement batch processing
   - Use streaming for large responses

2. Performance Issues
   - Optimize vector store configuration
   - Adjust retrieval parameters
   - Consider using GPU acceleration

3. Quality Issues
   - Fine-tune chunk size and overlap
   - Adjust similarity search parameters
   - Refine prompt templates


## Contributors

- Naveen Pandey - Original Author

  Read full article here: https://medium.com/ai-in-plain-english/rag-techniques-part-1-of-5-implementing-5-effective-methods-a92c58399875
  

---
