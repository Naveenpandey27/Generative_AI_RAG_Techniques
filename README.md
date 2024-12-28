# Generative_AI_RAG_Techniques
This folder will contains all the code to different RAG Techniques.


# RAG Techniques: Implementing 5 Effective Methods

## Introduction

Retrieval-Augmented Generation (RAG) enhances language model outputs by integrating external knowledge, resulting in more accurate and contextually aware responses. This project demonstrates the implementation of five effective RAG techniques to improve generative model performance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Chunking Documents](#1-chunking-documents)
  - [2. Creating Embeddings](#2-creating-embeddings)
  - [3. Storing in Vector Database and Getting Response](#3-storing-in-vector-database-and-getting-response)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-techniques.git

2. **Navigate to the project directory:**
  ```bash
cd rag-techniques

3. **Install the required dependencies:**
  ```bash
pip install -r requirements.txt

Usage
1. Chunking Documents
Break large documents into manageable chunks to facilitate efficient processing.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def data_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def chunk_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Example usage
pdf_path = '/path/to/your/document.pdf'
documents = data_loader(pdf_path)
chunks = chunk_document(documents)

** Pro Tip: Choose your chunk size carefully! Too large, and you might miss specific details. Too small, and you might lose context. A good starting point is 500–1000 characters with some overlap.**

2. Creating Embeddings
Convert text chunks into numerical representations to capture semantic meaning.


from sentence_transformers import SentenceTransformer

def create_embeddings_and_store_db(chunks):
    model_name = 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Create embeddings
chunk_embeddings = create_embeddings_and_store_db(chunks)

3. Storing in Vector Database and Getting Response
Store embeddings in a vector database for efficient retrieval and generate responses based on user queries.

def create_qa_retriever(vectorstore):
    prompt = ChatPromptTemplate.from_template("""
    Please answer the following question using only the information provided in the context below.

    1. Think through the details step by step before crafting your response.
    2. Deliver a comprehensive and well-structured answer.

    <context>
    {context}
    </context>

    Question: {input}""")

    llm = ChatGroq(model="mixtral-8x7b-32768")
    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# Example usage
qa_retriever = create_qa_retriever(chunk_embeddings)
query = "Your question here"
response = qa_retriever.invoke({"input": query})
print(f"Q: {query}")
print(f"A: {response['answer']}")


Features
1. **Document Chunking: Efficiently splits large documents into smaller, manageable chunks.**
2. **Embedding Creation: Generates embeddings that capture the semantic meaning of text chunks.**
3. **Vector Database Storage: Stores embeddings in a vector database for quick retrieval.**
4. **Question-Answer Retrieval: Retrieves relevant information based on user queries and generates accurate responses.**


**Dependencies**
Python 3.x
LangChain
SentenceTransformers
FAISS
PyPDFLoader
ChatGroq

**Configuration**
Ensure that you have the necessary API keys and configurations set up for the services used in this project. For example:

export GROQ_API_KEY='your_groq_api_key'

**Troubleshooting**
If you encounter issues:

Ensure all dependencies are installed correctly.
Verify that API keys and configurations are set up properly.

Consult the issues section for similar problems and solutions.

**Contributors**
**Naveen Pandey** - Author of the original article.
