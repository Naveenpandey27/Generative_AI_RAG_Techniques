from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever
import os
import time
from typing import List, Dict, Any
from pathlib import Path

class SpeculativeRAG:
    def __init__(self, api_key: str):
        """
        Initialize the Speculative RAG system.
        
        Args:
            api_key (str): Groq API key
        """
        self.embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        self.llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        self.vector_store = None
        self.bm25_retriever = None
        self.documents = []

    def load_documents(self, directory_path: str) -> None:
        """
        Load documents from a directory.
        
        Args:
            directory_path (str): Path to directory containing documents
        """
        # Load documents
        loader = DirectoryLoader(
            directory_path,
            glob = "**/*.txt",
            loader_cls = TextLoader
        )
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        self.documents = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            self.documents,
            self.embeddings
        )
        
        # Create BM25 retriever for lightweight retrieval
        texts = [doc.page_content for doc in self.documents]
        self.bm25_retriever = BM25Retriever.from_texts(texts)

    def lightweight_retrieval(self, query: str, k: int = 3) -> List[str]:
        """
        Perform quick BM25-based retrieval.
        
        Args:
            query (str): User query
            k (int): Number of documents to retrieve
            
        Returns:
            List[str]: Retrieved documents
        """
        return self.bm25_retriever.get_relevant_documents(query, k = k)

    def comprehensive_retrieval(self, query: str, k: int = 10) -> List[str]:
        """
        Perform comprehensive vector-based retrieval.
        
        Args:
            query (str): User query
            k (int): Number of documents to retrieve
            
        Returns:
            List[str]: Retrieved documents
        """
        time.sleep(2)  # Simulate longer processing time
        return self.vector_store.similarity_search(query, k=k)

    def get_speculative_answer(self, query: str) -> Dict[str, Any]:
        """
        Get both speculative and comprehensive answers.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Dictionary containing both speculative and final answers
        """
        # Get speculative answer using lightweight retrieval
        speculative_docs = self.lightweight_retrieval(query)
        speculative_qa = RetrievalQA.from_chain_type(
            llm = self.llm,
            retriever = self.bm25_retriever,
            return_source_documents = True
        )
        speculative_result = speculative_qa({"query": query})
        
        # Get comprehensive answer using vector store
        comprehensive_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever = self.vector_store.as_retriever(),
            return_source_documents = True
        )
        comprehensive_result = comprehensive_qa({"query": query})
        
        return {
            "speculative_answer": speculative_result["result"],
            "final_answer": comprehensive_result["result"],
            "speculative_sources": speculative_result["source_documents"],
            "final_sources": comprehensive_result["source_documents"]
        }

def main():
    # Example usage
    rag = SpeculativeRAG(groq_api_key = os.getenv('GROQ_API_KEY'))
    
    # Load documents
    docs_path = "/content/Docs"
    rag.load_documents(docs_path)
    
    # Example query
    query = "What is deep learning?"
    results = rag.get_speculative_answer(query)
    
    # Print results
    print("Speculative Answer:", results["speculative_answer"])
    print("\nFinal Answer:", results["final_answer"])
    print("\nSpeculative Sources:", len(results["speculative_sources"]))
    print("Final Sources:", len(results["final_sources"]))

if __name__ == "__main__":
    main()