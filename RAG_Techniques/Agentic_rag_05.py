# 1. Dependencies Installation
!pip install pandas langchain langchain-community sentence-transformers faiss-cpu "transformers[agents]"
!pip install "git+https://github.com/huggingface/transformers.git#egg=transformers[agents]"

# 2. Import Statements
import pandas as pd
import datasets
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import HfApiEngine, ReactJsonAgent
from huggingface_hub import InferenceClient, notebook_login
import logging
import os
from langchain_groq import ChatGroq
from typing import List, Dict
from transformers.agents.llm_engine import MessageRole, get_clean_message_list

# 3. Logging Configuration
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# 4. Knowledge Base Loading and Processing
def load_and_process_knowledge_base():
    # Load dataset
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    
    # Convert to Document objects
    source_docs = [
        Document(page_content = doc["text"], metadata = {"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]
    logger.info(f"Loaded {len(source_docs)} documents from the knowledge base")
    
    # Initialize tokenizer and text splitter
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size = 200,
        chunk_overlap = 20,
        add_start_index = True,
        strip_whitespace = True,
        separators = ["\n\n", "\n", ".", " ", ""],
    )
    
    # Process documents
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)
                
    logger.info(f"Processed {len(docs_processed)} unique document chunks")
    return docs_processed

# 5. Vector Database Creation
def create_vector_database(docs_processed):
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.from_documents(
        documents = docs_processed,
        embedding = embedding_model,
        distance_strategy = DistanceStrategy.COSINE,
    )
    logger.info("Vector database created successfully")
    return vectordb

# 6. Retriever Tool Implementation
class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves documents from the knowledge base."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. Use affirmative form rather than questions.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vectordb.similarity_search(query, k=7)
        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# 7. Groq Engine Implementation
class GroqEngine:
    def __init__(self, model_name = "mixtral-8x7b-32768"):
        self.model_name = model_name
        self.client = ChatGroq(api_key="gsk_👀👀")  # Replace with your API key

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(
            messages, 
            role_conversions={MessageRole.TOOL_RESPONSE: MessageRole.USER}
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.1,
        )
        return response.choices[0].message.content

# 8. RAG Implementation Functions
def run_agentic_rag(question: str, agent) -> str:
    enhanced_question = f"""Using the information from your knowledge base, give a comprehensive answer to:
{question}"""
    return agent.run(enhanced_question)

def run_standard_rag(question: str, retriever_tool) -> str:
    context = retriever_tool(question)
    prompt = f"""Given the question and supporting documents below, give a comprehensive answer.
Question: {question}
{context}"""
    messages = [{"role": "user", "content": prompt}]
    reader_llm = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
    return reader_llm.chat_completion(messages).choices[0].message.content

# 9. Main Execution
if __name__ == "__main__":
    # Initialize components
    docs_processed = load_and_process_knowledge_base()
    vectordb = create_vector_database(docs_processed)
    retriever_tool = RetrieverTool(vectordb)
    llm_engine = GroqEngine()
    agent = ReactJsonAgent(tools=[retriever_tool], 
                llm_engine=llm_engine, 
                max_iterations=4, 
                verbose=2)
    
    # Let's test this with question
    question = "How can I push a model to the Hub?"
    agentic_answer = run_agentic_rag(question, agent)
    standard_answer = run_standard_rag(question, retriever_tool)
    
    print("Agentic RAG Answer:", agentic_answer)
    print("\nStandard RAG Answer:", standard_answer)