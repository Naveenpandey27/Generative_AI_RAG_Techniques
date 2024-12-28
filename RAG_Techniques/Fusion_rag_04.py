import os
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.load import dumps, loads
from langchain_groq import ChatGroq

# Set up your api key
os.environ['GROQ_API_KEY'] = 'gsk_😎😎😎'

# Load your data
loader = TextLoader("/content/file.txt")
documents = loader.load()

# Split the data
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter =RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)

# Make splits
splits = text_splitter.split_documents(documents)

# Index
from langchain_community.vectorstores import Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=splits,
                                    embedding=embeddings)

retriever = vectorstore.as_retriever()

template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (5 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_rag_fusion 
    | ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

## Chain for extracting relevant documents

retrieval_chain_rag_fusion = generate_queries | retriever.map()
question = 'what is generative ai and how does it work?'
results = retrieval_chain_rag_fusion.invoke({"question": question})

template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (5 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_rag_fusion 
    | ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

## Chain for extracting relevant documents

retrieval_chain_rag_fusion = generate_queries | retriever.map()
question = 'what is generative ai and how does it work?'
results = retrieval_chain_rag_fusion.invoke({"question": question})

fused_scores = {}
k = 60
for docs in results:
  for rank, doc in enumerate(docs):
    doc_str = dumps(doc)
    if doc_str not in fused_scores:
      fused_scores[doc_str] = 0
    fused_scores[doc_str] += 1 / (rank + k)

reranked_results = [
    (loads(doc), score)
    for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
]

template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

final_rag_chain = (prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context": reranked_results, "question": question})