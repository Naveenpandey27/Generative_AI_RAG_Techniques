import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ['GROQ_API_KEY'] = "gsk_🙈🙈🙈"

def data_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def chunk_document(documents):
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings_and_store_db(chunks):
    # Initialize the embedding model
    model_name = 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create VectorStore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def create_qa_retriever(vectorstore):

    prompt = ChatPromptTemplate.from_template("""
    Please answer the following question using only the information provided in the context below.
    
    1. Think through the details step by step before crafting your response.
    2. Deliver a comprehensive and well-structured answer.
    
    **Note: If the user finds your answer exceptionally helpful, you will receive a tip of $1000.
    <context>
    {context}
    </context>

    Question: {input}""")    
    
    # Initialize the QA retriever with the ChatGroq
    llm = ChatGroq(model="mixtral-8x7b-32768") 
    retriever=vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrival_chain = create_retrieval_chain(retriever, document_chain)
    return retrival_chain

# provide your pdf
pdf_path = '/content/Rich-Dad-Poor-Dad_removed.pdf'
documents = data_loader(pdf_path)

# Split documents into chunks
chunks = chunk_document(documents)

# Create embeddings and store in FAISS vector store
vectorstore = create_embeddings_and_store_db(chunks)

# Create QA retriever with GroQ model
qa_retriever = create_qa_retriever(vectorstore)

# Test Question-Answering
query = "What are the key financial lessons from the book?"
response = qa_retriever.invoke({"input": query})

print(f"Q: {query}")
print(f"A: {response['answer']}")