import os
import lancedb
import pprint
from typing import Dict, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults


# Set up your API keys
os.environ['GROQ_API_KEY'] = "gsk_😁😁😁"
os.environ['TAVILY_API_KEY'] = "tvly-🤖🤖🤖"

# Load sample data from web URLs
urls = [
    "https://jalammar.github.io/illustrated-transformer/",
    "https://jalammar.github.io/illustrated-bert/",
    "https://jalammar.github.io/illustrated-retrieval-transformer/",
]

class GraphState(TypedDict):
    """Represents the state of our graph."""
    keys: Dict[str, any]

def setup_retriever():
    # Load documents from URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Set up embeddings
    model_name = 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Connect to LanceDB
    db = lancedb.connect('/tmp/lancedb')

    # Create the vectorstore directly without manual table creation
    # LanceDB will handle the schema internally based on the embeddings
    vectorstore = LanceDB.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        connection=db,
        table_name="corrective_rag"
    )

    return vectorstore.as_retriever()


# Initialize the retriever globally
retriever = setup_retriever()

def retrieve(state):
    """Helper function for retrieving documents"""
    print("*" * 5, " RETRIEVE ", "*" * 5)
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}

def generate(state):
    """Helper function for generating answers"""
    print("*" * 5, " GENERATE ", "*" * 5)
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = ChatPromptTemplate.from_template("""
    Please answer the following question using the provided context:

    Context: {context}

    Question: {question}

    Answer:""")

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})

    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }

def grade_documents(state):
    """Determines whether the retrieved documents are relevant"""
    print("*" * 5, " DOCS RELEVANCE CHECK", "*" * 5)
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    grade_tool_oai = convert_to_openai_tool(grade)
    llm_with_tool = llm.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    parser_tool = PydanticToolsParser(tools=[grade])

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document: {context}
        Here is the user question: {question}
        Give a binary score 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool | parser_tool

    filtered_docs = []
    search = "No"
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score[0].binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}

def transform_query(state):
    """Transform query for better retrieval"""
    print("*" * 5, "TRANSFORM QUERY", "*" * 5)
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = PromptTemplate(
        template="""You are generating questions that are well optimized for retrieval.
        Look at the input and try to reason about the underlying semantic intent / meaning.
        Initial question: {question}
        Formulate an improved question:""",
        input_variables=["question"],
    )

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"documents": documents, "question": better_question}}

def web_search(state):
    """Perform web search using Tavily"""
    print("*" * 5, " WEB SEARCH ", "*" * 5)
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "question": question}}

def decide_to_generate(state):
    """Decide whether to generate answer or transform query"""
    print("*" * 5, " DECIDE TO GENERATE ", "*" * 5)
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        print("*" * 5, " DECISION: TRANSFORM QUERY and RUN WEB SEARCH ", "*" * 5)
        return "transform_query"
    else:
        print("*" * 5, " DECISION: GENERATE ", "*" * 5)
        return "generate"

# Create and compile workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Run example
def run_query(query_text):
    inputs = {"keys": {"question": query_text}}
    print(f"\nProcessing query: {query_text}\n")

    for output in app.stream(inputs):
        for key, value in output.items():
            pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        print("-" * 50)

    print("\nFinal Answer:")
    print("-" * 50)
    pprint.pprint(value["keys"]["generation"])

# Example usage
if __name__ == "__main__":
    run_query("Explain tranformers in simple language?")