from dotenv import load_dotenv
from typing import Union, Optional, Literal
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import WebBaseLoader
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import Chroma

_ = load_dotenv()

embedder = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002",
                                    openai_api_key=os.environ["AZURE_OPENAI_KEY"])
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)


# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedder,
)
retriever = vectorstore.as_retriever()
context = "\n\n".join([docs  for docs in vectorstore.get()["documents"]])
llm = AzureChatOpenAI(deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"] ,
                  openai_api_version = os.environ["OPENAI_API_VERSION"], temperature=0.1)




class RouteQuery(BaseModel):
    datasource: Literal["vectorstore, web_search"] = Field(...
                                 , description="Given a user question choose between vector store and web_search")
    
parser =JsonOutputParser(pydantic_object=RouteQuery)
# query router



system = """As a skilled expert in summarization, your task is to distill the essence of documents stored in a vector store. You will be provided with the full content of these documents as context. Your summary must be crafted meticulously so that it can serve a dual purpose: not only to concisely convey the document's main points but also to act as an efficient guide for a Large Language Model (LLM) to route user queries to either a vector database or initiate a web search.

### Instructions:

1. **Understanding the Context**: Begin by thoroughly reading the document to grasp its overall theme, key arguments, and data points.
2. **Clarity and Brevity**: Your summary should be clear, concise, and no longer than a short paragraph. Aim for a summary that encapsulates the core message of the document in a few sentences.
3. **Dual-Purpose Design**: Construct the summary in a way that it not only provides a standalone overview of the document but also includes specific keywords or phrases that can aid an LLM in accurately routing queries related to the document's content.
4. **Formatting**: Structure your summary to start with a brief introduction to the document's main topic, followed by a sentence or two highlighting the most critical points or findings. Conclude with a sentence that ties the main ideas together, ensuring it contains terms crucial for query routing.

{context}
"""


summary_prompt = ChatPromptTemplate.from_messages([
    ("system", system)

])

summary_model = summary_prompt | llm
# query router



system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search.




{format_instruction}
"""


router_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")

]).partial(format_instruction = parser.get_format_instructions() )

router_model = router_prompt | llm | parser
class GradeScore(BaseModel):
    binary_score: str = Field(..., description="Give binary score yes or no basis document relevancy to quetion")

grade_score_parser = JsonOutputParser(pydantic_object=GradeScore)
# grade documents
system = """You are a grader assessing relevance of a retrieved document to a user question.\n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.


{format_instruction}
"""

grade_doc_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved Document: \n\n {document}\n\n Question: {question}\n\n Is the document relevant to the question?")

]).partial(format_instruction = grade_score_parser.get_format_instructions())

grade_doc_llm = grade_doc_prompt | llm | parser

# Re-Write Query

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")])

rewrite_llm = rewrite_prompt | llm | StrOutputParser()

# Generate



prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "/n/n".join([doc for doc in docs])

rag_llm = prompt | llm | StrOutputParser()


# hallucinations grader

class GraderHallucination(BaseModel):
    binary_score: str = Field(..., description="Give binary score yes or no basis answer is grounded or not")

grade_hallu_parser = JsonOutputParser(pydantic_object=GraderHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
     
     {format_instruction}
     """

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts documents \n\n {documents}\n\n Answer: {generation}\n\n Is the answer grounded in the facts?")

]).partial(format_instruction = grade_hallu_parser.get_format_instructions())

hallucination_llm = hallucination_prompt | llm | grade_hallu_parser


# Answer Grader

### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

answer_grade_parser = JsonOutputParser(pydantic_object=GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
     and relevant to the question. \n
     {format_instruction}

     """
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
).partial(format_instruction = answer_grade_parser.get_format_instructions())

answer_grader_llm = answer_prompt | llm | answer_grade_parser


### Search


web_search_tool = TavilySearchResults(k=3)

class AgentState(TypedDict):

    question: str
    transformed_question: str
    generation: str
    documents: list[str]
    filtered_documents: list[str]
def retrieve(state:AgentState):

    print("----RETRIEVE----")
    question = state["question"]
    docs = retriever.invoke(question)

    return {"documents": [doc.page_content for doc in docs]}

def grade_documents(state:AgentState):

    print("----GRADE DOCUMENTS----")
    docs = state["documents"]
    question = state["question"]
    filtered_documents = []
    for doc in docs:
        resp = grade_doc_llm.invoke({"document": doc, "question": question})
        score = resp["binary_score"]

        if score == "yes":
            print("----GRADE: DOCUMENT RELEVANT----")
            filtered_documents.append(doc)

        else:
            print("----GRADE: DOCUMENT NOT RELEVANT----")
            continue

    return {"filtered_documents": filtered_documents}

def generate(state:AgentState):

    print("----GENERATE----")
    question = state["question"]

    docs = state["filtered_documents"] or state["documents"]
    generation = rag_llm.invoke({"question": question, "context": format_docs(docs)})

    return {"generation": generation}

def transform_query(state:AgentState):

    print("----TRANSFORM QUERY----")
    question = state["question"]
    resp = rewrite_llm.invoke({"question": question})

    return {"transformed_question": resp}

def web_search(state:AgentState):

    print("----WEB SEARCH----")
    question = state["question"]
    search_results = web_search_tool.invoke(question)
    try:
        return {"documents": [doc["content"] for doc in search_results]}
    except:
        return {"documents":["No Content"]}

def route_query(state:AgentState):

    print("----ROUTE QUESTION----")
    question = state["question"]
    resp = router_model.invoke({"question": question})

    datasource = resp["datasource"]

    if datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif datasource == "vectorstore":
        print("---ROUTE QUESTION TO VECTOR STORE---")
        return "vectorstore"
    
def decide_to_generate(state:AgentState):

    print("----DECIDE TO GENERATE----")
    question = state["question"]
    filtered_documents = state["filtered_documents"]

    if not filtered_documents:
        print("--DECISION: ALL DOCUMENTS ARE NOT RELEVANT")
        return "transform_query"
    
    else:
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state:AgentState):
    print("---CHECK HALLUCINATION")

    question = state["question"]
    generation = state["generation"]

    if state.get("filtered_documents"):
        filtered_documents = state["filtered_documents"]
    else:
        filtered_documents = state["documents"]
    # filtered_documents = state["filtered_documents"] or state["documents"]

    score = hallucination_llm.invoke({"documents": format_docs(filtered_documents), "generation": generation})

    if score["binary_score"] == "yes":
        print("---HALLUCINATION: GROUNDED---")
        print("---GRADE GENERATION vs QUESTION---")
        print("GENERATION: ", generation)

        score = answer_grader_llm.invoke({"question": question, "generation": generation})
        print("SCORE: ", score)
        if score["binary_score"] == "yes":
            print("---GRADE: ANSWER ADDRESSES QUESTION---")
            return "useful"
        
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    

    
graph = StateGraph(AgentState)

graph.add_node("web_search", web_search)
graph.add_node("generate", generate)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("transform_query", transform_query)
graph.add_conditional_edges(
    START,
    route_query,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve"
    }
)

graph.add_edge("web_search", "generate")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)

graph.add_edge("transform_query", "retrieve")

graph.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query"
    }
)

memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)


def langchain_rag_method(input_date:str):
    thread = {"configurable":{"thread_id":"2"}}
    try:
        thread = {"configurable": {"thread_id": "2"}}
        inputs = {
            "question": input_date
        }
        final_state = app.invoke(inputs, thread)
        print(final_state["generation"])
        return final_state["generation"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}"
