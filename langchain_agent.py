from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from config import Config  # Import the Config class
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
_ = load_dotenv()
tool = TavilySearchResults(max_results=2)
# Define TypedDict
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Initialize models using configuration values
model = AzureChatOpenAI(
    deployment_name=Config.OPENAI_DEPLOYMENT_NAME,
    openai_api_version=Config.OPENAI_API_VERSION,
    temperature=Config.LANGCHAIN_TEMPERATURE
)


# Define prompts
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""
REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""
RESEARCH_PLAN = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 1 query max.

{format_instructions}
"""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 1 query max.

{format_instructions}
"""

# Define parsers and prompt templates
class Queries(BaseModel):
    """queries to be searched in json format"""
    queries: List[str]

parser = JsonOutputParser(pydantic_object=Queries)
RESEARCH_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(RESEARCH_PLAN, partial_variables={"format_instructions": parser.get_format_instructions()})]
)

# Define nodes
def plan_node(state: AgentState):
    task = state["task"]
    messages = [SystemMessage(content=PLAN_PROMPT),
                HumanMessage(content=task)]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    message = [SystemMessage(content=RESEARCH_PLAN.format(format_instructions=parser.get_format_instructions())),
               HumanMessage(content=state["task"])]

    research_model = model | parser
    queries = research_model.invoke(message)
    print("queries",queries)

    content = state["content"] or []
    try:
        for q in queries.queries:
            response = tool.invoke(q)
            for r in response:
                content.append(r['content'])
        print("content",content)
        return {"content": content}
    except Exception as e:
        return {"content": content}

def generate_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def critique_node(state: AgentState):
    message = [SystemMessage(content=RESEARCH_CRITIQUE_PROMPT.format(format_instructions=parser.get_format_instructions())),
               HumanMessage(content=state["critique"])]

    research_model = model | parser
    queries = research_model.invoke(message)
    content = state['content'] or []

    try:
        for q in queries.queries:
            response = tool.invoke(q)
            for r in response:
                content.append(r['content'])

        return {"content": content}
    except Exception as e:
        return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]: # 2>1:end 
        return "END"
    return "reflect"

# Build the state graph
graph = StateGraph(AgentState)
graph.add_node("planner", plan_node)
graph.add_node("research_plan", research_plan_node)
graph.add_node("generate", generate_node)
graph.add_node("critique_agent", critique_node)
graph.add_node("reflect", reflection_node)
graph.set_entry_point("planner")
graph.add_conditional_edges("generate",
                            should_continue,
                            {
                                "END": END,
                                "reflect": "reflect"
                            })
graph.add_edge("planner", "research_plan")
graph.add_edge("research_plan", "generate")
graph.add_edge("reflect", "critique_agent")
graph.add_edge("critique_agent", "generate")
memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)

def langchain_multiagent_method(input_data:str):
    thread = {"configurable":{"thread_id":"2"}}
    try:
        resp = app.invoke({"task": input_data,
                   "max_revisions": 1,
                   "revision_number": 1}, thread)
        final_state=resp["draft"]
        return final_state
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}"




