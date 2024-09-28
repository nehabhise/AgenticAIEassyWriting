
from tavily import TavilyClient
import autogen
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
import requests
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
from typing_extensions import Annotated
import json
import os
import sys
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from config import Config
from autogen import AssistantAgent


# Configuring LLM to OpenAI
config_list = [
    {
        "model": Config.OPENAI_DEPLOYMENT_NAME,
        "api_key": Config.AZURE_OPENAI_KEY,
        "base_url": Config.AZURE_OPENAI_ENDPOINT,
        "api_type": "azure",
        "api_version": Config.OPENAI_API_VERSION
    }
]
# Function to check termination message
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# Configuration for the language model
llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0.8, "seed": 1234}

# 1. create an RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            os.path.join(os.path.abspath(""), "..", "website", "docs"),
        ],
        "custom_text_types": ["non-existent-type"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        # "client": chromadb.PersistentClient(path="/tmp/chromadb"),  # deprecated, use "vector_db" instead
        "vector_db": "chroma",  # to use the deprecated `client` parameter, set to None and uncomment the line above
        "overwrite": False,  # set to True if you want to overwrite an existing collection
        "get_or_create": True,  # set to False if don't want to reuse an existing collection
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)

# Function to perform web search using Tavily
def web_search_tool(query):
    try:
        tavily_client = TavilyClient(api_key="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")
        context = tavily_client.search(query=query, max_results=2)
        val = context["results"]
        return ', '.join([d['content'] for d in val])
    except requests.exceptions.RequestException as e:
        print(f"Web search error: {e}")
        return ""


decisionagent = AssistantAgent(
    "decision_assistant",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    code_execution_config=False,  # Turn off code execution, by default it is off.
    human_input_mode="NEVER",  # Never ask for human input.
)


# Function to get decision from LLM via RetrieveAssistantAgent
def get_decision_from_llm(decisionagent, response_summary):
    print("response_summary",response_summary)
    decision_prompt = f"""
    Based on the following response, determine if the information provided precisely answers the query or if a fallback action is required. 
    
    Consider the following negative scenarios:
    1. The response is irrelevant or off-topic.
    2. The response is incomplete or lacks critical details.
    3. The response is inaccurate or incorrect.
    4. The response does not directly answer the query or is confusing.
    5. The response doesn't gives correct and precised information.
    
    Response: {response_summary if response_summary else 'None'}

    If any of the negative scenarios apply or if a fallback action, such as a web search, is necessary to find additional information, reply with "websearch".
    If the response adequately addresses the query and none of the negative scenarios apply, reply with "response".
    
    """

    # try:
    recipientval = "user"  # Adjust this based on your system's requirements
    decision_result = decisionagent.generate_reply(messages=[{"content": decision_prompt, "role": "user"}])
    print(decision_result)
    decision = decision_result.strip()
    return decision


def initiate_chat_with_fallback(agent, message_generator, problem, n_results):
    chat_result = ragproxyagent.initiate_chat(agent, message=message_generator, problem=problem, n_results=n_results)

    if chat_result is None or chat_result.summary is None:
        return {"source": "none", "results": "No response from the agent"}

    # Use the decision-making function to decide the next action
    action = get_decision_from_llm(decisionagent, chat_result.summary)

    if action == "websearch":
        # Perform web search as a fallback
        web_results = web_search_tool(problem)
        return {
            "source": "websearch",
            "results": web_results
        }
    else:
        return {
            "source": "db",
            "results": chat_result.summary
        }

def autogen_rag_method(qa_problem):
    # Implement your processing logic here
    # For demonstration, let's assume it simply echoes back the input with some modifications
    chat_result = initiate_chat_with_fallback(assistant, ragproxyagent.message_generator, qa_problem, 30)
    if isinstance(chat_result, dict) and chat_result.get("source") == "websearch":
        print("-----------Web Search------------")
        processed_output = chat_result['results']
    else:
        print("-----------DB------------")
        processed_output = chat_result['results']
    print(processed_output)
    return processed_output






