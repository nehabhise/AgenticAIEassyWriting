import os
import autogen
from autogen import ConversableAgent, Agent
import sys
import json
from config import Config

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

llm_config = {"config_list": config_list}

# Define User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "groupchat",
        "use_docker": False,
    },
    human_input_mode="NEVER",
)

# Define Essay Planning, Researching, Writing, and Critiquing Agents
planner = autogen.AssistantAgent(
    name="planner",
    llm_config=llm_config,
    system_message="""
        You are a professional essay planning expert. Your task is to create a detailed and well-structured outline for an essay on a given topic. 
        Your outline should include a clear thesis statement, section headers, sub-sections, and key points to be addressed. 
        Ensure that the outline logically organizes the content and sets clear objectives for each section.
    """
)

researcher = autogen.AssistantAgent(
    name="researcher",
    llm_config=llm_config,
    system_message="""
        You are a skilled essay researcher tasked with gathering and summarizing relevant information and sources for a given essay topic. 
        Your research should include recent and credible sources, organized according to the essay outline provided. 
        Provide a detailed summary of findings, include proper citations, and ensure the information is relevant to each section of the outline.
    """
)

writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional essay writer with a talent for crafting compelling and coherent narratives. 
        Your task is to write a complete essay based on the provided outline and research data. 
        Ensure that the essay is well-structured, logically organized, and engages the reader while adhering to the outlined objectives. 
        After writing, you will also revise the essay based on feedback received from the critic. Provide a proper essay using all details 
    """
)

critic = autogen.AssistantAgent(
    name="critic",
    llm_config=llm_config,
    system_message="""
        You are a teacher grading an essay submission. 
        Generate critique and recommendations for the user's submission. 
        Provide detailed recommendations, including requests for length, depth, style, etc.. 
        Finally, based on the critique above, 
        suggest a concrete list of actions that the writer should take steps to improve the essay.
    """
)

# Custom Speaker Selection Function
def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat):
    messages = groupchat.messages
    if last_speaker is user:
        return planner
    elif last_speaker is planner:
        return researcher
    elif last_speaker is researcher:
        return writer
    elif last_speaker is writer:
        return critic
    else:
        return "random"

# Group Chat and Manager Creation
groupchat_writer = autogen.GroupChat(
    agents=[user_proxy, planner, researcher, writer, critic], messages=[], max_round=10, speaker_selection_method=custom_speaker_selection_func
)

manager_2 = autogen.GroupChatManager(
    groupchat=groupchat_writer,
    name="Writing_manager",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)

def autogen_multiagent_input(user_input):
    global coding_task
    coding_task = ["Develop a short essay using the given topic " + user_input + "."]
 
    chat_results = user.initiate_chats([{"recipient": manager_2, "message": coding_task[0], "summary_method": "last_msg"}])
    conversations = []
    # Print the entire conversation
    for message in groupchat_writer.messages:
        if message['name'] == "writer":
            conversations.append(message['content'])
 
    return conversations[-1]
