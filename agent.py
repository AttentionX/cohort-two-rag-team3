"""Agent module"""
from typing import Any
import json
from langchain.agents import (
    Agent,
    initialize_agent,
    Tool,
    AgentType,
    ConversationalChatAgent,
    AgentExecutor,
)
from langchain.output_parsers import RegexParser
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import SerpAPIWrapper
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from chains import process_raw_chain
from langchain.vectorstores import Chroma

vectordb = Chroma()


llm = ChatOpenAI(temperature=0.7)

search = SerpAPIWrapper()


def search_and_process(keyword: str):
    """Process raw text"""
    raw_info = search.run(keyword)
    output = process_raw_chain.run(raw_info)
    parsed_output = summary_parser.parse(output)
    # Save parsed output to temporary.json file
    with open("temporary.json", "w") as f:
        f.write(json.dumps(parsed_output))

    return "Successfully saved to memory"


def save(info: str):
    print("save", info)


tools = [
    Tool(
        name="Search",
        func=search_and_process,
        description="useful for when you need to search information and save it temporarily to memory. You need to input a keyword to search for.",
    ),
    Tool(
        name="Save",
        func=save,
        description="useful for when you need to embed and save information into a vector database",
    ),
]

summary_parser = RegexParser(
    regex=r"\[SUMMARY START\](.*?)\[SUMMARY END\]|\[BULK START\](.*?)\[BULK END\]",
    output_keys=["summary", "bulk"],
)

search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

chat_agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


def search_and_insert(query, insert):
    results = search.run(query)
    insert(results)
