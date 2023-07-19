"""Agent module"""
from typing import Any
import re
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
from chains import process_raw_chain, generate_tasks_chain
from base import gptTurbo
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import prompts

from uuid import uuid4

# import spacy

embeddings = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embeddings.embed_query)

# vectordb.add_documents(
#     documents=[
#         Document(
#             id="1",
#             vector=[1, 2, 3],
#             metadata={"name": "John Doe", "age": 30},
#         ),
#     ]
# )


def save_from_memory():
    # load the document and split it into chunks
    with open("./temporary.json", "r") as f:
        docs = json.load(f)

    # nlp = spacy.load("en_core_web_sm")  #  use this as a sentencizer
    # sentences_by_paragraph: list[list[str]] = [
    #     [sent.text for sent in nlp(docs["bulk"]).sents]
    # ]
    # bigrams_by_paragraph: list[list[str]] = [
    #     [f"{sentences[i]} {sentences[i+1]}" for i in range(len(sentences) - 1)]
    #     for sentences in sentences_by_paragraph
    # ]

    # # just flatten it out
    # sentences: list[str] = [
    #     sent for sentences in bigrams_by_paragraph for sent in sentences
    # ]

    # embeddings = [
    #     r["embedding"]
    #     for r in openai.Embedding.create(
    #         input=sentences, model="text-embedding-ada-002"
    #     )["data"]
    # ]

    # for embedding in embeddings:
    vectordb.add_documents(
        documents=[
            Document(
                id=str(uuid4()),
                page_content=docs["bulk"],
                metadata={"summary": docs["summary"], "bulk": docs["bulk"]},
            ),
        ]
    )
    
    return "Successfully saved!"


llm = ChatOpenAI(temperature=0.7)

search = SerpAPIWrapper()


def retrieve_docs(keyword: str):
    # retriever = vectordb.as_retriever()
    ## = retriever.get_relevant_documents(keyword)[0]

    return  ##


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, chain_type="~~", retriever=retriever, return_source_documents=True
# )

query = "???"
# llm_response = qa_chain(query)


def generate_specific_tasks(goal: str):
    template = PromptTemplate.from_template(prompts.GENERATE_SPECIFIC_TASKS_PROMPT)
    tasks = gptTurbo.call_as_llm(template.format(goal=goal))
    # Save tasks to tasks.txt file
    with open("tasks.txt", "w") as f:
        f.write(tasks)
    return "Successfully saved tasks"


def search_and_process(keyword: str):
    """Process raw text"""
    raw_info = search.run(keyword)
    template = PromptTemplate.from_template(prompts.PROCESS_RAW_TEXT_PROMPT)
    output = gptTurbo.call_as_llm(template.format(content=raw_info))
    parsed_output = {
        "summary": re.search("\[SUMMARY START\](.*?)\[SUMMARY END\]", output, re.DOTALL)
        .group(1)
        .strip(),
        "bulk": re.search("\[BULK START\](.*?)\[BULK END\]", output, re.DOTALL)
        .group(1)
        .strip(),
    }
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
        func=save_from_memory,
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
