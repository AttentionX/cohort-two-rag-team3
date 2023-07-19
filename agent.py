from langchain.agents import (
    Agent,
    initialize_agent,
    Tool,
    AgentType,
    ConversationalChatAgent,
    AgentExecutor,
)
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import SerpAPIWrapper


llm = ChatOpenAI(temperature=0.7)


search = SerpAPIWrapper()


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events.\
        You should ask targeted questions",
    ),
]

search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)
