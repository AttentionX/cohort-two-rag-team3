from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from base import gptTurbo
import prompts

process_raw_chain = LLMChain(
    llm=gptTurbo,
    prompt=PromptTemplate.from_template(
        prompts.PROCESS_RAW_TEXT_PROMPT,
    ),
)

generate_tasks_chain = LLMChain(
    llm=gptTurbo,
    prompt=PromptTemplate.from_template(prompts.GENERATE_SPECIFIC_TASKS_PROMPT),
)
