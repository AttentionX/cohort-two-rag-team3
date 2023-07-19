"""Prompts"""

PROCESS_RAW_TEXT_PROMPT = """
You are required to generate a summary and bulk information from the provided CONTENT.
You MUST follow the provided output format.

CONTENT:
{content}

Output format:
[SUMMARY START]
...
[SUMMARY END]

[BULK START]
...
[BULK END]

Now generate the output.
"""

GENERATE_SPECIFIC_TASKS_PROMPT = """
You must generate 50 specific tasks from the provided goal.
The tasks will be used to search specific information from the internet.
Output format should be like this:
Output format (1~50):
1. ...
2. ...
3. ...
4. ...

Goal:
{goal}

Now generate the tasks:
"""
