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
