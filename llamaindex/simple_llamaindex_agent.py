#A very simple implementation of AI Agent capable
#of using custom tools in LlamaIndex.

#Example run:
# >python3 simple_llamaindex_agent.py
# Running model granite4:350m locally...
# Type EXIT to quit the program.
# >>Come up with a fortune cookie message.
# Here is your fortune cookie message: You will step on the soil of many countries.
# >>

from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
import asyncio, random

MODEL_NAME = "granite4:350m"
def get_fortune_cookie_message() -> str:
    """This tool does not receive any arguments and
       returns a string containing a message from a fortune cookie."""

    messages = ["A pleasant surprise is in store for you.",
                "You will step on the soil of many countries.",
                "Happy news is on its way to you.",
                "You will live a long, happy life.",
                "The time is right to make new friends.",
                "An old acquaintance will re-enter your life.",
                "A new friendship will bring unexpected joy",]

    return random.choice(messages)

async def run_agent():
    while (query := input(">>")) != "EXIT":
        print(await agent.run(query, ctx=ctx))

if __name__ == "__main__":
    llm = Ollama(model=MODEL_NAME,
        max_output_tokens=100,
        temperature=0.1,
        request_timeout=3600.0,
        num_ctx=12000
    )

    agent = AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(fn=get_fortune_cookie_message)],
        llm=llm
    )

    ctx = Context(agent)
    print(f"Running model {MODEL_NAME} locally...")
    print("Type EXIT to quit the program.")
    asyncio.run(run_agent())




