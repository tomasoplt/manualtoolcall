from langchain_core.tools import tool
from langchain_ollama import ChatOllama

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


llm = ChatOllama(
    model="qwen3-vl:8b-instruct",
    temperature=0
)

llm_with_tools = llm.bind_tools([multiply])

# 1. Ask the model
response = llm_with_tools.invoke(
    "What is 12 multiplied by 5?"
)

# 2. Inspect the requested tool call
print("Tool calls:", response.tool_calls)

# 3. Execute the tool ourselves
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = multiply.invoke(tool_call["args"])
    print("Result:", result)
else:
    print("The model did not request a tool call.")
    print("Response:", response.content)