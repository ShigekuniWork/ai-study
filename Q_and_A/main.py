from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1:latest", temperature=0.2).configurable_fields(
    max_tokens=ConfigurableField(id="max_tokens")
)
