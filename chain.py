from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama

# Model configuration
model = ChatOllama(model="llama3.1:latest", temperature=0.2)
parser = StrOutputParser()

# Recipe generation chain
recipe_prompt = ChatPromptTemplate.from_messages(
    [("system", "Think recipe from user input dish"), ("human", "{dish}")]
)
recipe_chain = recipe_prompt | model | parser

# Material extraction chain
material_prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me material for your recipe only"), ("human", "{dish}")]
)
material_chain = material_prompt | model | parser


def upper(text: str) -> str:
    """
    Convert uppercase.
    """
    return text.upper()


# Combined pipeline
full_recipe_pipeline = recipe_chain | material_chain | upper

# Execute with streaming
for chunk in full_recipe_pipeline.stream({"dish": "pasta"}):
    print(chunk, end="", flush=True)
