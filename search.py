from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
print(len(documents))

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """
    You should answer the question based on the below context.

    Context: {context}
    Question: {question}
    """
)

retriever = db.as_retriever()
model = ChatOllama(model="llama3.1:latest", temperature=0.2)
chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke("What is LangChain?")
print(output)
