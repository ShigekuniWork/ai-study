"""Requirements document generator module."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .persona import Interview


class RequirementsDocumentGenerator:
    """Generates requirements documents based on interview results."""

    def __init__(self, llm: ChatOllama):
        """Initialize the document generator with an LLM."""
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        """Generate a requirements document from user request and interviews."""
        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at creating requirements documents based on collected information.",
                ),
                (
                    "human",
                    "Please create a requirements document based on the following user request and interview results from multiple personas.\n\n"
                    "User Request: {user_request}\n\n"
                    "Interview Results:\n{interview_results}\n"
                    "Please include the following sections in the requirements document:\n"
                    "1. Project Overview\n"
                    "2. Key Features\n"
                    "3. Non-functional Requirements\n"
                    "4. Constraints\n"
                    "5. Target Users\n"
                    "6. Priorities\n"
                    "7. Risks and Mitigation Strategies\n\n"
                    "Please output in English.\n\nRequirements Document:",
                ),
            ]
        )
        # Create chain to generate requirements document
        chain = prompt | self.llm | StrOutputParser()
        # Generate requirements document
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"Persona: {i.persona.name} - {i.persona.background}\n"
                    f"Question: {i.question}\nAnswer: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

    def stream(self, user_request: str, interviews: list[Interview]):
        """Stream the requirements document generation."""
        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at creating requirements documents based on collected information.",
                ),
                (
                    "human",
                    "Please create a requirements document based on the following user request and interview results from multiple personas.\n\n"
                    "User Request: {user_request}\n\n"
                    "Interview Results:\n{interview_results}\n"
                    "Please include the following sections in the requirements document:\n"
                    "1. Project Overview\n"
                    "2. Key Features\n"
                    "3. Non-functional Requirements\n"
                    "4. Constraints\n"
                    "5. Target Users\n"
                    "6. Priorities\n"
                    "7. Risks and Mitigation Strategies\n\n"
                    "Please output in English.\n\nRequirements Document:",
                ),
            ]
        )
        # Create chain for streaming
        chain = prompt | self.llm

        # Stream the generation
        for chunk in chain.stream(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"Persona: {i.persona.name} - {i.persona.background}\n"
                    f"Question: {i.question}\nAnswer: {i.answer}\n"
                    for i in interviews
                ),
            }
        ):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)
