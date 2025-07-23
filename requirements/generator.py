"""Persona generator module for creating diverse personas."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .persona import Personas


class PersonaGenerator:
    """Generates diverse personas for requirements gathering."""

    def __init__(self, llm: ChatOllama, k: int = 5):
        """Initialize the persona generator with an LLM and number of personas."""
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        """Generate personas based on user request."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that generates diverse personas "
                    "based on user requests. Each persona should have a unique "
                    "perspective and background relevant to the request.",
                ),
                (
                    "human",
                    f"Please generate {self.k} diverse personas for interviews "
                    "related to the following user request.\n\n"
                    "Each persona should include a name and brief background. "
                    "Please ensure diversity in names, gender, occupation, and technical expertise.\n\n"
                    "User Request: {user_request}",
                ),
            ]
        )

        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})
