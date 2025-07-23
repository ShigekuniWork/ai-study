"""Information evaluator module for assessing interview completeness."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .persona import EvaluationResult, Interview
from typing import cast


class InformationEvaluator:
    """Evaluates whether collected information is sufficient for requirements."""

    def __init__(self, llm: ChatOllama):
        """Initialize the evaluator with an LLM."""
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
            """Evaluate if interviews provide sufficient information for requirements."""
            # Define prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert at evaluating the sufficiency of information for creating comprehensive requirements documents.",
                    ),
                    (
                        "human",
                        "Based on the following user request and interview results, please determine whether sufficient information has been gathered to create a comprehensive requirements document.\n\n"
                        "User Request: {user_request}\n\n"
                        "Interview Results:\n{interview_results}",
                    ),
                ]
            )
            # Create chain to evaluate information sufficiency
            chain = prompt | self.llm
            # Return evaluation result
            return cast(
                EvaluationResult,
                chain.invoke(
                    {
                        "user_request": user_request,
                        "interview_results": "\n".join(
                            f"Persona: {i.persona.name} - {i.persona.background}\n"
                            f"Question: {i.question}\nAnswer: {i.answer}\n"
                            for i in interviews
                        ),
                    }
                ),
            )
