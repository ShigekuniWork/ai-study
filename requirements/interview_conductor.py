"""Interview conductor module for generating questions and answers from personas."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .persona import Interview, InterviewResult, Persona


class InterviewConductor:
    """Conducts interviews with personas to gather requirements."""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        """Run the interview process with given personas and user request."""
        # Generate questions
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        # Generate answers
        answers = self._generate_answers(personas=personas, questions=questions)
        # Create interview list from questions and answers combinations
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )
        # Return interview results
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: list[Persona]
    ) -> list[str]:
        # Define prompt for question generation
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at generating appropriate questions "
                    "based on user requirements.",
                ),
                (
                    "human",
                    "Generate one question about the user request related to "
                    "the following persona.\n\n"
                    "User Request: {user_request}\n"
                    "Persona: {persona_name} - {persona_background}\n\n"
                    "The question should be specific and designed to extract "
                    "important information from this persona's perspective.",
                ),
            ]
        )
        # Create chain for question generation
        question_chain = question_prompt | self.llm | StrOutputParser()

        # Create question queries for each persona
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        # Generate questions in batch processing
        return question_chain.batch(question_queries)

    def _generate_answers(
        self, personas: list[Persona], questions: list[str]
    ) -> list[str]:
        # Define prompt for answer generation
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are responding as the following persona: "
                    "{persona_name} - {persona_background}",
                ),
                ("human", "Question: {question}"),
            ]
        )
        # Create chain for answer generation
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # Create answer queries for each persona
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        # Generate answers in batch processing
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, personas: list[Persona], questions: list[str], answers: list[str]
    ) -> list[Interview]:
        # Create interview objects from combinations of questions and answers for each persona
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]
