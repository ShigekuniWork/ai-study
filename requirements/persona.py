"""Persona and interview data models."""

from pydantic import BaseModel, Field


class Persona(BaseModel):
    """Represents a persona for requirements gathering."""

    name: str = Field(..., description="Name of the person")
    background: str = Field(..., description="Background of the person")


class Personas(BaseModel):
    """Collection of personas."""

    personas: list[Persona] = Field(..., description="List of personas")


class Interview(BaseModel):
    """Represents an interview with a persona."""

    persona: Persona = Field(..., description="Persona to interview")
    question: str = Field(..., description="Question to ask the person")
    answer: str = Field(..., description="Answer from the person")


class InterviewResult(BaseModel):
    """Collection of interview results."""

    interviews: list[Interview] = Field(..., description="List of interviews")


class EvaluationResult(BaseModel):
    """Result of evaluating information sufficiency."""

    reason: str = Field(..., description="Reason for the evaluation")
    is_sufficient: bool = Field(..., description="Whether the evaluation is sufficient")
