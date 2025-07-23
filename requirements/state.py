"""State management for interview process."""

from typing import Annotated

from pydantic import BaseModel, Field

from .persona import Interview, Persona


class InterviewState(BaseModel):
    """State object for tracking interview process."""

    user_request: str = Field(..., description="Request from the user")
    personas: Annotated[
        list[Persona], Field(default_factory=list, description="List of personas")
    ]
    interviews: Annotated[
        list[Interview], Field(default_factory=list, description="List of interviews")
    ]
    requirements_doc: str = Field(default="", description="Generated document")
    iteration: int = Field(default=0, description="Iteration number")
    is_information_sufficient: bool = Field(
        default=False, description="Whether the information is sufficient"
    )
    evaluation_reason: str = Field(default="", description="Reason for evaluation")
