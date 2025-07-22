import operator
from typing import Annotated

from langchain_core.pydantic_v1 import BaseModel, Field


class State(BaseModel):
    query: str = Field(..., description="The user's query")
    current_role: str = Field(..., description="The current role of the agent")
    messages: Annotated[list, operator.add] = Field(
        ..., description="The messages in the conversation"
    )
    current_judge: bool = Field(..., description="The current judge of the agent")
    judgement_reason: str = Field(..., description="The reason for the current judge")