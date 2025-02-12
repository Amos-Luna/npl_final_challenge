from pydantic import BaseModel, Field
from typing import List


class OutputQuerySchema(BaseModel):
    """Output format of new querys"""
    new_querys: List[str] = Field(description="List of new querys related to original query")
