from pydantic import BaseModel, Field


# Original model for raw responses.
class ModelResponse(BaseModel):
    text_output: str = Field(description="The raw text output from the model.")
