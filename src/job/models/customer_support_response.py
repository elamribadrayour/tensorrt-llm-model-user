from typing import Any
from pydantic import BaseModel, Field


# New Pydantic model describing the creative structured response for a customer support agent.
class CustomerSupportResponse(BaseModel):
    answer: str = Field(description="The actual text answer from the agent.")
    additional_details: dict[str, Any] | None = Field(
        description="Any extra information"
    )
    classification: str = Field(
        description="For example 'Coverage Inquiry', 'Billing Issue', etc."
    )
    next_action: str | None = Field(
        description="Suggested next step from the agent (e.g., 'review_policy', 'contact_agent')"
    )
