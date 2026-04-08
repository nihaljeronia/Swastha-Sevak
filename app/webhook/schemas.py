from pydantic import BaseModel, Field
from typing import Literal


class WhatsAppWebhookEvent(BaseModel):
    message_id: str = Field(..., alias="message_id")
    from_number: str = Field(..., alias="from_number")
    message_text: str = Field(..., alias="message_text")
    language: str = Field("unknown", alias="language")
    message_type: Literal["text", "audio"] = Field("text", alias="message_type")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
