from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.services.openai_assistants.assistant_manager.types import AssistantType


class AssistantConfig(BaseSettings):
    """
    Handles configuration for an AI assistant system using Pydantic BaseSettings.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore"
    )

    # API Configuration
    api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4-turbo-preview", validation_alias="OPENAI_MODEL")

    # Timeout Configuration
    timeout: int = Field(default=30, validation_alias="ASSISTANT_TIMEOUT")
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, validation_alias="RETRY_DELAY")

    # Assistant IDs
    transaction_assistant_id: Optional[str] = Field(default=None, validation_alias="TRANSACTION_ASSISTANT_ID")
    account_assistant_id: Optional[str] = Field(default=None, validation_alias="ACCOUNT_ASSISTANT_ID")
    bank_connection_assistant_id: Optional[str] = Field(default=None, validation_alias="BANK_CONNECTION_ASSISTANT_ID")
    payment_processing_assistant_id: Optional[str] = Field(default=None, validation_alias="PAYMENT_PROCESSING_ASSISTANT_ID")
    invoice_assistant_id: Optional[str] = Field(default=None, validation_alias="INVOICE_ASSISTANT_ID")
    forecasting_assistant_id: Optional[str] = Field(default=None, validation_alias="FORECASTING_ASSISTANT_ID")
    budget_assistant_id: Optional[str] = Field(default=None, validation_alias="BUDGET_ASSISTANT_ID")
    insights_assistant_id: Optional[str] = Field(default=None, validation_alias="INSIGHTS_ASSISTANT_ID")

    def _load_assistant_ids(self) -> Dict[str, Optional[str]]:
        """Load assistant IDs from the model fields."""
        return {
            "TRANSACTION": self.transaction_assistant_id,
            "ACCOUNT": self.account_assistant_id,
            "BANK_CONNECTION": self.bank_connection_assistant_id,
            "PAYMENT_PROCESSING": self.payment_processing_assistant_id,
            "INVOICE": self.invoice_assistant_id,
            "FORECASTING": self.forecasting_assistant_id,
            "BUDGET": self.budget_assistant_id,
            "INSIGHTS": self.insights_assistant_id,
        }

    def get_assistant_id(self, assistant_type: Union[str, AssistantType]) -> Optional[str]:
        assistant_ids = self._load_assistant_ids()
        if isinstance(assistant_type, AssistantType):
            key = assistant_type.name
        else:
            key = assistant_type
        return assistant_ids.get(key.upper())

    def is_assistant_configured(self, assistant_type: Union[str, AssistantType]) -> bool:
        """Check if an assistant is configured."""
        return self.get_assistant_id(assistant_type) is not None

    def validate(self) -> Dict[str, Any]:
        """Validate configuration."""
        issues = []

        if not self.api_key:
            issues.append("OPENAI_API_KEY not set")

        # Check which assistants are configured
        assistant_ids = self._load_assistant_ids()
        configured = [key for key, value in assistant_ids.items() if value is not None]
        missing = [key for key, value in assistant_ids.items() if value is None]

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "configured_assistants": configured,
            "missing_assistants": missing
        }
