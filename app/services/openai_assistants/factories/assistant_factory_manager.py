"""
Assistant Factory Manager for OpenAI Assistants.
Manages and orchestrates all specialized assistant factories.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from enum import Enum

from .base_assistant_factory import BaseAssistantFactory
from .transaction_assistant_factory import TransactionAssistantFactory
from .invoice_assistant_factory import InvoiceAssistantFactory
from .account_assistant_factory import AccountAssistantFactory
from .connection_assistant_factory import BankConnectionAssistantFactory
from .payment_processing_assistant_factory import PaymentProcessingAssistantFactory
from .forecasting_assistant_factory import ForecastingAssistantFactory
from .budget_assistant_factory import BudgetAssistantFactory
from .insights_assistant_factory import InsightsAssistantFactory

logger = logging.getLogger(__name__)


class AssistantType(Enum):
    """Enumeration of available assistant types."""

    TRANSACTION = "transaction"
    ACCOUNT = "account"
    BANK_CONNECTION = "bank_connection"
    PAYMENT_PROCESSING = "payment_processing"
    INVOICE = "invoice"
    FORECASTING = "forecasting"
    BUDGET = "budget"
    INSIGHTS = "insights"


class AssistantFactoryManager:
    """
    Manages all specialized assistant factories and provides a unified interface
    for creating, updating, and managing OpenAI Assistants.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

        # Initialize specialized factories
        self.factories: Dict[AssistantType, BaseAssistantFactory] = {
            AssistantType.TRANSACTION: TransactionAssistantFactory(api_key),
            AssistantType.ACCOUNT: AccountAssistantFactory(api_key),
            AssistantType.BANK_CONNECTION: BankConnectionAssistantFactory(api_key),
            AssistantType.PAYMENT_PROCESSING: PaymentProcessingAssistantFactory(
                api_key
            ),
            AssistantType.INVOICE: InvoiceAssistantFactory(api_key),
            AssistantType.FORECASTING: ForecastingAssistantFactory(api_key),
            AssistantType.BUDGET: BudgetAssistantFactory(api_key),
            AssistantType.INSIGHTS: InsightsAssistantFactory(api_key),
        }

        # Track created assistants
        self.assistant_registry: Dict[AssistantType, str] = {}

    def create_all_assistants(self) -> Dict[str, str]:
        """
        Create all available assistants using their specialized factories.

        Returns:
            Dictionary mapping assistant type names to their IDs
        """
        logger.info("Creating all OpenAI Assistants...")

        assistant_ids = {}
        creation_results = {}

        for assistant_type, factory in self.factories.items():
            try:
                logger.info(f"Creating {assistant_type.value} assistant...")

                # Validate configuration before creation
                validation = factory.validate_configuration()
                if not validation["valid"]:
                    logger.error(
                        f"Invalid configuration for {assistant_type.value}: {validation['issues']}"
                    )
                    creation_results[assistant_type.value] = {
                        "success": False,
                        "error": f"Configuration validation failed: {', '.join(validation['issues'])}",
                    }
                    continue

                # Create the assistant
                assistant_id = factory.create_assistant()
                assistant_ids[assistant_type.value] = assistant_id
                self.assistant_registry[assistant_type] = assistant_id

                creation_results[assistant_type.value] = {
                    "success": True,
                    "assistant_id": assistant_id,
                    "validation": validation,
                }

                logger.info(
                    f"✅ Created {assistant_type.value} assistant: {assistant_id}"
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to create {assistant_type.value} assistant: {e}"
                )
                creation_results[assistant_type.value] = {
                    "success": False,
                    "error": str(e),
                }

        # Log summary
        successful = sum(1 for result in creation_results.values() if result["success"])
        total = len(creation_results)
        logger.info(f"Assistant creation complete: {successful}/{total} successful")

        return assistant_ids

    def create_assistant(self, assistant_type: AssistantType) -> str:
        """
        Create a single assistant of the specified type.

        Args:
            assistant_type: Type of assistant to create

        Returns:
            Assistant ID of the created assistant

        Raises:
            ValueError: If assistant type is not supported
            Exception: If assistant creation fails
        """
        if assistant_type not in self.factories:
            raise ValueError(f"Unsupported assistant type: {assistant_type}")

        factory = self.factories[assistant_type]
        assistant_id = factory.create_assistant()
        self.assistant_registry[assistant_type] = assistant_id

        return assistant_id

    def update_assistant(
        self, assistant_type: AssistantType, assistant_id: str, **updates
    ) -> bool:
        """
        Update an existing assistant.

        Args:
            assistant_type: Type of assistant to update
            assistant_id: ID of the assistant to update
            **updates: Configuration updates to apply

        Returns:
            True if update was successful, False otherwise
        """
        if assistant_type not in self.factories:
            logger.error(f"Unsupported assistant type: {assistant_type}")
            return False

        factory = self.factories[assistant_type]
        success = factory.delete_assistant(assistant_id)

        if success and assistant_type in self.assistant_registry:
            del self.assistant_registry[assistant_type]

        return success

    def get_assistant_info(
        self, assistant_type: AssistantType, assistant_id: str
    ) -> Dict[str, Any]:
        """
        Get information about an assistant.

        Args:
            assistant_type: Type of assistant
            assistant_id: ID of the assistant

        Returns:
            Dictionary containing assistant information
        """
        if assistant_type not in self.factories:
            return {"error": f"Unsupported assistant type: {assistant_type}"}

        factory = self.factories[assistant_type]
        return factory.get_assistant_info(assistant_id)

    def get_factory(
        self, assistant_type: AssistantType
    ) -> Optional[BaseAssistantFactory]:
        """
        Get the factory for a specific assistant type.

        Args:
            assistant_type: Type of assistant

        Returns:
            Factory instance or None if not found
        """
        return self.factories.get(assistant_type)

    def validate_all_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate configurations for all assistant types.

        Returns:
            Dictionary with validation results for each assistant type
        """
        logger.info("Validating all assistant configurations...")

        validation_results = {}

        for assistant_type, factory in self.factories.items():
            try:
                validation = factory.validate_configuration()
                validation_results[assistant_type.value] = validation

                if validation["valid"]:
                    logger.info(f"✅ {assistant_type.value} configuration is valid")
                else:
                    logger.warning(
                        f"⚠️ {assistant_type.value} configuration has issues: {validation['issues']}"
                    )

            except Exception as e:
                logger.error(
                    f"❌ Error validating {assistant_type.value} configuration: {e}"
                )
                validation_results[assistant_type.value] = {
                    "valid": False,
                    "issues": [f"Validation error: {str(e)}"],
                    "warnings": [],
                }

        return validation_results

    def get_assistant_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities and features for all assistant types.

        Returns:
            Dictionary with capabilities for each assistant type
        """
        capabilities = {}

        for assistant_type, factory in self.factories.items():
            try:
                # Get specialized features if available
                if hasattr(factory, "get_specialized_features"):
                    features = factory.get_specialized_features()
                else:
                    features = {"primary_domain": assistant_type.value}

                # Add basic info
                capabilities[assistant_type.value] = {
                    "name": factory.get_assistant_name(),
                    "type": assistant_type.value,
                    "features": features,
                    "factory_class": factory.__class__.__name__,
                }

            except Exception as e:
                logger.error(
                    f"Error getting capabilities for {assistant_type.value}: {e}"
                )
                capabilities[assistant_type.value] = {
                    "name": f"Unknown {assistant_type.value}",
                    "error": str(e),
                }

        return capabilities

    def list_available_assistants(self) -> List[Dict[str, str]]:
        """
        List all available assistant types and their current status.

        Returns:
            List of assistant information dictionaries
        """
        assistants = []

        for assistant_type, factory in self.factories.items():
            assistant_info = {
                "type": assistant_type.value,
                "name": factory.get_assistant_name(),
                "factory": factory.__class__.__name__,
                "created": assistant_type in self.assistant_registry,
            }

            if assistant_type in self.assistant_registry:
                assistant_info["assistant_id"] = self.assistant_registry[assistant_type]

            assistants.append(assistant_info)

        return assistants

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all assistants and factories.

        Returns:
            Health check results
        """
        logger.info("Performing assistant factory health check...")

        health_status = {
            "status": "healthy",
            "factories": {},
            "assistants": {},
            "summary": {
                "total_factories": len(self.factories),
                "healthy_factories": 0,
                "created_assistants": len(self.assistant_registry),
                "missing_assistants": [],
            },
        }

        # Check each factory
        for assistant_type, factory in self.factories.items():
            factory_health = {
                "status": "healthy",
                "name": factory.get_assistant_name(),
                "validation": None,
                "assistant_created": assistant_type in self.assistant_registry,
            }

            try:
                # Validate configuration
                validation = factory.validate_configuration()
                factory_health["validation"] = validation

                if not validation["valid"]:
                    factory_health["status"] = "degraded"
                    health_status["status"] = "degraded"
                else:
                    health_status["summary"]["healthy_factories"] += 1

                # Check if assistant was created
                if assistant_type in self.assistant_registry:
                    assistant_id = self.assistant_registry[assistant_type]

                    # Try to get assistant info to verify it exists
                    try:
                        assistant_info = factory.get_assistant_info(assistant_id)
                        if "error" in assistant_info:
                            factory_health["status"] = "error"
                            factory_health["error"] = assistant_info["error"]
                            health_status["status"] = "degraded"
                        else:
                            health_status["assistants"][assistant_type.value] = {
                                "id": assistant_id,
                                "name": assistant_info.get("name"),
                                "status": "active",
                            }
                    except Exception as e:
                        factory_health["status"] = "error"
                        factory_health["error"] = f"Cannot verify assistant: {str(e)}"
                        health_status["status"] = "degraded"
                else:
                    health_status["summary"]["missing_assistants"].append(
                        assistant_type.value
                    )

            except Exception as e:
                factory_health["status"] = "error"
                factory_health["error"] = str(e)
                health_status["status"] = "unhealthy"
                logger.error(f"Health check failed for {assistant_type.value}: {e}")

            health_status["factories"][assistant_type.value] = factory_health

        return health_status

    def add_factory(self, assistant_type: AssistantType, factory: BaseAssistantFactory):
        """
        Add a new factory for an assistant type.

        Args:
            assistant_type: Type of assistant
            factory: Factory instance
        """
        self.factories[assistant_type] = factory
        logger.info(f"Added factory for {assistant_type.value} assistant")

    def remove_factory(self, assistant_type: AssistantType):
        """
        Remove a factory for an assistant type.

        Args:
            assistant_type: Type of assistant to remove
        """
        if assistant_type in self.factories:
            del self.factories[assistant_type]

            # Also remove from registry if present
            if assistant_type in self.assistant_registry:
                del self.assistant_registry[assistant_type]

            logger.info(f"Removed factory for {assistant_type.value} assistant")

    def get_creation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of assistant creation status.

        Returns:
            Summary of which assistants have been created
        """
        total_types = len(self.factories)
        created_count = len(self.assistant_registry)

        created_assistants = []
        missing_assistants = []

        for assistant_type in self.factories.keys():
            if assistant_type in self.assistant_registry:
                created_assistants.append(
                    {
                        "type": assistant_type.value,
                        "id": self.assistant_registry[assistant_type],
                    }
                )
            else:
                missing_assistants.append(assistant_type.value)

        return {
            "total_assistant_types": total_types,
            "created_count": created_count,
            "missing_count": len(missing_assistants),
            "completion_percentage": (
                (created_count / total_types * 100) if total_types > 0 else 0
            ),
            "created_assistants": created_assistants,
            "missing_assistants": missing_assistants,
        }
