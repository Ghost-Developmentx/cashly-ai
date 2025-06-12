import logging
from typing import Dict, Any, Optional

from .manager import UnifiedAssistantManager, AssistantType

logger = logging.getLogger(__name__)

class AssistantFactory:
    """
    Simple factory for creating/updating assistants.
    Uses configuration instead of multiple specialized classes.
    """

    def __init__(self, manager: Optional[UnifiedAssistantManager] = None):
        """
        Initialize factory with assistant manager.

        Args:
            manager: UnifiedAssistantManager instance (creates one if not provided)
        """
        self.manager = manager or UnifiedAssistantManager()

    async def create_all_assistants(self) -> Dict[str, str]:
        """
        Create or update all configured assistants.

        Returns:
            Dictionary mapping assistant types to their IDs
        """
        logger.info("Creating/updating all assistants...")

        assistant_ids = {}
        results = {
            "created": [],
            "updated": [],
            "failed": []
        }

        for assistant_type in AssistantType:
            try:
                # Check if assistant is configured
                if assistant_type not in self.manager.assistant_configs:
                    logger.warning(f"No configuration for {assistant_type.value}")
                    continue

                config = self.manager.assistant_configs[assistant_type]

                # Determine if we're creating or updating
                is_update = bool(config.assistant_id)

                # Create or update assistant
                assistant_id = await self.manager.create_or_update_assistant(
                    assistant_type,
                    force_update=False
                )

                assistant_ids[assistant_type.value] = assistant_id

                if is_update:
                    results["updated"].append(assistant_type.value)
                else:
                    results["created"].append(assistant_type.value)

                logger.info(f"✅ {'Updated' if is_update else 'Created'} {assistant_type.value}: {assistant_id}")

            except Exception as e:
                logger.error(f"❌ Failed to process {assistant_type.value}: {e}")
                results["failed"].append({
                    "type": assistant_type.value,
                    "error": str(e)
                })

        # Summary
        total = len(AssistantType)
        successful = len(results["created"]) + len(results["updated"])

        logger.info(f"""
Assistant creation/update complete:
- Total: {total}
- Created: {len(results["created"])}
- Updated: {len(results["updated"])}
- Failed: {len(results["failed"])}
- Success rate: {(successful / total) * 100:.1f}%
        """)

        return assistant_ids

    async def create_assistant(
            self,
            assistant_type: AssistantType,
            force_create: bool = False
    ) -> str:
        """
        Create or update a single assistant.

        Args:
            assistant_type: Type of assistant to create
            force_create: Force creation of new assistant even if one exists

        Returns:
            Assistant ID
        """
        return await self.manager.create_or_update_assistant(
            assistant_type,
            force_update=force_create
        )

    async def validate_assistants(self) -> Dict[str, Any]:
        """
        Validate all assistant configurations and their tools.

        Returns:
            Validation results
        """
        return await self.manager.validate_all_assistants()

    async def update_all_assistants(self) -> Dict[str, Any]:
        """
        Update all existing assistants with latest configuration.

        Returns:
            Update results
        """
        logger.info("Updating all existing assistants...")

        results = {
            "updated": [],
            "skipped": [],
            "failed": []
        }

        for assistant_type, config in self.manager.assistant_configs.items():
            if not config.assistant_id:
                results["skipped"].append({
                    "type": assistant_type.value,
                    "reason": "No assistant ID configured"
                })
                continue

            try:
                await self.manager.create_or_update_assistant(
                    assistant_type,
                    force_update=True
                )
                results["updated"].append(assistant_type.value)
                logger.info(f"✅ Updated {assistant_type.value}")

            except Exception as e:
                logger.error(f"❌ Failed to update {assistant_type.value}: {e}")
                results["failed"].append({
                    "type": assistant_type.value,
                    "error": str(e)
                })

        return results

    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current assistant configurations.

        Returns:
            Configuration summary
        """
        summary = {
            "total_types": len(AssistantType),
            "configured": len(self.manager.assistant_configs),
            "assistants": {}
        }

        for assistant_type, config in self.manager.assistant_configs.items():
            summary["assistants"][assistant_type.value] = {
                "name": config.name,
                "model": config.model,
                "tools_count": len(config.tools),
                "tools": config.tools,
                "has_assistant_id": bool(config.assistant_id),
                "assistant_id": config.assistant_id
            }

        return summary

    async def cleanup_orphaned_assistants(self) -> Dict[str, Any]:
        """
        Find and optionally delete assistants not in current configuration.

        Returns:
            Cleanup results
        """
        logger.info("Checking for orphaned assistants...")

        results = {
            "found": [],
            "deleted": [],
            "errors": []
        }

        try:
            # Get all assistants from OpenAI
            all_assistants = await self.manager.client.beta.assistants.list()

            # Get configured assistant IDs
            configured_ids = {
                config.assistant_id
                for config in self.manager.assistant_configs.values()
                if config.assistant_id
            }

            # Find orphaned assistants
            for assistant in all_assistants.data:
                if assistant.id not in configured_ids:
                    # Check if it's a Cashly assistant by name
                    if "Cashly" in assistant.name:
                        results["found"].append({
                            "id": assistant.id,
                            "name": assistant.name,
                            "created_at": assistant.created_at
                        })

            logger.info(f"Found {len(results['found'])} potentially orphaned assistants")

        except Exception as e:
            logger.error(f"Error checking for orphaned assistants: {e}")
            results["errors"].append(str(e))

        return results
