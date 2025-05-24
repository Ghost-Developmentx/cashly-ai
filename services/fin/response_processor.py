import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """
    Responsible for:
    - Cleaning Claude's text responses
    - Executing tools using ToolRegistry
    - Sending follow-up requests if tools were used
    """

    def __init__(self, tool_executor):
        self.tool_executor = tool_executor

    def _clean_text(self, text: str) -> str:
        """Strip Claude tags and generic XML-like blocks."""
        patterns = [
            (re.compile(r"<thinking>.*?</thinking>", re.DOTALL), ""),
            (
                re.compile(
                    r"<search_quality_reflection>.*?</search_quality_reflection>",
                    re.DOTALL,
                ),
                "",
            ),
            (
                re.compile(
                    r"<search_quality_score>.*?</search_quality_score>", re.DOTALL
                ),
                "",
            ),
            (re.compile(r"<result>(.*?)</result>", re.DOTALL), lambda m: m.group(1)),
            (re.compile(r"<[^>]+>"), ""),  # remove any other tags
        ]
        for pattern, repl in patterns:
            text = pattern.sub(repl, text)
        return text.strip()

    def process(
        self,
        response: Any,
        *,
        user_id: str,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        original_messages: List[Dict[str, Any]],
        system_prompt: str,
        claude_client: Any,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"response_text": "", "actions": []}
        tool_results = []
        has_tool_use = False

        for block in getattr(response, "content", []):
            if block.type == "text":
                cleaned = self._clean_text(block.text or "")
                result["response_text"] += cleaned + "\n"

            elif block.type == "tool_use":
                has_tool_use = True
                logger.info(f"🔧 Tool use detected: {block.name}")

                tool_result = self.tool_executor(
                    block.name,
                    block.input,
                    user_id=user_id,
                    transactions=transactions,
                    user_context=user_context,
                )

                logger.info(
                    f"🔧 Tool result keys: {tool_result.keys() if isinstance(tool_result, dict) else 'Not a dict'}"
                )

                tool_results.append(
                    {
                        "tool": block.name,
                        "id": block.id,
                        "result": tool_result,
                    }
                )

                # Map tool names to action types properly
                action_type_mapping = {
                    "get_transactions": "show_transactions",
                    "get_user_accounts": "show_accounts",
                    "forecast_cash_flow": "show_forecast",
                    "analyze_trends": "show_trends",
                    "generate_budget": "show_budget",
                    "calculate_category_spending": "show_categories",
                    "detect_anomalies": "show_anomalies",
                    "create_transaction": "create_transaction",
                    "update_transaction": "update_transaction",
                    "delete_transaction": "delete_transaction",
                    "initiate_plaid_connection": "initiate_plaid_connection",
                    "connect_stripe": "connect_stripe",
                    "create_invoice": "create_invoice",
                    "get_invoices": "show_invoices",
                    "send_invoice_reminder": "send_invoice_reminder",
                    "mark_invoice_paid": "mark_invoice_paid",
                }

                # Add the action with the appropriate type
                if block.name in action_type_mapping:
                    action = {
                        "type": action_type_mapping[block.name],
                        "data": tool_result,
                    }
                    result["actions"].append(action)
                    logger.info(
                        f"🎯 Added action: type={action['type']}, has_data={bool(action.get('data'))}"
                    )
                else:
                    logger.warning(f"❌ Unmapped tool: {block.name}")
                    result["actions"].append(
                        {
                            "type": f"show_{block.name}",
                            "data": tool_result,
                        }
                    )

        if has_tool_use:
            # Follow-up messages to Claude
            follow_up_messages = original_messages[:]
            follow_up_messages.append(
                {"role": "assistant", "content": response.content}
            )
            follow_up_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r["id"],
                            "content": json.dumps(r["result"]),
                        }
                        for r in tool_results
                    ],
                }
            )

            try:
                follow_up = claude_client.messages.create(
                    model="claude-3-opus-20240229",
                    system=system_prompt,
                    messages=follow_up_messages,
                    max_tokens=2000,
                )
                if hasattr(follow_up, "content"):
                    for block in follow_up.content:
                        if block.type == "text":
                            result["response_text"] = self._clean_text(block.text or "")
            except Exception as e:
                logger.error(f"Follow-up Claude call failed: {e}")

        logger.info(
            f"📤 Final response: actions_count={len(result['actions'])}, has_text={bool(result['response_text'])}"
        )
        logger.info(f"📤 Action types: {[a['type'] for a in result['actions']]}")

        result["tool_results"] = tool_results
        result["response_text"] = result["response_text"].strip()
        return result
