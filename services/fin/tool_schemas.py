from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "forecast_cash_flow",
        "description": "Create a cash flow forecast based on historical transactions and optional adjustments",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to forecast",
                },
                "adjustments": {
                    "type": "object",
                    "description": "Optional adjustments to the forecast",
                    "properties": {
                        "income_adjustment": {
                            "type": "number",
                            "description": "Amount to adjust monthly income by (positive or negative)",
                        },
                        "expense_adjustment": {
                            "type": "number",
                            "description": "Amount to adjust monthly expenses by (positive or negative)",
                        },
                        "category_adjustments": {
                            "type": "object",
                            "description": "Category-specific adjustments",
                            "additionalProperties": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["days"],
        },
    },
    {
        "name": "analyze_trends",
        "description": "Analyze financial trends and patterns in transactions",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period for analysis",
                    "enum": ["1m", "3m", "6m", "1y"],
                }
            },
            "required": ["period"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": "Detect unusual or anomalous transactions",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Anomaly score threshold",
                }
            },
        },
    },
    {
        "name": "generate_budget",
        "description": "Generate budget recommendations based on spending patterns",
        "input_schema": {
            "type": "object",
            "properties": {
                "monthly_income": {
                    "type": "number",
                    "description": "Monthly income amount (if different from calculated)",
                }
            },
        },
    },
    {
        "name": "calculate_category_spending",
        "description": "Calculate spending in specific categories over a time period",
        "input_schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of categories to analyze",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in ISO format (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in ISO format (YYYY-MM-DD)",
                },
            },
            "required": ["categories"],
        },
    },
    {
        "name": "get_user_accounts",
        "description": "Get information about the user's connected bank accounts",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_account_details",
        "description": "Get detailed information about a specific account",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The account ID to get details for",
                }
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "initiate_plaid_connection",
        "description": "Start the process to connect a new bank account via Plaid",
        "input_schema": {
            "type": "object",
            "properties": {
                "institution_preference": {
                    "type": "string",
                    "description": "Optional preference for bank type (e.g., 'major_bank', 'credit_union', 'any')",
                }
            },
        },
    },
    {
        "name": "disconnect_account",
        "description": "Disconnect a bank account from the user's profile",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The account ID to disconnect",
                }
            },
            "required": ["account_id"],
        },
    },
]
