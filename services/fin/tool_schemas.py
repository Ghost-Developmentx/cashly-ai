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
        "name": "get_transactions",
        "description": "Retrieve and filter user transactions based on various criteria like account, date range, category, or amount",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Specific account ID to filter by",
                },
                "account_name": {
                    "type": "string",
                    "description": "Account name to search for (partial match)",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days back to search (default: 30)",
                    "default": 30,
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date for filtering (YYYY-MM-DD format)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date for filtering (YYYY-MM-DD format)",
                },
                "category": {
                    "type": "string",
                    "description": "Category to filter by (partial match)",
                },
                "min_amount": {
                    "type": "number",
                    "description": "Minimum transaction amount (absolute value)",
                },
                "max_amount": {
                    "type": "number",
                    "description": "Maximum transaction amount (absolute value)",
                },
                "type": {
                    "type": "string",
                    "enum": ["income", "expense", "all"],
                    "description": "Type of transactions to show",
                    "default": "all",
                },
            },
        },
    },
    {
        "name": "create_transaction",
        "description": "Create a new transaction entry for the user",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Transaction amount (positive for income, negative for expense)",
                },
                "description": {
                    "type": "string",
                    "description": "Description of the transaction",
                },
                "account_id": {
                    "type": "string",
                    "description": "ID of the account for this transaction",
                },
                "account_name": {
                    "type": "string",
                    "description": "Name of the account (if account_id not provided)",
                },
                "category": {
                    "type": "string",
                    "description": "Category for the transaction",
                    "default": "Uncategorized",
                },
                "date": {
                    "type": "string",
                    "description": "Transaction date (YYYY-MM-DD), defaults to today",
                },
                "recurring": {
                    "type": "boolean",
                    "description": "Whether this is a recurring transaction",
                    "default": "false",
                },
            },
            "required": ["amount", "description"],
        },
    },
    {
        "name": "update_transaction",
        "description": "Update an existing transaction with new information",
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "ID of the transaction to update",
                },
                "amount": {"type": "number", "description": "New transaction amount"},
                "description": {
                    "type": "string",
                    "description": "New transaction description",
                },
                "category": {
                    "type": "string",
                    "description": "New category for the transaction",
                },
                "date": {
                    "type": "string",
                    "description": "New transaction date (YYYY-MM-DD)",
                },
                "recurring": {
                    "type": "boolean",
                    "description": "Update recurring status",
                },
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "delete_transaction",
        "description": "Delete a transaction permanently",
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "ID of the transaction to delete",
                }
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "categorize_transactions",
        "description": "Automatically categorize uncategorized transactions using AI",
        "input_schema": {
            "type": "object",
            "properties": {
                "category_mappings": {
                    "type": "object",
                    "description": "Optional manual category mappings",
                    "additionalProperties": {"type": "string"},
                }
            },
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
