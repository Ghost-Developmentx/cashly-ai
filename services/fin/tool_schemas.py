from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "setup_stripe_connect",
        "description": "Set up Stripe Connect to accept payments and manage invoices with platform fees",
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Country code for the business (default: US)",
                    "default": "US",
                },
                "business_type": {
                    "type": "string",
                    "enum": [
                        "individual",
                        "company",
                        "non_profit",
                        "government_entity",
                    ],
                    "description": "Type of business entity",
                    "default": "individual",
                },
            },
        },
    },
    {
        "name": "check_stripe_connect_status",
        "description": "Check the current status of the user's Stripe Connect account",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_stripe_connect_dashboard_link",
        "description": "Create a link to the Stripe Express dashboard for payment management",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_stripe_connect_earnings",
        "description": "Get earnings and platform fee information from Stripe Connect",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["week", "month", "quarter", "year"],
                    "description": "Time period for earnings report",
                    "default": "month",
                }
            },
        },
    },
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
        "name": "create_invoice",
        "description": "Create a new invoice for a client",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_name": {"type": "string", "description": "Client's name"},
                "client_email": {
                    "type": "string",
                    "description": "Client's email address",
                },
                "amount": {"type": "number", "description": "Invoice amount"},
                "description": {
                    "type": "string",
                    "description": "Description of services/products",
                },
                "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"},
            },
            "required": ["client_name", "client_email", "amount"],
        },
    },
    {
        "name": "get_invoices",
        "description": "Retrieve invoices with optional filters",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["draft", "pending", "paid", "overdue", "cancelled"],
                    "description": "Filter by invoice status",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back",
                },
                "client_name": {
                    "type": "string",
                    "description": "Filter by client name",
                },
            },
        },
    },
    {
        "name": "send_invoice_reminder",
        "description": "Send a payment reminder for an invoice",
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "ID of the invoice"}
            },
            "required": ["invoice_id"],
        },
    },
    {
        "name": "mark_invoice_paid",
        "description": "Mark an invoice as paid",
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "ID of the invoice"}
            },
            "required": ["invoice_id"],
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
    {
        "name": "restart_stripe_connect_setup",
        "description": "Restart Stripe Connect setup from scratch, disconnecting any existing incomplete account",
        "input_schema": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force restart even if account exists",
                    "default": False,
                }
            },
        },
    },
    {
        "name": "resume_stripe_connect_onboarding",
        "description": "Resume an incomplete Stripe Connect onboarding process",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["continue", "restart"],
                    "description": "Whether to continue existing setup or restart",
                    "default": "continue",
                }
            },
        },
    },
    {
        "name": "get_stripe_connect_requirements",
        "description": "Get detailed information about what's needed to complete Stripe Connect setup",
        "input_schema": {"type": "object", "properties": {}},
    },
]
