"""
Consolidated tool schemas.
Replaces app/services/fin/tool_schemas.py
This will be imported by handlers when registering tools.
"""
TRANSACTION_SCHEMAS = {
    "GET_TRANSACTIONS": {
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Number of days to look back (default: 30)"
            },
            "category": {
                "type": "string",
                "description": "Filter by category"
            },
            "account_id": {
                "type": "string",
                "description": "Filter by account ID"
            },
            "account_name": {
                "type": "string",
                "description": "Filter by account name"
            },
            "type": {
                "type": "string",
                "enum": ["income", "expense", "all"],
                "description": "Transaction type filter"
            }
        }
    },
    "CREATE_TRANSACTION": {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "description": "Transaction amount (negative for expenses)"
            },
            "description": {
                "type": "string",
                "description": "Transaction description"
            },
            "category": {
                "type": "string",
                "description": "Transaction category"
            },
            "date": {
                "type": "string",
                "description": "Transaction date (YYYY-MM-DD)"
            },
            "account_id": {
                "type": "string",
                "description": "Account ID for the transaction"
            },
            "account_name": {
                "type": "string",
                "description": "Account name (alternative to ID)"
            },
            "recurring": {
                "type": "boolean",
                "description": "Whether this is a recurring transaction"
            }
        },
        "required": ["amount", "description"]
    },
    "UPDATE_TRANSACTION": {
        "type": "object",
        "properties": {
            "transaction_id": {
                "type": "string",
                "description": "ID of transaction to update"
            },
            "amount": {
                "type": "number",
                "description": "New amount"
            },
            "description": {
                "type": "string",
                "description": "New description"
            },
            "category": {
                "type": "string",
                "description": "New category"
            },
            "date": {
                "type": "string",
                "description": "New date (YYYY-MM-DD)"
            },
            "recurring": {
                "type": "boolean",
                "description": "Update recurring status"
            }
        },
        "required": ["transaction_id"]
    },
    "DELETE_TRANSACTION": {
        "type": "object",
        "properties": {
            "transaction_id": {
                "type": "string",
                "description": "ID of transaction to delete"
            }
        },
        "required": ["transaction_id"]
    },
    "CATEGORIZE_TRANSACTIONS": {
        "type": "object",
        "properties": {}
    }
}
# Account tool schemas
ACCOUNT_SCHEMAS = {
    "GET_USER_ACCOUNTS": {
        "type": "object",
        "properties": {}
    },
    "GET_ACCOUNT_DETAILS": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "ID of the account to get details for"
            }
        },
        "required": ["account_id"]
    },
    "INITIATE_PLAID_CONNECTION": {
        "type": "object",
        "properties": {}
    },
    "DISCONNECT_ACCOUNT": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "ID of the account to disconnect"
            }
        },
        "required": ["account_id"]
    }
}

# Invoice tool schemas
INVOICE_SCHEMAS = {
    "CONNECT_STRIPE": {
        "type": "object",
        "properties": {}
    },
    "CREATE_INVOICE": {
        "type": "object",
        "properties": {
            "client_name": {
                "type": "string",
                "description": "Name of the client"
            },
            "client_email": {
                "type": "string",
                "description": "Email address of the client"
            },
            "amount": {
                "type": "string",
                "description": "Invoice amount"
            },
            "description": {
                "type": "string",
                "description": "Description of services"
            },
            "due_date": {
                "type": "string",
                "description": "Due date for the invoice"
            }
        },
        "required": ["client_name", "client_email", "amount"]
    },
    "SEND_INVOICE": {
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "ID of the invoice to send"
            }
        },
        "required": ["invoice_id"]
    },
    "DELETE_INVOICE": {
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "ID of the invoice to delete"
            }
        },
        "required": ["invoice_id"]
    },
    "GET_INVOICES": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["all", "paid", "unpaid", "overdue"],
                "description": "Filter invoices by status"
            }
        }
    },
    "SEND_INVOICE_REMINDER": {
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "ID of the invoice"
            }
        },
        "required": ["invoice_id"]
    },
    "MARK_INVOICE_PAID": {
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "ID of the invoice"
            }
        },
        "required": ["invoice_id"]
    }
}

# Stripe Connect tool schemas
STRIPE_SCHEMAS = {
    "SETUP_STRIPE_CONNECT": {
        "type": "object",
        "properties": {}
    },
    "CHECK_STRIPE_CONNECT_STATUS": {
        "type": "object",
        "properties": {}
    },
    "CREATE_STRIPE_CONNECT_DASHBOARD_LINK": {
        "type": "object",
        "properties": {}
    },
    "GET_STRIPE_CONNECT_EARNINGS": {
        "type": "object",
        "properties": {
            "period": {
                "type": "string",
                "enum": ["today", "week", "month", "year", "all"],
                "description": "Time period for earnings"
            }
        }
    },
    "DISCONNECT_STRIPE_CONNECT": {
        "type": "object",
        "properties": {}
    },
    "RESTART_STRIPE_CONNECT_SETUP": {
        "type": "object",
        "properties": {}
    },
    "RESUME_STRIPE_CONNECT_ONBOARDING": {
        "type": "object",
        "properties": {}
    },
    "GET_STRIPE_CONNECT_REQUIREMENTS": {
        "type": "object",
        "properties": {}
    }
}

# Analytics tool schemas
ANALYTICS_SCHEMAS = {
    "FORECAST_CASH_FLOW": {
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Number of days to forecast (default: 30)"
            },
            "adjustments": {
                "type": "object",
                "description": "Scenario adjustments",
                "properties": {
                    "income_change": {
                        "type": "number",
                        "description": "Percentage change in income"
                    },
                    "expense_change": {
                        "type": "number",
                        "description": "Percentage change in expenses"
                    }
                }
            }
        }
    },
    "ANALYZE_TRENDS": {
        "type": "object",
        "properties": {
            "period": {
                "type": "string",
                "enum": ["1m", "3m", "6m", "1y"],
                "description": "Analysis period"
            }
        },
        "required": ["period"]
    },
    "DETECT_ANOMALIES": {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "description": "Anomaly score threshold (default: 0.8)"
            }
        }
    },
    "GENERATE_BUDGET": {
        "type": "object",
        "properties": {
            "monthly_income": {
                "type": "number",
                "description": "Monthly income amount (if different from calculated)"
            }
        }
    },
    "CALCULATE_CATEGORY_SPENDING": {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of categories to analyze"
            },
            "start_date": {
                "type": "string",
                "description": "Start date in ISO format (YYYY-MM-DD)"
            },
            "end_date": {
                "type": "string",
                "description": "End date in ISO format (YYYY-MM-DD)"
            }
        },
        "required": ["categories"]
    }
}