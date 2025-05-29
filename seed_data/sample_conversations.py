"""
Comprehensive sample conversation data for intent classification training.
Organized by intent with realistic user queries and assistant responses.
"""

from typing import Dict, List, Any


class SampleConversations:
    """Comprehensive sample conversations for seeding the vector database."""

    @staticmethod
    def get_invoice_conversations() -> List[Dict[str, Any]]:
        """Invoice-related conversations with high variety."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Show me all my invoices"},
                    {
                        "role": "assistant",
                        "content": "Here are all your invoices. You have 5 invoices total: 2 paid, 2 pending, and 1 draft.",
                    },
                ],
                "topics": ["invoices", "list", "all", "status"],
                "success": True,
                "user_context": "User with Stripe Connect enabled",
            },
            {
                "messages": [
                    {"role": "user", "content": "Can you show me my invoice list?"},
                    {
                        "role": "assistant",
                        "content": "I'll show you your complete invoice list with current status for each one.",
                    },
                ],
                "topics": ["invoices", "list", "show"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "List all invoices"},
                    {
                        "role": "assistant",
                        "content": "Here's your complete invoice list organized by status.",
                    },
                ],
                "topics": ["invoices", "list", "all"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "What invoices do I have?"},
                    {
                        "role": "assistant",
                        "content": "You currently have 8 invoices across different statuses.",
                    },
                ],
                "topics": ["invoices", "query", "status"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create an invoice for John Smith for $500",
                    },
                    {
                        "role": "assistant",
                        "content": "I've created a $500 invoice for John Smith. The invoice is ready to send.",
                    },
                ],
                "topics": ["create", "invoice", "client", "amount"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "I need to bill my client $1200 for consulting work",
                    },
                    {
                        "role": "assistant",
                        "content": "I'll create a $1,200 consulting invoice for your client.",
                    },
                ],
                "topics": ["create", "invoice", "billing", "consulting"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Send the invoice to my client"},
                    {
                        "role": "assistant",
                        "content": "I've sent the invoice to your client via email with a secure payment link.",
                    },
                ],
                "topics": ["send", "invoice", "client", "email"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I send an invoice?"},
                    {
                        "role": "assistant",
                        "content": "To send an invoice, I'll need the client's details and the amount. Then I can create and send it directly.",
                    },
                ],
                "topics": ["send", "invoice", "how-to", "process"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Show me pending invoices"},
                    {
                        "role": "assistant",
                        "content": "Here are your 3 pending invoices awaiting payment.",
                    },
                ],
                "topics": ["invoices", "pending", "status", "filter"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Which invoices haven't been paid yet?",
                    },
                    {
                        "role": "assistant",
                        "content": "You have 4 unpaid invoices totaling $3,200.",
                    },
                ],
                "topics": ["invoices", "unpaid", "pending", "status"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Delete that draft invoice"},
                    {
                        "role": "assistant",
                        "content": "I've deleted the draft invoice for you.",
                    },
                ],
                "topics": ["delete", "invoice", "draft", "remove"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Can you send a payment reminder?"},
                    {
                        "role": "assistant",
                        "content": "I've sent a payment reminder to your client for the overdue invoice.",
                    },
                ],
                "topics": ["reminder", "payment", "overdue", "follow-up"],
                "success": True,
            },
        ]

    @staticmethod
    def get_transaction_conversations() -> List[Dict[str, Any]]:
        """Transaction-related conversations with high variety."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Show me my recent transactions"},
                    {
                        "role": "assistant",
                        "content": "Here are your recent transactions from the past 30 days. You have 45 transactions totaling $2,340 in expenses.",
                    },
                ],
                "topics": ["transactions", "recent", "history", "list"],
                "success": True,
                "user_context": "User with connected bank accounts",
            },
            {
                "messages": [
                    {"role": "user", "content": "What transactions do I have?"},
                    {
                        "role": "assistant",
                        "content": "I'll show you your transaction history with details about each payment.",
                    },
                ],
                "topics": ["transactions", "query", "history"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "List my transactions"},
                    {
                        "role": "assistant",
                        "content": "Here's your complete transaction list organized by date.",
                    },
                ],
                "topics": ["transactions", "list", "history"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Show me transactions from last month"},
                    {
                        "role": "assistant",
                        "content": "Here are all your transactions from last month, totaling $1,890 in activity.",
                    },
                ],
                "topics": ["transactions", "last month", "history", "period"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "What did I spend on groceries?"},
                    {
                        "role": "assistant",
                        "content": "You spent $342.50 on groceries this month across 8 transactions.",
                    },
                ],
                "topics": ["spending", "groceries", "category", "analysis"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "How much did I spend on food this week?",
                    },
                    {
                        "role": "assistant",
                        "content": "This week you spent $127.30 on food and dining.",
                    },
                ],
                "topics": ["spending", "food", "week", "category"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Add a $50 coffee expense for today"},
                    {
                        "role": "assistant",
                        "content": "I've added a $50 coffee expense for today to your Chase checking account.",
                    },
                ],
                "topics": ["add", "expense", "manual", "coffee"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a transaction for $200 consulting income",
                    },
                    {
                        "role": "assistant",
                        "content": "I've created a $200 consulting income transaction for today.",
                    },
                ],
                "topics": ["create", "transaction", "income", "consulting"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Show me my spending by category"},
                    {
                        "role": "assistant",
                        "content": "Here's your spending breakdown: Groceries $342, Dining $198, Transportation $156, Utilities $89.",
                    },
                ],
                "topics": ["spending", "category", "breakdown", "analysis"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "What are my biggest expenses?"},
                    {
                        "role": "assistant",
                        "content": "Your largest expenses this month were: Rent $1,200, Groceries $342, Car payment $280.",
                    },
                ],
                "topics": ["expenses", "largest", "biggest", "analysis"],
                "success": True,
            },
        ]

    @staticmethod
    def get_account_conversations() -> List[Dict[str, Any]]:
        """Account-related conversations with high variety."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "What's my account balance?"},
                    {
                        "role": "assistant",
                        "content": "Your total balance across all accounts is $4,250.32. Chase Checking: $2,100, Savings: $2,150.32.",
                    },
                ],
                "topics": ["balance", "accounts", "total", "summary"],
                "success": True,
                "user_context": "User with 2 connected accounts",
            },
            {
                "messages": [
                    {"role": "user", "content": "How much money do I have?"},
                    {
                        "role": "assistant",
                        "content": "You have a total of $4,250.32 across your connected bank accounts.",
                    },
                ],
                "topics": ["balance", "money", "total", "accounts"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Show me my account balances"},
                    {
                        "role": "assistant",
                        "content": "Here are your current account balances for all connected accounts.",
                    },
                ],
                "topics": ["balances", "accounts", "show", "current"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "What accounts do I have connected?"},
                    {
                        "role": "assistant",
                        "content": "You have 3 accounts connected: Chase Checking, Chase Savings, and Capital One Credit Card.",
                    },
                ],
                "topics": ["accounts", "connected", "list", "banks"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "How many bank accounts are linked?"},
                    {
                        "role": "assistant",
                        "content": "You have 3 bank accounts linked to your Cashly account.",
                    },
                ],
                "topics": ["accounts", "linked", "count", "banks"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "I want to connect my Wells Fargo account",
                    },
                    {
                        "role": "assistant",
                        "content": "I'll help you connect your Wells Fargo account through our secure Plaid integration.",
                    },
                ],
                "topics": ["connect", "account", "Wells Fargo", "plaid"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Add my Bank of America checking account",
                    },
                    {
                        "role": "assistant",
                        "content": "I'll help you add your Bank of America checking account. Let's start the secure connection process.",
                    },
                ],
                "topics": ["add", "account", "Bank of America", "checking"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Link my credit card"},
                    {
                        "role": "assistant",
                        "content": "I can help you link your credit card account for transaction tracking.",
                    },
                ],
                "topics": ["link", "credit card", "account", "connect"],
                "success": True,
            },
        ]

    @staticmethod
    def get_forecasting_conversations() -> List[Dict[str, Any]]:
        """Forecasting-related conversations with high variety."""
        return [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What's my cash flow forecast for next month?",
                    },
                    {
                        "role": "assistant",
                        "content": "Based on your patterns, next month you'll likely have $1,200 income and $980 expenses, ending with $220 positive cash flow.",
                    },
                ],
                "topics": ["forecast", "cash flow", "next month", "prediction"],
                "success": True,
                "user_context": "User with 3+ months of transaction history",
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Predict my expenses for the next 90 days",
                    },
                    {
                        "role": "assistant",
                        "content": "Your predicted expenses for the next 90 days are $2,940 based on your spending patterns.",
                    },
                ],
                "topics": ["predict", "expenses", "90 days", "forecast"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "How much will I spend next month?"},
                    {
                        "role": "assistant",
                        "content": "Based on your trends, you'll likely spend around $1,180 next month.",
                    },
                ],
                "topics": ["spending", "next month", "prediction", "forecast"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Show me my financial forecast"},
                    {
                        "role": "assistant",
                        "content": "Here's your 3-month financial forecast showing projected income, expenses, and cash flow.",
                    },
                ],
                "topics": ["financial", "forecast", "projection", "overview"],
                "success": True,
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What if I increase my marketing budget by $500?",
                    },
                    {
                        "role": "assistant",
                        "content": "If you increase marketing by $500, your monthly expenses would rise to $1,480, reducing your cash flow to $-280.",
                    },
                ],
                "topics": ["what if", "scenario", "marketing", "budget"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Project my income for next quarter"},
                    {
                        "role": "assistant",
                        "content": "Based on your trends, your projected income for next quarter is $3,600.",
                    },
                ],
                "topics": ["project", "income", "quarter", "forecast"],
                "success": True,
            },
        ]

    @staticmethod
    def get_budget_conversations() -> List[Dict[str, Any]]:
        """Budget-related conversations with high variety."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Help me create a budget"},
                    {
                        "role": "assistant",
                        "content": "I'll help create a budget based on your spending patterns. You spend about $1,200/month, so let's set realistic limits.",
                    },
                ],
                "topics": ["create", "budget", "planning", "limits"],
                "success": True,
                "user_context": "User with spending history",
            },
            {
                "messages": [
                    {"role": "user", "content": "Am I staying within my budget?"},
                    {
                        "role": "assistant",
                        "content": "You're currently at 85% of your monthly budget with 8 days remaining. You're on track!",
                    },
                ],
                "topics": ["budget", "tracking", "performance", "status"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Set up a monthly budget for me"},
                    {
                        "role": "assistant",
                        "content": "I'll set up a monthly budget based on your income and spending patterns from the last 3 months.",
                    },
                ],
                "topics": ["setup", "monthly", "budget", "create"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "What's my budget status?"},
                    {
                        "role": "assistant",
                        "content": "Your budget status: Groceries 78% used, Dining 120% over, Transportation 45% used.",
                    },
                ],
                "topics": ["budget", "status", "categories", "tracking"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "I'm going over budget on dining"},
                    {
                        "role": "assistant",
                        "content": "You've spent $320 on dining this month vs your $250 budget. Consider cooking more meals at home.",
                    },
                ],
                "topics": ["over budget", "dining", "overspending", "advice"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Create a budget for groceries"},
                    {
                        "role": "assistant",
                        "content": "Based on your spending, I recommend a $350/month grocery budget. You typically spend $320-380.",
                    },
                ],
                "topics": ["create", "budget", "groceries", "category"],
                "success": True,
            },
        ]

    @staticmethod
    def get_insights_conversations() -> List[Dict[str, Any]]:
        """Insights and analysis-related conversations."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Show me my spending trends"},
                    {
                        "role": "assistant",
                        "content": "Your spending has increased 15% over the last 3 months, mainly in dining and entertainment categories.",
                    },
                ],
                "topics": ["spending", "trends", "analysis", "patterns"],
                "success": True,
                "user_context": "User with transaction history",
            },
            {
                "messages": [
                    {"role": "user", "content": "Analyze my financial patterns"},
                    {
                        "role": "assistant",
                        "content": "You spend most on weekends, have consistent monthly bills, and dining expenses spike on Fridays.",
                    },
                ],
                "topics": ["analyze", "financial", "patterns", "insights"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "What are my spending patterns?"},
                    {
                        "role": "assistant",
                        "content": "Your main patterns: Higher spending on weekends, consistent weekly grocery trips, monthly subscription charges.",
                    },
                ],
                "topics": ["spending", "patterns", "analysis", "behavior"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Show me a financial report"},
                    {
                        "role": "assistant",
                        "content": "Here's your monthly financial report with income, expenses, trends, and category breakdowns.",
                    },
                ],
                "topics": ["financial", "report", "summary", "analysis"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Analyze my business expenses"},
                    {
                        "role": "assistant",
                        "content": "Your business expenses total $890 this month: $400 software, $250 marketing, $240 office supplies.",
                    },
                ],
                "topics": ["analyze", "business", "expenses", "categories"],
                "success": True,
            },
        ]

    @staticmethod
    def get_bank_connection_conversations() -> List[Dict[str, Any]]:
        """Bank connection-related conversations."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Connect my bank account"},
                    {
                        "role": "assistant",
                        "content": "I'll help you securely connect your bank account through Plaid. Which bank would you like to connect?",
                    },
                ],
                "topics": ["connect", "bank", "account", "plaid"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "I want to link my Chase account"},
                    {
                        "role": "assistant",
                        "content": "Great! I'll start the secure connection process for your Chase account through Plaid.",
                    },
                ],
                "topics": ["link", "Chase", "account", "connect"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Add another bank account"},
                    {
                        "role": "assistant",
                        "content": "I can help you add another bank account. Let's go through the secure Plaid connection process.",
                    },
                ],
                "topics": ["add", "bank", "account", "another"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Set up Plaid connection"},
                    {
                        "role": "assistant",
                        "content": "I'll set up a secure Plaid connection to link your bank accounts for automatic transaction sync.",
                    },
                ],
                "topics": ["setup", "plaid", "connection", "secure"],
                "success": True,
            },
        ]

    @staticmethod
    def get_payment_processing_conversations() -> List[Dict[str, Any]]:
        """Payment processing-related conversations."""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Set up Stripe to accept payments"},
                    {
                        "role": "assistant",
                        "content": "I'll help you set up Stripe Connect so you can accept payments through your invoices with a small platform fee.",
                    },
                ],
                "topics": ["setup", "stripe", "payments", "accept"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I process payments?"},
                    {
                        "role": "assistant",
                        "content": "To process payments, you'll need to set up Stripe Connect. Then you can send invoices with payment links.",
                    },
                ],
                "topics": ["process", "payments", "how-to", "stripe"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "I want to accept credit cards"},
                    {
                        "role": "assistant",
                        "content": "I can help you accept credit card payments through Stripe Connect integration.",
                    },
                ],
                "topics": ["accept", "credit cards", "payments", "stripe"],
                "success": True,
            },
            {
                "messages": [
                    {"role": "user", "content": "Set up payment processing"},
                    {
                        "role": "assistant",
                        "content": "I'll help you set up payment processing through Stripe so you can accept payments from clients.",
                    },
                ],
                "topics": ["setup", "payment", "processing", "clients"],
                "success": True,
            },
        ]


# Export all conversation types
def get_all_sample_conversations() -> Dict[str, List[Dict[str, Any]]]:
    """Get all sample conversations organized by intent."""
    return {
        "invoices": SampleConversations.get_invoice_conversations(),
        "transactions": SampleConversations.get_transaction_conversations(),
        "accounts": SampleConversations.get_account_conversations(),
        "forecasting": SampleConversations.get_forecasting_conversations(),
        "budgets": SampleConversations.get_budget_conversations(),
        "insights": SampleConversations.get_insights_conversations(),
        "bank_connection": SampleConversations.get_bank_connection_conversations(),
        "payment_processing": SampleConversations.get_payment_processing_conversations(),
    }
