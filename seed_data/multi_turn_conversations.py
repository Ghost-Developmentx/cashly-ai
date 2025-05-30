"""
Multi-turn conversation samples for realistic intent classification.
These conversations demonstrate context preservation and multiple tool usage.
"""

from typing import Dict, List, Any


class MultiTurnConversations:
    """Realistic multi-turn conversations spanning multiple queries and tools."""

    @staticmethod
    def get_invoice_multi_turn() -> List[Dict[str, Any]]:
        """Invoice conversations with multiple turns and tool calls."""
        return [
            {
                # Create -> Send -> List flow
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you create an invoice for Thomas Bernier at thomas@example.com for $250 for marketing services",
                    },
                    {
                        "role": "assistant",
                        "content": "I've created a draft invoice for Thomas Bernier:\n\n"
                        "**Invoice Details:**\n"
                        "- Client: Thomas Bernier (thomas@example.com)\n"
                        "- Amount: $250.00\n"
                        "- Description: Marketing services\n"
                        "- Due Date: 30 days from today\n\n"
                        "The invoice is ready to send. Would you like me to send it now?",
                        "tools_used": [{"name": "create_invoice", "success": True}],
                    },
                    {"role": "user", "content": "Yes send it"},
                    {
                        "role": "assistant",
                        "content": "✅ I've sent the invoice to Thomas Bernier at thomas@example.com. "
                        "They'll receive an email with a secure payment link. "
                        "You'll be notified when the invoice is viewed or paid.\n\n"
                        "Is there anything else you need help with?",
                        "tools_used": [{"name": "send_invoice", "success": True}],
                    },
                    {"role": "user", "content": "Can you show me all my invoices"},
                    {
                        "role": "assistant",
                        "content": "Here are all your invoices:\n\n"
                        "**Recent Invoices:**\n"
                        "1. Thomas Bernier - $250 (Marketing services) - **Sent** just now\n"
                        "2. Sarah Johnson - $1,200 (Web Design) - **Paid** 3 days ago\n"
                        "3. Mike Chen - $800 (Consulting) - **Pending** 5 days ago\n"
                        "4. Lisa Wong - $450 (Logo Design) - **Draft**\n\n"
                        "You have 4 invoices total: 1 paid ($1,200), 2 pending ($1,050), and 1 draft ($450).",
                        "tools_used": [{"name": "get_invoices", "success": True}],
                    },
                ],
                "topics": [
                    "create_invoice",
                    "send_invoice",
                    "list_invoices",
                    "multi_turn",
                ],
                "success": True,
                "user_context": "Freelancer with active clients",
                "metadata": {
                    "conversation_type": "multi_turn",
                    "tools_sequence": [
                        "create_invoice",
                        "send_invoice",
                        "get_invoices",
                    ],
                    "maintains_context": True,
                },
            },
            {
                # Check status -> Send reminder -> Mark paid flow
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the status of my invoice to Mike Chen?",
                    },
                    {
                        "role": "assistant",
                        "content": "The invoice to Mike Chen for $800 (Consulting services) is currently **Pending**. "
                        "It was sent 5 days ago and hasn't been paid yet. The due date is in 25 days.\n\n"
                        "Would you like me to send a payment reminder?",
                        "tools_used": [{"name": "get_invoices", "success": True}],
                    },
                    {"role": "user", "content": "Yes please send a reminder"},
                    {
                        "role": "assistant",
                        "content": "I've sent a payment reminder to Mike Chen for the $800 invoice. "
                        "The reminder includes the original invoice details and payment link.\n\n"
                        "I'll notify you when there's any activity on this invoice.",
                        "tools_used": [
                            {"name": "send_invoice_reminder", "success": True}
                        ],
                    },
                    {
                        "role": "user",
                        "content": "Actually, he just told me he paid it offline. Can you mark it as paid?",
                    },
                    {
                        "role": "assistant",
                        "content": "I've marked the invoice to Mike Chen as paid. The $800 payment has been recorded.\n\n"
                        "Your updated invoice summary:\n"
                        "- Total Paid: $2,000 (including this payment)\n"
                        "- Pending: $250 (Thomas Bernier)\n"
                        "- Draft: $450 (Lisa Wong)",
                        "tools_used": [{"name": "mark_invoice_paid", "success": True}],
                    },
                ],
                "topics": [
                    "invoice_status",
                    "send_reminder",
                    "mark_paid",
                    "payment_tracking",
                ],
                "success": True,
                "user_context": "Consultant tracking client payments",
                "metadata": {
                    "conversation_type": "multi_turn",
                    "tools_sequence": [
                        "get_invoices",
                        "send_invoice_reminder",
                        "mark_invoice_paid",
                    ],
                    "payment_management": True,
                },
            },
            {
                # Complex flow: List -> Create -> Edit amount -> Send
                "messages": [
                    {"role": "user", "content": "Do I have any draft invoices?"},
                    {
                        "role": "assistant",
                        "content": "Yes, you have 1 draft invoice:\n\n"
                        "**Draft Invoice:**\n"
                        "- Client: Lisa Wong\n"
                        "- Amount: $450\n"
                        "- Description: Logo Design\n"
                        "- Status: Draft (not sent)\n\n"
                        "Would you like to send this invoice or make any changes first?",
                        "tools_used": [{"name": "get_invoices", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "Actually create a new one for Alex Park, alex@techcorp.com for $1500 for app development",
                    },
                    {
                        "role": "assistant",
                        "content": "I've created a new draft invoice:\n\n"
                        "**Invoice Details:**\n"
                        "- Client: Alex Park (alex@techcorp.com)\n"
                        "- Amount: $1,500.00\n"
                        "- Description: App development\n"
                        "- Due Date: 30 days from today\n\n"
                        "The invoice is ready. Would you like to send it or make any adjustments?",
                        "tools_used": [{"name": "create_invoice", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "Wait, I meant $2500 not $1500. Can you update it?",
                    },
                    {
                        "role": "assistant",
                        "content": "I've updated the invoice amount to $2,500. Here's the corrected invoice:\n\n"
                        "**Updated Invoice:**\n"
                        "- Client: Alex Park (alex@techcorp.com)\n"
                        "- Amount: $2,500.00 (updated from $1,500)\n"
                        "- Description: App development\n"
                        "- Due Date: 30 days from today\n\n"
                        "Is this correct now? Ready to send?",
                        "tools_used": [{"name": "update_invoice", "success": True}],
                    },
                    {"role": "user", "content": "Perfect, send it please"},
                    {
                        "role": "assistant",
                        "content": "✅ Invoice sent successfully to Alex Park at alex@techcorp.com!\n\n"
                        "The $2,500 invoice for app development has been delivered with a secure payment link. "
                        "You'll receive notifications for any activity.\n\n"
                        "Your current pending invoices now total $2,750.",
                        "tools_used": [{"name": "send_invoice", "success": True}],
                    },
                ],
                "topics": [
                    "draft_invoices",
                    "create_invoice",
                    "update_invoice",
                    "error_correction",
                ],
                "success": True,
                "user_context": "App developer managing client invoices",
                "metadata": {
                    "conversation_type": "multi_turn",
                    "tools_sequence": [
                        "get_invoices",
                        "create_invoice",
                        "update_invoice",
                        "send_invoice",
                    ],
                    "includes_correction": True,
                },
            },
        ]

    @staticmethod
    def get_transaction_multi_turn() -> List[Dict[str, Any]]:
        """Transaction conversations with multiple turns."""
        return [
            {
                # View transactions -> Analyze spending -> Create transaction
                "messages": [
                    {
                        "role": "user",
                        "content": "Show me my transactions from this week",
                    },
                    {
                        "role": "assistant",
                        "content": "Here are your transactions from this week:\n\n"
                        "**Monday:**\n- Starbucks: -$12.50\n- AWS: -$89.00\n\n"
                        "**Tuesday:**\n- Client Payment: +$1,200.00\n- Whole Foods: -$67.80\n\n"
                        "**Wednesday:**\n- Uber: -$18.90\n- Office Depot: -$45.00\n\n"
                        "**Today:**\n- Lunch: -$15.30\n\n"
                        "Total this week: +$951.50 (Income: $1,200, Expenses: $248.50)",
                        "tools_used": [{"name": "get_transactions", "success": True}],
                    },
                    {"role": "user", "content": "How much have I spent on food?"},
                    {
                        "role": "assistant",
                        "content": "Looking at your food-related expenses this week:\n\n"
                        "**Food & Dining: $95.60**\n"
                        "- Starbucks: $12.50 (Monday)\n"
                        "- Whole Foods: $67.80 (Tuesday)\n"
                        "- Lunch: $15.30 (Today)\n\n"
                        "That's about 38% of your total weekly expenses. "
                        "Your food spending is trending higher than last week's $72.40.",
                        "tools_used": [
                            {"name": "calculate_category_spending", "success": True}
                        ],
                    },
                    {
                        "role": "user",
                        "content": "I forgot to add a $50 dinner expense from yesterday",
                    },
                    {
                        "role": "assistant",
                        "content": "I've added the $50 dinner expense for yesterday (Wednesday).\n\n"
                        "**Updated Food & Dining Total: $145.60**\n"
                        "Your total weekly expenses are now $298.50.\n\n"
                        "Would you like me to categorize this under 'Dining' or would you prefer a different category?",
                        "tools_used": [{"name": "create_transaction", "success": True}],
                    },
                ],
                "topics": [
                    "view_transactions",
                    "analyze_spending",
                    "add_transaction",
                    "category_analysis",
                ],
                "success": True,
                "user_context": "User tracking weekly expenses",
                "metadata": {
                    "conversation_type": "multi_turn",
                    "tools_sequence": [
                        "get_transactions",
                        "calculate_category_spending",
                        "create_transaction",
                    ],
                    "includes_analysis": True,
                },
            },
            {
                # Search -> Update -> Analyze impact
                "messages": [
                    {"role": "user", "content": "Find my Amazon transactions"},
                    {
                        "role": "assistant",
                        "content": "I found 4 Amazon transactions this month:\n\n"
                        "1. Amazon Prime: -$14.99 (Monthly subscription)\n"
                        "2. Amazon Purchase: -$67.43 (Electronics)\n"
                        "3. Amazon Fresh: -$89.20 (Groceries)\n"
                        "4. Amazon Web Services: -$127.50 (Cloud hosting)\n\n"
                        "Total Amazon spending: $299.12",
                        "tools_used": [{"name": "get_transactions", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "The AWS charge should be $227.50, can you fix it?",
                    },
                    {
                        "role": "assistant",
                        "content": "I've updated the AWS transaction from $127.50 to $227.50.\n\n"
                        "**Updated Amazon spending: $399.12** (+$100)\n\n"
                        "This change also affects your monthly totals:\n"
                        "- Technology expenses: now $455.00\n"
                        "- Total monthly expenses: increased by $100",
                        "tools_used": [{"name": "update_transaction", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "What's my technology spending for the month now?",
                    },
                    {
                        "role": "assistant",
                        "content": "Your technology spending for this month:\n\n"
                        "**Total: $455.00**\n"
                        "- Amazon Web Services: $227.50 (Cloud hosting)\n"
                        "- Adobe Creative Suite: $52.99 (Software subscription)\n"
                        "- GitHub Pro: $7.00 (Developer tools)\n"
                        "- Zoom Pro: $14.99 (Video conferencing)\n"
                        "- Domain renewal: $15.00\n"
                        "- Digital Ocean: $12.00 (Server hosting)\n"
                        "- Slack: $8.52 (Team communication)\n"
                        "- AWS (additional): $117.00 (From last invoice)\n\n"
                        "This is 18% of your total monthly expenses.",
                        "tools_used": [
                            {"name": "calculate_category_spending", "success": True}
                        ],
                    },
                ],
                "topics": [
                    "search_transactions",
                    "update_transaction",
                    "spending_analysis",
                    "corrections",
                ],
                "success": True,
                "user_context": "Tech professional tracking business expenses",
                "metadata": {
                    "conversation_type": "multi_turn",
                    "tools_sequence": [
                        "get_transactions",
                        "update_transaction",
                        "calculate_category_spending",
                    ],
                    "includes_correction": True,
                },
            },
        ]

    @staticmethod
    def get_cross_domain_multi_turn() -> List[Dict[str, Any]]:
        """Conversations that span multiple domains/intents."""
        return [
            {
                # Account balance -> Transactions -> Invoice -> Forecast
                "messages": [
                    {"role": "user", "content": "What's my current balance?"},
                    {
                        "role": "assistant",
                        "content": "Your current account balances:\n\n"
                        "**Total: $12,450.75**\n"
                        "- Business Checking: $8,200.50\n"
                        "- Business Savings: $4,250.25\n\n"
                        "Your balance has increased by $2,100 since last week.",
                        "tools_used": [{"name": "get_user_accounts", "success": True}],
                    },
                    {"role": "user", "content": "Nice! What came in this week?"},
                    {
                        "role": "assistant",
                        "content": "Here's your income this week:\n\n"
                        "**Total Income: $3,450.00**\n"
                        "- Sarah Johnson (Invoice #1024): $1,200.00\n"
                        "- TechCorp Monthly Retainer: $1,500.00\n"
                        "- Mike Chen Consulting: $750.00\n\n"
                        "Your expenses were $1,350, giving you a net gain of $2,100.",
                        "tools_used": [{"name": "get_transactions", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "Speaking of invoices, can you create one for a new client Jennifer Wu, jen@startup.io, $1800 for UX design?",
                    },
                    {
                        "role": "assistant",
                        "content": "I've created the invoice for Jennifer Wu:\n\n"
                        "**Invoice #1027**\n"
                        "- Client: Jennifer Wu (jen@startup.io)\n"
                        "- Amount: $1,800.00\n"
                        "- Service: UX design\n"
                        "- Due: 30 days\n\n"
                        "Ready to send! This would bring your pending invoices to $3,050.",
                        "tools_used": [{"name": "create_invoice", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "Great, send it. Also, what's my cash flow looking like for next month?",
                    },
                    {
                        "role": "assistant",
                        "content": "✅ Invoice sent to Jennifer Wu!\n\n"
                        "**Cash Flow Forecast for Next Month:**\n\n"
                        "📈 **Expected Income: $5,250**\n"
                        "- Recurring retainers: $3,000\n"
                        "- Pending invoices likely to be paid: $2,250\n\n"
                        "📉 **Projected Expenses: $3,800**\n"
                        "- Fixed costs: $2,200 (rent, subscriptions, etc.)\n"
                        "- Variable expenses: ~$1,600 (based on 3-month average)\n\n"
                        "💰 **Net Cash Flow: +$1,450**\n\n"
                        "Your cash position should grow to approximately $13,900 by month end.",
                        "tools_used": [
                            {"name": "send_invoice", "success": True},
                            {"name": "forecast_cash_flow", "success": True},
                        ],
                    },
                ],
                "topics": [
                    "balance",
                    "income_analysis",
                    "invoice_creation",
                    "cash_flow_forecast",
                ],
                "success": True,
                "user_context": "Freelance designer managing finances",
                "metadata": {
                    "conversation_type": "cross_domain",
                    "tools_sequence": [
                        "get_user_accounts",
                        "get_transactions",
                        "create_invoice",
                        "send_invoice",
                        "forecast_cash_flow",
                    ],
                    "spans_intents": [
                        "accounts",
                        "transactions",
                        "invoices",
                        "forecasting",
                    ],
                },
            },
            {
                # Budget check -> Overspending -> Transactions -> Advice
                "messages": [
                    {"role": "user", "content": "Am I over budget anywhere?"},
                    {
                        "role": "assistant",
                        "content": "Yes, you're over budget in 2 categories:\n\n"
                        "🔴 **Dining: 142% of budget**\n"
                        "- Spent: $426 / Budget: $300\n"
                        "- Over by: $126\n\n"
                        "🔴 **Entertainment: 118% of budget**\n"
                        "- Spent: $177 / Budget: $150\n"
                        "- Over by: $27\n\n"
                        "Your other categories are on track. Total overage: $153",
                        "tools_used": [{"name": "generate_budget", "success": True}],
                    },
                    {"role": "user", "content": "Wow, show me my dining transactions"},
                    {
                        "role": "assistant",
                        "content": "Here are your dining transactions this month:\n\n"
                        "**Total: $426 (17 transactions)**\n\n"
                        "Top expenses:\n"
                        "- The French Laundry: $128.50 (business dinner)\n"
                        "- Multiple Uber Eats: $94.30 (6 orders)\n"
                        "- Starbucks: $67.80 (12 visits)\n"
                        "- Various restaurants: $135.40\n\n"
                        "Your dining expenses increased 40% from last month's $304.",
                        "tools_used": [{"name": "get_transactions", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "That's a lot of Uber Eats. How can I reduce this?",
                    },
                    {
                        "role": "assistant",
                        "content": "Here's a plan to reduce your dining expenses:\n\n"
                        "**Quick Wins:**\n"
                        "1. **Meal prep Sundays**: Could save $60-80/month on Uber Eats\n"
                        "2. **Coffee at home**: Reducing Starbucks to 2x/week saves $40\n"
                        "3. **Lunch limits**: Set a $15 max for workday lunches\n\n"
                        "**Smart Strategies:**\n"
                        "- Use grocery delivery instead (typically 50% less than prepared food)\n"
                        "- Designate 1 Uber Eats day per week as a treat\n"
                        "- Track business meals separately for tax deductions\n\n"
                        "These changes could bring dining back to your $300 budget while still enjoying eating out!",
                        "tools_used": [{"name": "analyze_trends", "success": True}],
                    },
                ],
                "topics": [
                    "budget_check",
                    "overspending",
                    "transaction_analysis",
                    "financial_advice",
                ],
                "success": True,
                "user_context": "Professional trying to control spending",
                "metadata": {
                    "conversation_type": "advisory",
                    "tools_sequence": [
                        "generate_budget",
                        "get_transactions",
                        "analyze_trends",
                    ],
                    "includes_recommendations": True,
                },
            },
            {
                # Complex business flow: Check stripe status -> Create invoice -> Check balance -> Forecast with new income
                "messages": [
                    {"role": "user", "content": "Is my Stripe account set up?"},
                    {
                        "role": "assistant",
                        "content": "Yes! Your Stripe Connect account is active and ready to accept payments.\n\n"
                        "**Account Status:**\n"
                        "✅ Charges enabled\n"
                        "✅ Payouts enabled\n"
                        "✅ Platform fee: 2.9%\n\n"
                        "You can send invoices with payment links and accept credit cards directly.",
                        "tools_used": [
                            {"name": "check_stripe_connect_status", "success": True}
                        ],
                    },
                    {
                        "role": "user",
                        "content": "Perfect. Create an invoice for BigCorp, accounting@bigcorp.com for $5000 for consulting services due in 15 days",
                    },
                    {
                        "role": "assistant",
                        "content": "I've created a substantial invoice for BigCorp:\n\n"
                        "**Invoice #1028**\n"
                        "- Client: BigCorp (accounting@bigcorp.com)\n"
                        "- Amount: $5,000.00\n"
                        "- Service: Consulting services\n"
                        "- Due: 15 days (expedited)\n"
                        "- Platform fee: $145 (2.9%)\n"
                        "- Your net: $4,855\n\n"
                        "This is your largest invoice this month! Ready to send?",
                        "tools_used": [{"name": "create_invoice", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "Yes send it. How much will I have if they pay on time?",
                    },
                    {
                        "role": "assistant",
                        "content": "✅ Invoice sent to BigCorp! With 15-day terms, you should receive payment by [date].\n\n"
                        "**Projected Balance After Payment:**\n\n"
                        "Current balance: $12,450.75\n"
                        "+ BigCorp payment (net): $4,855.00\n"
                        "= **New balance: $17,305.75**\n\n"
                        "This would be a 39% increase in your cash position! Based on BigCorp's history, "
                        "they typically pay within 10-12 days.",
                        "tools_used": [
                            {"name": "send_invoice", "success": True},
                            {"name": "get_user_accounts", "success": True},
                        ],
                    },
                    {
                        "role": "user",
                        "content": "Awesome! Update my forecast with this new income",
                    },
                    {
                        "role": "assistant",
                        "content": "I've updated your forecast with the BigCorp payment:\n\n"
                        "**Revised 30-Day Cash Flow Forecast:**\n\n"
                        "📈 **Income: $10,250** (was $5,250)\n"
                        "- BigCorp invoice: $4,855 (net after fees)\n"
                        "- Existing recurring: $3,000\n"
                        "- Other pending: $2,395\n\n"
                        "📉 **Expenses: $3,800** (unchanged)\n\n"
                        "💰 **Net Cash Flow: +$6,450** (was +$1,450)\n\n"
                        "**Key Insights:**\n"
                        "- Cash position could reach $18,900 by month end\n"
                        "- This gives you runway for 4.7 months at current burn rate\n"
                        "- Consider moving $5,000 to savings for tax reserves",
                        "tools_used": [{"name": "forecast_cash_flow", "success": True}],
                    },
                ],
                "topics": [
                    "stripe_status",
                    "large_invoice",
                    "balance_projection",
                    "forecast_update",
                ],
                "success": True,
                "user_context": "Consultant landing large client",
                "metadata": {
                    "conversation_type": "business_planning",
                    "tools_sequence": [
                        "check_stripe_connect_status",
                        "create_invoice",
                        "send_invoice",
                        "get_user_accounts",
                        "forecast_cash_flow",
                    ],
                    "financial_planning": True,
                },
            },
        ]

    @staticmethod
    def get_error_handling_multi_turn() -> List[Dict[str, Any]]:
        """Conversations showing error handling and corrections."""
        return [
            {
                # Typos, corrections, clarifications
                "messages": [
                    {"role": "user", "content": "Show me my invocies"},  # Typo
                    {
                        "role": "assistant",
                        "content": "I'll show you your invoices. You currently have 6 invoices:\n\n"
                        "- 2 Paid: $3,200\n"
                        "- 3 Pending: $2,100\n"
                        "- 1 Draft: $500\n\n"
                        "Would you like to see the detailed list?",
                        "tools_used": [{"name": "get_invoices", "success": True}],
                    },
                    {
                        "role": "user",
                        "content": "Create invoice for Bob",  # Missing information
                    },
                    {
                        "role": "assistant",
                        "content": "I'll help you create an invoice for Bob. I need a few more details:\n\n"
                        "- Bob's email address\n"
                        "- Invoice amount\n"
                        "- Description of services/products\n\n"
                        "For example: 'Create invoice for Bob at bob@example.com for $500 for consulting'",
                        "tools_used": [],
                    },
                    {
                        "role": "user",
                        "content": "bob@company.com, $750 for website updates",
                    },
                    {
                        "role": "assistant",
                        "content": "Perfect! I've created the invoice:\n\n"
                        "**Invoice Details:**\n"
                        "- Client: Bob (bob@company.com)\n"
                        "- Amount: $750.00\n"
                        "- Description: Website updates\n"
                        "- Due: 30 days\n\n"
                        "Would you like to send it now?",
                        "tools_used": [{"name": "create_invoice", "success": True}],
                    },
                ],
                "topics": [
                    "error_handling",
                    "typo_correction",
                    "missing_info",
                    "clarification",
                ],
                "success": True,
                "user_context": "User making common input errors",
                "metadata": {
                    "conversation_type": "error_handling",
                    "includes_corrections": True,
                    "user_guidance": True,
                },
            }
        ]

    @staticmethod
    def get_all_multi_turn_conversations() -> Dict[str, List[Dict[str, Any]]]:
        """Get all multi-turn conversations organized by primary intent."""
        return {
            "invoices": MultiTurnConversations.get_invoice_multi_turn(),
            "transactions": MultiTurnConversations.get_transaction_multi_turn(),
            "cross_domain": MultiTurnConversations.get_cross_domain_multi_turn(),
            "error_handling": MultiTurnConversations.get_error_handling_multi_turn(),
        }
