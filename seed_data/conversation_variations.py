"""
Generates variations of sample conversations to increase training data diversity.
Uses different phrasings, contexts, and user types.
"""

import random
from typing import Dict, List, Any


class ConversationVariations:
    """Generates variations of conversations for better intent classification training."""

    def __init__(self):
        self.user_types = [
            "freelancer",
            "small_business",
            "contractor",
            "consultant",
            "agency_owner",
            "startup_founder",
            "solopreneur",
        ]

        self.politeness_variations = {
            "polite": ["please", "could you", "would you mind", "if you don't mind"],
            "casual": ["hey", "yo", "sup", "quick question"],
            "direct": ["show me", "give me", "I want", "I need"],
            "professional": ["I would like to", "could you provide", "please display"],
        }

        self.time_contexts = [
            "this month",
            "last month",
            "this week",
            "last week",
            "today",
            "yesterday",
            "this year",
            "last year",
            "past 30 days",
            "past 7 days",
            "past 90 days",
        ]

        self.urgency_markers = [
            "urgent",
            "ASAP",
            "quickly",
            "right away",
            "immediately",
            "when you can",
            "no rush",
            "whenever",
            "at your convenience",
        ]

    def generate_invoice_variations(
        self, base_conversations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate variations for invoice conversations."""
        variations = []

        # Invoice listing variations
        invoice_list_queries = [
            "Show me all my invoices",
            "Can I see my invoice list?",
            "Display all invoices",
            "What invoices do I have?",
            "Give me my complete invoice list",
            "I want to see all my bills",
            "Show invoice summary",
            "List every invoice",
            "Pull up all my invoices",
            "Can you show my invoicing history?",
            "I need to see all invoices",
            "Display my invoice dashboard",
            "Show me the invoice overview",
        ]

        # Create variations for each query
        for query in invoice_list_queries:
            variations.append(
                {
                    "messages": [
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": self._generate_invoice_response(),
                        },
                    ],
                    "topics": ["invoices", "list", "display", "overview"],
                    "success": True,
                    "user_context": random.choice(
                        [
                            "Freelancer with Stripe Connect",
                            "Small business with multiple clients",
                            "Consultant with recurring invoices",
                        ]
                    ),
                }
            )

        # Status-specific variations
        status_queries = [
            (
                "pending",
                [
                    "Show pending invoices",
                    "Which invoices are unpaid?",
                    "List overdue invoices",
                ],
            ),
            (
                "paid",
                [
                    "Show paid invoices",
                    "Which invoices were paid?",
                    "List completed payments",
                ],
            ),
            (
                "draft",
                [
                    "Show draft invoices",
                    "What invoices are in draft?",
                    "List unsent invoices",
                ],
            ),
        ]

        for status, queries in status_queries:
            for query in queries:
                variations.append(
                    {
                        "messages": [
                            {"role": "user", "content": query},
                            {
                                "role": "assistant",
                                "content": f"Here are your {status} invoices.",
                            },
                        ],
                        "topics": ["invoices", status, "filter", "status"],
                        "success": True,
                    }
                )

        return variations

    def generate_transaction_variations(
        self, base_conversations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate variations for transaction conversations."""
        variations = []

        # Transaction listing with time contexts
        base_queries = [
            "Show me transactions",
            "List my transactions",
            "Display transaction history",
            "What transactions do I have?",
            "Give me my spending history",
            "Show my expenses",
        ]

        for query in base_queries:
            for time_context in self.time_contexts:
                variations.append(
                    {
                        "messages": [
                            {"role": "user", "content": f"{query} from {time_context}"},
                            {
                                "role": "assistant",
                                "content": f"Here are your transactions from {time_context}.",
                            },
                        ],
                        "topics": ["transactions", "history", "time_filter"],
                        "success": True,
                    }
                )

        # Category-specific spending queries
        categories = [
            "groceries",
            "food",
            "dining",
            "restaurants",
            "coffee",
            "gas",
            "transportation",
            "utilities",
            "rent",
            "mortgage",
            "insurance",
            "software",
            "subscriptions",
            "marketing",
            "office supplies",
            "equipment",
            "consulting",
            "freelance",
        ]

        spending_patterns = [
            "How much did I spend on {}?",
            "What did I spend on {} {}?",
            "Show me {} expenses",
            "Display {} spending",
            "How much went to {}?",
            "What's my {} budget usage?",
        ]

        for category in categories:
            for pattern in spending_patterns:
                for time_context in self.time_contexts[:5]:  # Limit combinations
                    query = (
                        pattern.format(category, time_context)
                        if "{} {}" in pattern
                        else pattern.format(category)
                    )
                    variations.append(
                        {
                            "messages": [
                                {"role": "user", "content": query},
                                {
                                    "role": "assistant",
                                    "content": f"You spent $X on {category} {time_context}.",
                                },
                            ],
                            "topics": ["spending", "category", category, "analysis"],
                            "success": True,
                        }
                    )

        return variations

    def generate_account_variations(
        self, base_conversations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate variations for account conversations."""
        variations = []

        # Balance inquiry variations
        balance_queries = [
            "What's my balance?",
            "How much money do I have?",
            "Show my account balance",
            "Display current balance",
            "What's my total balance?",
            "How much is in my accounts?",
            "Check my balance",
            "Give me my balance",
            "Show account balances",
            "What's my financial position?",
            "How much cash do I have?",
            "Display my money",
            "Show my funds",
            "What's my current balance?",
        ]

        for query in balance_queries:
            # Add politeness variations
            for politeness_type, markers in self.politeness_variations.items():
                marker = random.choice(markers)
                polite_query = (
                    f"{marker} {query.lower()}"
                    if politeness_type != "direct"
                    else query
                )

                variations.append(
                    {
                        "messages": [
                            {"role": "user", "content": polite_query},
                            {
                                "role": "assistant",
                                "content": "Your total balance across all accounts is $X,XXX.XX",
                            },
                        ],
                        "topics": ["balance", "accounts", "total", "money"],
                        "success": True,
                        "user_context": f"{politeness_type.title()} user query",
                    }
                )

        # Account connection variations
        banks = [
            "Chase",
            "Bank of America",
            "Wells Fargo",
            "Citi",
            "Capital One",
            "US Bank",
            "PNC",
            "TD Bank",
            "Ally",
            "Discover",
            "USAA",
        ]

        connection_patterns = [
            "Connect my {} account",
            "Link my {} bank",
            "Add my {} account",
            "I want to connect {}",
            "Set up {} connection",
            "Integrate my {} account",
        ]

        for bank in banks:
            for pattern in connection_patterns:
                query = pattern.format(bank)
                variations.append(
                    {
                        "messages": [
                            {"role": "user", "content": query},
                            {
                                "role": "assistant",
                                "content": f"I'll help you connect your {bank} account securely.",
                            },
                        ],
                        "topics": ["connect", "account", bank.lower(), "bank"],
                        "success": True,
                    }
                )

        return variations

    def generate_contextual_variations(
        self, intent: str, base_conversations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate contextual variations based on user types and scenarios."""
        variations = []

        # User type specific contexts
        user_contexts = {
            "freelancer": {
                "invoices": [
                    "client invoices",
                    "project billing",
                    "freelance payments",
                ],
                "transactions": [
                    "business expenses",
                    "client payments",
                    "project costs",
                ],
                "accounts": ["business account", "personal account", "client payments"],
            },
            "small_business": {
                "invoices": ["customer invoices", "service billing", "product sales"],
                "transactions": ["business expenses", "revenue", "operational costs"],
                "accounts": [
                    "business checking",
                    "business savings",
                    "payroll account",
                ],
            },
            "consultant": {
                "invoices": [
                    "consulting invoices",
                    "hourly billing",
                    "project invoices",
                ],
                "transactions": [
                    "consulting expenses",
                    "client payments",
                    "business costs",
                ],
                "accounts": ["consulting account", "expense account", "client account"],
            },
        }

        # Generate user-specific variations
        for user_type, contexts in user_contexts.items():
            if intent in contexts:
                for context in contexts[intent]:
                    variations.extend(
                        self._create_user_context_variations(intent, user_type, context)
                    )

        return variations

    def _create_user_context_variations(
        self, intent: str, user_type: str, context: str
    ) -> List[Dict[str, Any]]:
        """Create variations for specific user contexts."""
        variations = []

        context_queries = {
            "invoices": [
                f"Show me my {context}",
                f"List all {context}",
                f"Display my {context}",
                f"What {context} do I have?",
                f"Give me {context} overview",
            ],
            "transactions": [
                f"Show {context}",
                f"List my {context}",
                f"Display {context} history",
                f"What are my {context}?",
                f"Show me {context} breakdown",
            ],
            "accounts": [
                f"What's my {context} balance?",
                f"Show {context} details",
                f"Display {context} information",
                f"How much is in my {context}?",
                f"Connect my {context}",
            ],
        }

        if intent in context_queries:
            for query in context_queries[intent]:
                variations.append(
                    {
                        "messages": [
                            {"role": "user", "content": query},
                            {
                                "role": "assistant",
                                "content": f"Here's your {context} information.",
                            },
                        ],
                        "topics": [intent, context, user_type],
                        "success": True,
                        "user_context": f"{user_type.title()} asking about {context}",
                    }
                )

        return variations

    def _generate_invoice_response(self) -> str:
        """Generate varied invoice response."""
        responses = [
            "Here are all your invoices organized by status.",
            "I found your complete invoice list with current status for each.",
            "Here's your invoice overview with payment status.",
            "Your invoices are displayed below, sorted by date.",
            "Here's a summary of all your invoices and their current status.",
        ]
        return random.choice(responses)

    def add_urgency_variations(
        self, conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add urgency markers to create more varied training data."""
        urgent_variations = []

        for conv in conversations[:10]:  # Only modify first 10 to avoid too much data
            original_query = conv["messages"][0]["content"]

            # Add urgency markers
            for urgency in self.urgency_markers[:3]:  # Limit to 3 urgency types
                urgent_query = f"{original_query} {urgency}"

                urgent_conv = conv.copy()
                urgent_conv["messages"] = [
                    {"role": "user", "content": urgent_query},
                    urgent_conv["messages"][1],  # Keep same response
                ]
                urgent_conv["topics"] = conv["topics"] + ["urgent"]
                urgent_variations.append(urgent_conv)

        return urgent_variations

    def generate_all_variations(
        self, sample_conversations: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all variations for the sample conversations."""
        all_variations = {}

        for intent, conversations in sample_conversations.items():
            print(f"Generating variations for {intent}...")

            # Start with base conversations
            variations = conversations.copy()

            # Add intent-specific variations
            if intent == "invoices":
                variations.extend(self.generate_invoice_variations(conversations))
            elif intent == "transactions":
                variations.extend(self.generate_transaction_variations(conversations))
            elif intent == "accounts":
                variations.extend(self.generate_account_variations(conversations))

            # Add contextual variations for all intents
            variations.extend(
                self.generate_contextual_variations(intent, conversations)
            )

            # Add urgency variations (limited to avoid explosion)
            variations.extend(self.add_urgency_variations(conversations))

            all_variations[intent] = variations
            print(f"  Generated {len(variations)} total conversations for {intent}")

        return all_variations
