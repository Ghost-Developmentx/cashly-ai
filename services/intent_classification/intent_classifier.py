import os
import logging
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Intent classification service to route financial queries to appropriate assistants.
    Uses both traditional ML (sklearn) and modern NLP (transformers) approaches.
    """

    def __init__(self, model_path: str = "models/intent_classifier"):
        self.model_path = model_path
        self.intent_categories = {
            "transactions": [
                "show transactions",
                "view transactions",
                "list transactions",
                "add transaction",
                "create transaction",
                "new transaction",
                "edit transaction",
                "update transaction",
                "modify transaction",
                "delete transaction",
                "remove transaction",
                "categorize transactions",
                "organize transactions",
                "spending on",
                "expenses for",
                "income from",
                "spent on",
                "expense to",
                "add expense",
                "record expense",
            ],
            "accounts": [
                "connect bank account",
                "link account",
                "add bank",
                "view accounts",
                "show accounts",
                "my accounts",
                "account balance",
                "balance of",
                "how much in",
                "disconnect account",
                "remove account",
                "unlink",
                "plaid connection",
                "bank connection",
                "total balance",
            ],
            "invoices": [
                "create invoice",
                "send invoice",
                "new invoice",
                "make invoice",
                "view invoices",
                "show invoices",
                "my invoices",
                "send reminder",
                "payment reminder",
                "follow up",
                "mark paid",
                "mark as paid",
                "invoice paid",
                "stripe connect",
                "payment processing",
                "accept payments",
                "invoice for",
                "bill client",
                "bill for",
                "charge client",
            ],
            "forecasting": [
                "cash flow forecast",
                "predict cash flow",
                "future cash",
                "predict spending",
                "forecast expenses",
                "future expenses",
                "scenario planning",
                "what if",
                "projection",
                "future balance",
                "projected balance",
                "forecast balance",
                "cash flow look like",
                "predict expenses",
                "forecast",
                "predict",
            ],
            "budgets": [
                "create budget",
                "make budget",
                "budget plan",
                "monthly budget",
                "budget recommendations",
                "budget advice",
                "suggested budget",
                "spending limits",
                "budget limits",
                "expense limits",
                "budget analysis",
                "budget review",
                "budget performance",
                "help me budget",
                "budget help",
                "budgeting",
            ],
            "insights": [
                "spending trends",
                "expense trends",
                "financial trends",
                "financial insights",
                "money insights",
                "spending insights",
                "anomaly detection",
                "unusual transactions",
                "strange spending",
                "category analysis",
                "spending by category",
                "category breakdown",
                "analyze spending",
                "financial analysis",
                "spending analysis",
            ],
            "general": [
                "hello",
                "hi",
                "help",
                "how to",
                "what is",
                "explain",
                "financial advice",
                "money advice",
                "suggestions",
                "greeting",
                "thanks",
                "thank you",
            ],
        }

        # Models
        self.sklearn_model = None
        self.tfidf_vectorizer = None
        self.huggingface_classifier = None

        # Load or initialize models
        self._load_models()

    def _load_models(self):
        """Load existing models or initialize new ones."""
        try:
            # Try to load sklearn model
            if os.path.exists(f"{self.model_path}/sklearn_model.joblib"):
                self.sklearn_model = joblib.load(
                    f"{self.model_path}/sklearn_model.joblib"
                )
                self.tfidf_vectorizer = joblib.load(
                    f"{self.model_path}/tfidf_vectorizer.joblib"
                )
                logger.info("Loaded existing sklearn intent classifier")

            # Try to load HuggingFace model
            if os.path.exists(f"{self.model_path}/huggingface"):
                self.huggingface_classifier = pipeline(
                    "text-classification",
                    model=f"{self.model_path}/huggingface",
                    tokenizer=f"{self.model_path}/huggingface",
                )
                logger.info("Loaded existing HuggingFace intent classifier")
            else:
                # Use pre-trained model as fallback
                try:
                    self.huggingface_classifier = pipeline(
                        "text-classification", model="microsoft/DialoGPT-medium"
                    )
                    logger.info("Using pre-trained HuggingFace model")
                except Exception as e:
                    logger.warning(f"Could not load HuggingFace model: {e}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._train_default_model()

    def _generate_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate training data from intent categories."""
        texts = []
        labels = []

        # Create training examples from predefined patterns
        for intent, patterns in self.intent_categories.items():
            for pattern in patterns:
                # Add the pattern as-is
                texts.append(pattern)
                labels.append(intent)

                # Add variations
                texts.append(f"Can you {pattern}?")
                labels.append(intent)

                texts.append(f"I want to {pattern}")
                labels.append(intent)

                texts.append(f"Help me {pattern}")
                labels.append(intent)

        # Add some realistic financial queries
        realistic_queries = [
            ("Show me my transactions from last month", "transactions"),
            ("How much did I spend on groceries?", "transactions"),
            ("Connect my Wells Fargo account", "accounts"),
            ("What's my checking account balance?", "accounts"),
            ("Create an invoice for $500", "invoices"),
            ("Send a payment reminder to John", "invoices"),
            ("What will my cash flow look like next month?", "forecasting"),
            ("Predict my expenses for Q2", "forecasting"),
            ("Help me create a monthly budget", "budgets"),
            ("Am I overspending on entertainment?", "budgets"),
            ("Analyze my spending trends", "insights"),
            ("Are there any unusual transactions?", "insights"),
            ("What is a good savings rate?", "general"),
            ("How do I improve my credit score?", "general"),
        ]

        for query, intent in realistic_queries:
            texts.append(query)
            labels.append(intent)

        return texts, labels

    def _train_default_model(self):
        """Train a basic sklearn model with generated data."""
        logger.info("Training default intent classification model...")

        texts, labels = self._generate_training_data()

        # Train TF-IDF + Logistic Regression model
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english"
        )

        X = self.tfidf_vectorizer.fit_transform(texts)

        self.sklearn_model = LogisticRegression(max_iter=1000)
        self.sklearn_model.fit(X, labels)

        # Save the model
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.sklearn_model, f"{self.model_path}/sklearn_model.joblib")
        joblib.dump(self.tfidf_vectorizer, f"{self.model_path}/tfidf_vectorizer.joblib")

        logger.info("Default model trained and saved")

    def classify_intent(self, query: str) -> Dict[str, any]:
        """
        Classify the intent of a financial query.

        Args:
            query: The user's query text

        Returns:
            Dictionary with intent, confidence, and reasoning
        """
        query = query.strip().lower()

        if not query:
            return {
                "intent": "general",
                "confidence": 0.5,
                "method": "default",
                "reasoning": "Empty query",
            }

        # Try sklearn model first (faster)
        sklearn_result = self._classify_with_sklearn(query)

        # For high confidence, return sklearn result
        if sklearn_result["confidence"] > 0.8:
            return sklearn_result

        # For lower confidence, try rule-based approach
        rule_result = self._classify_with_rules(query)

        # Return the result with higher confidence
        if rule_result["confidence"] > sklearn_result["confidence"]:
            return rule_result
        else:
            return sklearn_result

    def _classify_with_sklearn(self, query: str) -> Dict[str, any]:
        """Classify using a sklearn model."""
        if not self.sklearn_model or not self.tfidf_vectorizer:
            return {
                "intent": "general",
                "confidence": 0.3,
                "method": "sklearn_fallback",
                "reasoning": "Model not available",
            }

        try:
            X = self.tfidf_vectorizer.transform([query])
            probabilities = self.sklearn_model.predict_proba(X)[0]
            intent_idx = np.argmax(probabilities)
            intent = self.sklearn_model.classes_[intent_idx]
            confidence = probabilities[intent_idx]

            return {
                "intent": intent,
                "confidence": float(confidence),
                "method": "sklearn",
                "reasoning": f"ML classification with {confidence:.2%} confidence",
            }
        except Exception as e:
            logger.error(f"Sklearn classification error: {e}")
            return {
                "intent": "general",
                "confidence": 0.3,
                "method": "sklearn_error",
                "reasoning": str(e),
            }

    def _classify_with_rules(self, query: str) -> Dict[str, any]:
        """Classify using rule-based approach."""
        query_lower = query.lower()

        # Check for exact keyword matches (more specific patterns first)
        for intent, keywords in self.intent_categories.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    # Calculate confidence based on keyword match quality and intent specificity
                    confidence = 0.8 if len(keyword.split()) > 1 else 0.7

                    # Boost confidence for very specific terms
                    specific_terms = {
                        "invoice": 0.1,
                        "forecast": 0.1,
                        "budget": 0.1,
                        "connect": 0.1,
                        "account": 0.05,
                    }

                    for term, boost in specific_terms.items():
                        if term in keyword.lower():
                            confidence += boost
                            break

                    return {
                        "intent": intent,
                        "confidence": min(confidence, 0.9),
                        "method": "rules",
                        "reasoning": f'Matched keyword: "{keyword}"',
                    }

        # Enhanced financial action words with better patterns
        action_patterns = {
            "invoices": ["invoice", "bill", "payment", "reminder", "stripe", "client"],
            "forecasting": [
                "forecast",
                "predict",
                "future",
                "projection",
                "what if",
                "cash flow",
            ],
            "budgets": ["budget", "limit", "plan", "budgeting"],
            "transactions": [
                "show",
                "list",
                "view",
                "add",
                "create",
                "edit",
                "delete",
                "spent",
                "expense",
            ],
            "accounts": ["connect", "link", "balance", "account"],
            "insights": ["analyze", "trend", "insight", "unusual", "anomaly"],
        }

        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in action_patterns.items():
            score = 0
            matched_patterns = []
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1
                    matched_patterns.append(pattern)

            if score > 0:
                # Base confidence + bonus for multiple matches
                confidence = 0.6 + (score * 0.05)
                intent_scores[intent] = {
                    "confidence": min(confidence, 0.8),
                    "matches": matched_patterns,
                    "score": score,
                }

        # Return the highest scoring intent
        if intent_scores:
            best_intent = max(
                intent_scores.keys(), key=lambda x: intent_scores[x]["confidence"]
            )
            best_score = intent_scores[best_intent]

            return {
                "intent": best_intent,
                "confidence": best_score["confidence"],
                "method": "rules_pattern",
                "reasoning": f'Matched {best_score["score"]} patterns: {", ".join(best_score["matches"])}',
            }

        # Default to a general
        return {
            "intent": "general",
            "confidence": 0.4,
            "method": "rules_default",
            "reasoning": "No specific patterns matched",
        }

    def train_with_conversation_data(self, conversations: List[Dict]) -> Dict[str, any]:
        """
        Train the model with real conversation data.

        Args:
            conversations: List of conversation dictionaries with 'messages' and labels

        Returns:
            Training results and metrics
        """
        logger.info(f"Training with {len(conversations)} conversations...")

        # Extract training data from conversations
        texts = []
        labels = []

        for conv in conversations:
            for message in conv.get("messages", []):
                if message.get("role") == "user":
                    content = message.get("content", "").strip()
                    if content and conv.get("intent_label"):
                        texts.append(content)
                        labels.append(conv["intent_label"])

        if len(texts) < 10:
            logger.warning("Not enough training data, using generated data")
            texts, labels = self._generate_training_data()

        # Train new model
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 3), stop_words="english"
        )

        X = self.tfidf_vectorizer.fit_transform(texts)

        self.sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
        self.sklearn_model.fit(X, labels)

        # Save an updated model
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.sklearn_model, f"{self.model_path}/sklearn_model.joblib")
        joblib.dump(self.tfidf_vectorizer, f"{self.model_path}/tfidf_vectorizer.joblib")

        # Calculate some basic metrics
        train_accuracy = self.sklearn_model.score(X, labels)

        return {
            "status": "success",
            "training_samples": len(texts),
            "unique_intents": len(set(labels)),
            "train_accuracy": train_accuracy,
            "model_path": self.model_path,
        }

    def get_intent_suggestions(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, any]]:
        """Get top K intent predictions with confidence scores."""
        if not self.sklearn_model:
            return [self.classify_intent(query)]

        try:
            X = self.tfidf_vectorizer.transform([query])
            probabilities = self.sklearn_model.predict_proba(X)[0]

            # Get top K intents
            top_indices = np.argsort(probabilities)[-top_k:][::-1]

            suggestions = []
            for idx in top_indices:
                intent = self.sklearn_model.classes_[idx]
                confidence = probabilities[idx]

                suggestions.append(
                    {
                        "intent": intent,
                        "confidence": float(confidence),
                        "method": "sklearn_multi",
                    }
                )

            return suggestions

        except Exception as e:
            logger.error(f"Error getting intent suggestions: {e}")
            return [self.classify_intent(query)]
