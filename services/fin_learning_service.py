import os
import json
import logging
import anthropic
import re
import numpy as np
from typing import Dict, List, Any, Optional
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


class FinLearningService:
    """
    Service to analyze feedback and improve Fin's capabilities over time.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.system_prompt_template = self._load_system_prompt_template()
        self.example_exchanges = self._load_example_exchanges()

        # Load NLP models
        self._load_nlp_models()

        # Load or initialize intent classification data
        self.intents_data = self._load_intents_data()

    def _load_nlp_models(self):
        """Load NLP models for text processing"""
        # Load spaCy model for general NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            # Fallback to a smaller model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy fallback model")
            except:
                logger.error("Could not load any spaCy model")
                self.nlp = None

        # Load Named Entity Recognition model from HuggingFace
        try:
            model_name = "flair/ner-english-ontonotes-large"
            self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.ner_model,
                tokenizer=self.ner_tokenizer,
            )
            logger.info("Loaded HuggingFace NER model")
        except Exception as e:
            logger.error(f"Error loading HuggingFace NER model: {str(e)}")
            self.ner_pipeline = None

        # Initialize TF-IDF vectorizer for intent classification
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), max_features=5000, stop_words="english"
        )

    @staticmethod
    def _load_intents_data() -> Dict[str, List[str]]:
        """Load intent classification data"""
        try:
            with open("models/intents_data.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default intents data if file doesn't exist
            return {
                "forecast": [
                    "forecast my cash flow",
                    "predict my finances",
                    "what will my balance be next month",
                    "project my expenses",
                    "financial outlook",
                    "future spending prediction",
                    "what if I spend more",
                    "how will my finances look",
                ],
                "budget": [
                    "create a budget",
                    "budget recommendation",
                    "suggest a budget",
                    "how should I budget",
                    "spending limits",
                    "allocate my income",
                    "budget for groceries",
                    "help me budget better",
                ],
                "analysis": [
                    "analyze my spending",
                    "spending patterns",
                    "expense breakdown",
                    "where is my money going",
                    "spending trends",
                    "expense analysis",
                    "show me spending patterns",
                ],
                "comparison": [
                    "compare my spending",
                    "how does my spending compare",
                    "spending this month versus last month",
                    "expenses compared to income",
                    "spending difference",
                    "budget vs actual",
                ],
                "recommendation": [
                    "recommend savings",
                    "how can I save more",
                    "should I spend less",
                    "advise on reducing expenses",
                    "recommendations for my finances",
                    "suggest ways to improve",
                ],
                "information": [
                    "what is my balance",
                    "how much did I spend",
                    "tell me about my accounts",
                    "show me transactions",
                    "explain my finances",
                    "when did I last spend",
                ],
            }

    def _update_intents_data(self, new_intents: Dict[str, List[str]]):
        """Update intents data with new examples"""
        updated_intents = self.intents_data.copy()

        # Add new examples to existing intents
        for intent, examples in new_intents.items():
            if intent in updated_intents:
                # Add unique examples
                current_examples = set(updated_intents[intent])
                for example in examples:
                    if example not in current_examples:
                        updated_intents[intent].append(example)
            else:
                updated_intents[intent] = examples

        # Save updated intents data
        try:
            os.makedirs("models", exist_ok=True)
            with open("models/intents_data.json", "w") as f:
                json.dump(updated_intents, f, indent=2)

            # Update in-memory data
            self.intents_data = updated_intents
            logger.info(f"Updated intents data with new examples")
        except Exception as e:
            logger.error(f"Failed to save updated intents data: {str(e)}")

    def process_learning_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Process a dataset of conversations to improve Fin."""
        helpful_conversations = dataset.get("helpful_conversations", [])
        unhelpful_conversations = dataset.get("unhelpful_conversations", [])
        tool_usage = dataset.get("tool_usage", {})

        if not helpful_conversations and not unhelpful_conversations:
            return {
                "status": "error",
                "message": "No conversations provided in dataset",
            }

        logger.info(
            f"Processing learning dataset with {len(helpful_conversations)} helpful and {len(unhelpful_conversations)} unhelpful conversations"
        )

        # Extract patterns from helpful conversations
        success_patterns = self._extract_patterns_from_conversations(
            helpful_conversations, is_helpful=True
        )

        # Extract patterns from unhelpful conversations to learn what doesn't work
        failure_patterns = self._extract_patterns_from_conversations(
            unhelpful_conversations, is_helpful=False
        )

        # Gather new intents data from conversations
        new_intents = self._extract_intents_from_conversations(
            helpful_conversations, unhelpful_conversations
        )

        # Update our intents data with new examples
        if new_intents:
            self._update_intents_data(new_intents)

        # Combine patterns and prioritize success patterns
        all_patterns = success_patterns + failure_patterns

        # Update the system prompt with new patterns
        self._update_system_prompt(all_patterns)

        # Extract example exchanges for few-shot learning
        new_examples = self._extract_example_exchanges(helpful_conversations)
        self._update_example_exchanges(new_examples)

        # Analyze tool usage patterns
        tool_patterns = self._analyze_tool_usage(tool_usage)

        # Update tool usage guidance in the prompt
        self._update_tool_guidance(tool_patterns)

        return {
            "status": "success",
            "processed_conversations": len(helpful_conversations)
            + len(unhelpful_conversations),
            "extracted_patterns": len(all_patterns),
            "new_examples": len(new_examples),
            "tool_patterns": len(tool_patterns),
            "new_intent_examples": sum(
                len(examples) for examples in new_intents.values()
            ),
        }

    def _extract_intents_from_conversations(
        self, helpful_conversations, unhelpful_conversations
    ) -> Dict[str, List[str]]:
        """Extract new intent examples from conversations"""
        new_intents = {}

        # We primarily want to learn from helpful conversations
        for conversation in helpful_conversations:
            messages = conversation.get("messages", [])

            # Focus on user messages that received high ratings
            for i, message in enumerate(messages):
                if message["role"] != "user":
                    continue

                # Check if this was followed by a helpful assistant message
                if (
                    i + 1 < len(messages)
                    and messages[i + 1]["role"] == "assistant"
                    and messages[i + 1].get("was_helpful", False)
                    and messages[i + 1].get("feedback_rating", 0) >= 4
                ):

                    # Extract intent
                    user_query = message["content"]
                    intent = self._extract_intent(user_query)

                    if intent != "other":
                        if intent not in new_intents:
                            new_intents[intent] = []
                        new_intents[intent].append(user_query)

        return new_intents

    def _extract_patterns_from_conversations(self, conversations, is_helpful=True):
        """Extract patterns from a list of conversations using NLP techniques"""
        patterns = []

        for conversation in conversations:
            messages = conversation.get("messages", [])

            # Group messages into exchanges (user question + assistant response)
            exchanges = []
            for i in range(0, len(messages) - 1, 2):
                if (
                    i + 1 < len(messages)
                    and messages[i]["role"] == "user"
                    and messages[i + 1]["role"] == "assistant"
                ):
                    exchanges.append(
                        {
                            "user": messages[i]["content"],
                            "assistant": messages[i + 1]["content"],
                            "tools": messages[i + 1].get("tools_used", []),
                        }
                    )

            # Apply NLP techniques to extract patterns
            for exchange in exchanges:
                user_query = exchange["user"]
                assistant_response = exchange["assistant"]

                # Use NLP to extract intent, entities, and patterns
                intent = self._extract_intent(user_query)
                entities = self._extract_entities(user_query)
                query_type = self._classify_query_type(user_query)

                pattern = {
                    "type": "query_pattern",
                    "query_type": query_type,
                    "intent": intent,
                    "entities": entities,
                    "example": user_query,
                    "is_helpful": is_helpful,
                }

                patterns.append(pattern)

                # If tools were used, analyze their effectiveness
                if exchange.get("tools"):
                    tool_patterns = self._analyze_tool_effectiveness(
                        user_query, assistant_response, exchange["tools"], is_helpful
                    )
                    patterns.extend(tool_patterns)

        return patterns

    def _extract_intent(self, text: str) -> str:
        """
        Extract the user's intent from text using NLP

        Uses TF-IDF vectorization and cosine similarity to identify the most likely
        intent based on example queries for each intent category.
        """
        if not text or not self.intents_data:
            return "other"

        # Fallback to keyword-based approach if vectorizer initialization fails
        if not hasattr(self, "tfidf_vectorizer"):
            return self._keyword_based_intent(text)

        try:
            # Prepare training data from intents
            train_texts = []
            train_labels = []

            for intent, examples in self.intents_data.items():
                for example in examples:
                    train_texts.append(example)
                    train_labels.append(intent)

            if not train_texts:
                return "other"

            # Fit vectorizer and transform texts
            X_train = self.tfidf_vectorizer.fit_transform(train_texts)
            X_query = self.tfidf_vectorizer.transform([text])

            # Calculate similarity scores
            similarities = cosine_similarity(X_query, X_train)[0]

            # Get the most similar example
            max_sim_idx = np.argmax(similarities)
            best_match_score = similarities[max_sim_idx]

            # If the similarity is too low, return "other"
            if best_match_score < 0.3:
                # Try spaCy similarity if TF-IDF similarity is low
                if self.nlp:
                    intent_scores = self._spacy_intent_similarity(text)
                    if intent_scores:
                        best_intent, score = max(
                            intent_scores.items(), key=lambda x: x[1]
                        )
                        if score > 0.6:  # Higher threshold for spaCy similarity
                            return best_intent
                return "other"

            return train_labels[max_sim_idx]

        except Exception as e:
            logger.error(f"Error in TF-IDF intent extraction: {str(e)}")
            # Fallback to keyword-based approach
            return self._keyword_based_intent(text)

    def _spacy_intent_similarity(self, text: str) -> Dict[str, float]:
        """Use spaCy word vectors to calculate intent similarity"""
        if not self.nlp:
            return {}

        try:
            text_doc = self.nlp(text)

            # Calculate similarity with representative examples for each intent
            intent_scores = {}

            for intent, examples in self.intents_data.items():
                # Use the first few examples as representatives
                sample_examples = examples[:3]

                # Calculate average similarity
                intent_score = 0
                for example in sample_examples:
                    example_doc = self.nlp(example)
                    intent_score += text_doc.similarity(example_doc)

                if sample_examples:
                    intent_score /= len(sample_examples)

                intent_scores[intent] = intent_score

            return intent_scores

        except Exception as e:
            logger.error(f"Error in spaCy intent similarity: {str(e)}")
            return {}

    @staticmethod
    def _keyword_based_intent(text: str) -> str:
        """Fallback method using keywords to determine intent"""
        text_lower = text.lower()

        intents = {
            "forecast": ["forecast", "predict", "future", "projection", "outlook"],
            "budget": ["budget", "spending", "limit", "allocate"],
            "analysis": ["analyze", "analysis", "trend", "pattern"],
            "comparison": ["compare", "versus", "vs", "difference"],
            "recommendation": ["recommend", "suggest", "advise", "should"],
            "information": ["what is", "tell me", "explain", "how much", "show me"],
        }

        for intent, keywords in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent

        return "other"

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract financial entities from text using NLP

        Uses a combination of NER from HuggingFace Transformers and
        custom entity extraction for financial terms.
        """
        entities = []

        # Try using the HuggingFace NER pipeline
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)

                # Process standard NER results
                money_values = []
                date_values = []
                org_values = []

                for entity in ner_results:
                    if entity["entity"].startswith("B-MONEY") or entity[
                        "entity"
                    ].startswith("I-MONEY"):
                        money_values.append(entity["word"])
                    elif entity["entity"].startswith("B-DATE") or entity[
                        "entity"
                    ].startswith("I-DATE"):
                        date_values.append(entity["word"])
                    elif entity["entity"].startswith("B-ORG") or entity[
                        "entity"
                    ].startswith("I-ORG"):
                        org_values.append(entity["word"])

                # Group consecutive tokens for the same entity
                if money_values:
                    entities.append(
                        {
                            "type": "amount",
                            "values": self._consolidate_tokens(money_values),
                        }
                    )
                if date_values:
                    entities.append(
                        {
                            "type": "time_period",
                            "values": self._consolidate_tokens(date_values),
                        }
                    )
                if org_values:
                    entities.append(
                        {
                            "type": "organization",
                            "values": self._consolidate_tokens(org_values),
                        }
                    )

            except Exception as e:
                logger.error(f"Error using HuggingFace NER: {str(e)}")

        # Try using spaCy as a backup or additional NER
        if self.nlp:
            try:
                doc = self.nlp(text)

                # Extract entities from spaCy
                money_values = []
                date_values = []
                org_values = []

                for ent in doc.ents:
                    if ent.label_ == "MONEY":
                        money_values.append(ent.text)
                    elif ent.label_ in ["DATE", "TIME"]:
                        date_values.append(ent.text)
                    elif ent.label_ == "ORG":
                        org_values.append(ent.text)

                # Only add if not already added by HuggingFace
                if money_values and not any(e["type"] == "amount" for e in entities):
                    entities.append({"type": "amount", "values": money_values})
                if date_values and not any(
                    e["type"] == "time_period" for e in entities
                ):
                    entities.append({"type": "time_period", "values": date_values})
                if org_values and not any(
                    e["type"] == "organization" for e in entities
                ):
                    entities.append({"type": "organization", "values": org_values})

            except Exception as e:
                logger.error(f"Error using spaCy NER: {str(e)}")

        # Extract monetary amounts using regex as a fallback
        if not any(e["type"] == "amount" for e in entities):
            amount_matches = re.findall(r"\$?(\d+(?:,\d+)*(?:\.\d+)?)", text)
            if amount_matches:
                entities.append({"type": "amount", "values": amount_matches})

        # Extract time periods using regex as a fallback
        if not any(e["type"] == "time_period" for e in entities):
            self._extract_time_periods_regex(text, entities)

        # Extract financial categories - these are domain-specific and best done with custom logic
        self._extract_financial_categories(text, entities)

        return entities

    @staticmethod
    def _consolidate_tokens(
        tokens: List[str], entity_spans: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Consolidate named entity tokens into meaningful entities.

        Args:
            tokens: List of token strings to consolidate
            entity_spans: Optional list of entity spans with indices for precise consolidation
                          Each span should have 'start' and 'end' keys

        Returns:
            List of consolidated entity strings
        """
        if not tokens:
            return []

        # If we have entity spans information (from models like spaCy or HuggingFace),
        # use it for precise consolidation
        if entity_spans:
            # Sort spans by start position
            sorted_spans = sorted(entity_spans, key=lambda x: x.get("start", 0))
            consolidated = []

            for span in sorted_spans:
                # Extract the complete entity using span boundaries
                start, end = span.get("start", 0), span.get("end", 0)
                if 0 <= start < end <= len(tokens):
                    entity = " ".join(tokens[start:end])
                    consolidated.append(entity)

            return consolidated if consolidated else [" ".join(tokens)]

        # When no span information is available, use a heuristic approach
        # to group potential entity tokens
        result = []
        current_entity = []

        for i, token in enumerate(tokens):
            # Skip punctuation tokens
            if token in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}"]:
                if current_entity:
                    result.append(" ".join(current_entity))
                    current_entity = []
                continue

            # Check if token might be part of an entity
            if (token.lower() in ["the", "a", "an"] and i == 0) or (
                token[0].isupper() or token.isdigit() or token in ["$", "€", "£", "¥"]
            ):
                current_entity.append(token)
            else:
                # If we have collected entity tokens and find a non-entity token,
                # finalize the current entity
                if current_entity:
                    result.append(" ".join(current_entity))
                    current_entity = []

                # Add standalone tokens that aren't part of entities but might be relevant
                if token not in ["and", "or", "of", "the", "a", "an"]:
                    result.append(token)

        # Add the last entity if there is one
        if current_entity:
            result.append(" ".join(current_entity))

        # Handle special cases for money/currency
        consolidated_result = []
        for i, entity in enumerate(result):
            if (
                entity in ["$", "€", "£", "¥"]
                and i + 1 < len(result)
                and any(c.isdigit() for c in result[i + 1])
            ):
                # Combine currency symbol with the following number
                consolidated_result.append(f"{entity}{result[i + 1]}")
                # Skip the number in the next iteration
                result[i + 1] = None
            elif entity is not None:
                consolidated_result.append(entity)

        # Remove empty or None entities
        return [e for e in consolidated_result if e]

    @staticmethod
    def _extract_time_periods_regex(text: str, entities: List[Dict[str, Any]]):
        """Extract time periods using regex patterns"""
        time_periods = ["day", "week", "month", "year", "quarter"]
        time_matches = []

        for period in time_periods:
            # Match patterns like "last 3 months", "next month", etc.
            matches = re.findall(
                r"(last|next|this|coming|previous|past)\s+(\d+)?\s*" + period + "s?",
                text,
                re.IGNORECASE,
            )
            if matches:
                time_matches.extend(
                    [
                        f"{rel} {num if num else '1'} {period}{'s' if num and num != '1' else ''}"
                        for rel, num in matches
                    ]
                )

        if time_matches:
            entities.append({"type": "time_period", "values": time_matches})

    def _extract_financial_categories(self, text: str, entities: List[Dict[str, Any]]):
        """Extract financial categories from text"""
        # Use a more comprehensive list of financial categories
        category_patterns = [
            "groceries",
            "dining",
            "restaurants",
            "utilities",
            "rent",
            "mortgage",
            "transportation",
            "entertainment",
            "shopping",
            "travel",
            "healthcare",
            "insurance",
            "education",
            "savings",
            "income",
            "salary",
            "investments",
            "stocks",
            "bonds",
            "real estate",
            "retirement",
            "loan",
            "debt",
            "credit card",
            "subscription",
            "marketing",
            "advertising",
            "payroll",
            "taxes",
            "supplies",
            "equipment",
            "maintenance",
            "repairs",
            "legal",
            "accounting",
        ]

        # Try extracting specific categories
        category_matches = []

        # Use spaCy for noun phrase extraction if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

                # Check if any noun phrase contains a category term
                for phrase in noun_phrases:
                    for category in category_patterns:
                        if category in phrase:
                            category_matches.append(phrase)
                            break
            except Exception as e:
                logger.error(f"Error extracting noun phrases: {str(e)}")

        # Fallback to simple matching
        if not category_matches:
            for category in category_patterns:
                if re.search(r"\b" + category + r"\b", text, re.IGNORECASE):
                    category_matches.append(category)

        if category_matches:
            entities.append({"type": "category", "values": category_matches})

    def _classify_query_type(self, text: str) -> str:
        """
        Classify the type of financial query

        Uses a combination of spaCy similarity and regex patterns to determine
        the most likely query type.
        """
        # Try to use spaCy doc similarity if available
        if self.nlp:
            try:
                query_doc = self.nlp(text)

                # Define prototypical queries for each query type
                query_prototypes = {
                    "historical_spending": [
                        "How much did I spend last month?",
                        "What were my expenses on groceries last week?",
                        "How much did I pay for utilities in the past year?",
                    ],
                    "budget_creation": [
                        "Create a budget for me",
                        "Help me make a budget for groceries",
                        "Suggest a budget based on my spending",
                    ],
                    "financial_forecast": [
                        "Forecast my cash flow for next month",
                        "What will my finances look like next year?",
                        "Predict my expenses for the coming quarter",
                    ],
                    "what_if_scenario": [
                        "What if I spend $500 more on marketing?",
                        "What would happen if I reduce my grocery budget?",
                        "How would buying a car affect my finances?",
                    ],
                    "category_analysis": [
                        "Where is most of my money going?",
                        "What categories do I spend the most on?",
                        "Analyze my spending by category",
                    ],
                    "saving_goal": [
                        "Help me save for a vacation",
                        "How much should I save each month for a down payment?",
                        "I want to save $10,000 by next year",
                    ],
                }

                # Calculate similarity scores
                best_type = None
                best_score = 0

                for query_type, examples in query_prototypes.items():
                    avg_score = 0
                    for example in examples:
                        example_doc = self.nlp(example)
                        similarity = query_doc.similarity(example_doc)
                        avg_score += similarity

                    avg_score /= len(examples)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_type = query_type

                # Only use spaCy result if score is high enough
                if best_score > 0.7:
                    return best_type

            except Exception as e:
                logger.error(f"Error in spaCy query type classification: {str(e)}")

        # Fallback to regex-based classification
        text_lower = text.lower()

        query_types = {
            "historical_spending": [
                r"(spend|spent|spending|cost|costs|paid|pay).*?(last|previous|past).*?(month|week|year)",
                r"how much.*?(spend|spent|spending).*?(on|for)",
            ],
            "budget_creation": [
                r"(create|make|suggest|recommend).*?budget",
                r"budget.*?(for|on)",
            ],
            "financial_forecast": [
                r"(forecast|project|predict|outlook|future)",
                r"what.*?(happen|look like).*?(if|when)",
            ],
            "what_if_scenario": [
                r"what if",
                r"what would happen if",
                r"scenario",
            ],
            "category_analysis": [
                r"(where|what).*?(money|spending)",
                r"(most|least).*?(spend|spent|spending)",
            ],
            "saving_goal": [
                r"(save|saving|savings).*?(for|goal)",
                r"how (long|much).*?(save|saving|savings)",
            ],
        }

        for query_type, patterns in query_types.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                return query_type

        return "general_inquiry"

    @staticmethod
    def _analyze_tool_usage(tool_usage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze tool usage statistics to identify patterns"""
        patterns = []

        for tool_name, stats in tool_usage.items():
            total = stats.get("total", 0)
            success = stats.get("success", 0)

            if total > 0:
                success_rate = success / total

                # Record pattern for tools with good success rates
                if (
                    success_rate > 0.7 and total >= 5
                ):  # Minimum threshold for significance
                    patterns.append(
                        {
                            "type": "tool_success_rate",
                            "tool": tool_name,
                            "success_rate": success_rate,
                            "sample_size": total,
                        }
                    )

                # Record pattern for tools with poor success rates
                elif success_rate < 0.3 and total >= 5:
                    patterns.append(
                        {
                            "type": "tool_failure_rate",
                            "tool": tool_name,
                            "failure_rate": 1 - success_rate,
                            "sample_size": total,
                        }
                    )

        return patterns

    def _analyze_tool_effectiveness(
        self,
        user_query: str,
        assistant_response: str,
        tools: List[Dict[str, Any]],
        is_helpful: bool,
    ) -> List[Dict[str, Any]]:
        """Analyze when tools are effective or ineffective"""
        patterns = []

        for tool in tools:
            tool_name = tool.get("name", "")
            success = tool.get("success", False)

            # Skip if no tool name
            if not tool_name:
                continue

            # Record successful tool usage in helpful responses
            if success and is_helpful:
                # Extract intent and query type
                intent = self._extract_intent(user_query)
                query_type = self._classify_query_type(user_query)

                patterns.append(
                    {
                        "type": "tool_success",
                        "tool": tool_name,
                        "query_context": user_query,
                        "intent": intent,
                        "query_type": query_type,
                    }
                )

            # Record unsuccessful tool usage
            elif not success and not is_helpful:
                patterns.append(
                    {
                        "type": "tool_failure",
                        "tool": tool_name,
                        "query_context": user_query,
                    }
                )

        return patterns

    def _update_tool_guidance(self, tool_patterns: List[Dict[str, Any]]) -> None:
        """Update the system prompt with tool usage guidance"""
        if not tool_patterns:
            return

        # Separate success and failure patterns
        success_patterns = [
            p for p in tool_patterns if p["type"] == "tool_success_rate"
        ]
        failure_patterns = [
            p for p in tool_patterns if p["type"] == "tool_failure_rate"
        ]

        guidance_sections = []

        # Add guidance for successful tools
        if success_patterns:
            guidance_sections.append("\n\n# TOOL USAGE RECOMMENDATIONS")
            guidance_sections.append(
                "Based on past performance, these tools work well for specific types of queries:"
            )

            for pattern in success_patterns:
                tool = pattern.get("tool", "")
                rate = pattern.get("success_rate", 0) * 100
                guidance_sections.append(
                    f"- {tool}: {rate:.1f}% success rate ({pattern.get('sample_size', 0)} uses)"
                )

        # Add guidance for tools to use cautiously
        if failure_patterns:
            if not guidance_sections:
                guidance_sections.append("\n\n# TOOL USAGE RECOMMENDATIONS")
            guidance_sections.append("\nThese tools should be used more cautiously:")

            for pattern in failure_patterns:
                tool = pattern.get("tool", "")
                rate = pattern.get("failure_rate", 0) * 100
                guidance_sections.append(
                    f"- {tool}: {rate:.1f}% failure rate ({pattern.get('sample_size', 0)} uses)"
                )

        # Update the system prompt if we have guidance
        if guidance_sections:
            updated_prompt = self.system_prompt_template
            guidance_text = "\n".join(guidance_sections)

            # Check if we already have a TOOL USAGE RECOMMENDATIONS section
            if "# TOOL USAGE RECOMMENDATIONS" in updated_prompt:
                # Replace existing section
                updated_prompt = re.sub(
                    r"(# TOOL USAGE RECOMMENDATIONS\n)(.+?)(\n# |\Z)",
                    r"\1" + guidance_text + r"\3",
                    updated_prompt,
                    flags=re.DOTALL,
                )
            else:
                # Add new section at the end
                updated_prompt += guidance_text

            # Save updated prompt
            try:
                os.makedirs("prompts", exist_ok=True)
                with open("prompts/fin_system_prompt.txt", "w") as f:
                    f.write(updated_prompt)

                self.system_prompt_template = updated_prompt
                logger.info("Updated system prompt with tool usage guidance")
            except Exception as e:
                logger.error(f"Failed to save updated tool guidance: {str(e)}")

    @staticmethod
    def _load_system_prompt_template() -> str:
        """Load the system prompt template from file or use default"""
        try:
            with open("prompts/fin_system_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Default system prompt if file doesn't exist
            return """
            You are Fin, an AI-powered financial assistant for the Cashly app.

            Your role is to help users understand their finances, answer questions about their spending,
            income, budgets, and provide forecasts and financial advice.

            When answering:
            1. Be concise, friendly, and helpful
            2. For complex questions, use the appropriate tool to calculate the answer
            3. If the user asks a question that requires creating a forecast or scenario, use the forecast_cash_flow tool
            4. For category-based questions, analyze the transactions data
            5. For budget-related questions, use the budget tools
            6. When asked about trends or patterns, use the analyze_trends tool
            7. When the user asks about unusual spending or transactions, use the detect_anomalies tool

            IMPORTANT: Do not make up information. If you don't have enough data or the right tool to answer a question,
            tell the user you can't answer that question with the available data.

            Use your tools to generate accurate responses based on the user's financial data.
            """

    @staticmethod
    def _load_example_exchanges() -> List[Dict[str, Any]]:
        """Load example exchanges from file or use defaults"""
        try:
            with open("prompts/fin_examples.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default examples if file doesn't exist or is invalid
            return []

    def _update_system_prompt(self, patterns: List[Dict[str, Any]]) -> None:
        """Update the system prompt based on extracted patterns"""
        # Extract query patterns
        query_patterns = [p for p in patterns if p["type"] == "query_pattern"]

        # Group by query type
        query_types = {}
        for pattern in query_patterns:
            query_type = pattern.get("query_type", "")
            if query_type:
                if query_type not in query_types:
                    query_types[query_type] = []
                query_types[query_type].append(pattern.get("example", ""))

        # Create new sections for the system prompt
        new_sections = []

        if query_types:
            new_sections.append(
                "\n\nThese are common financial questions you should be prepared to answer:"
            )

            for query_type, examples in query_types.items():
                example = examples[0] if examples else ""
                description = self._get_query_type_description(query_type)

                new_sections.append(f"- {description} (e.g., '{example}')")

        # Add tool success patterns
        tool_patterns = [p for p in patterns if p["type"] == "tool_success"]
        if tool_patterns:
            tool_mapping = {}
            for pattern in tool_patterns:
                tool = pattern.get("tool", "")
                if tool:
                    if tool not in tool_mapping:
                        tool_mapping[tool] = []
                    tool_mapping[tool].append(pattern.get("query_context", ""))

            new_sections.append(
                "\n\nFor these types of questions, use these specific tools:"
            )

            for tool, contexts in tool_mapping.items():
                context = contexts[0] if contexts else ""
                new_sections.append(
                    f"- Use '{tool}' when questions are like: '{context}'"
                )

        # Create updated system prompt
        updated_prompt = self.system_prompt_template

        # Add learning-derived sections
        if new_sections:
            updated_section = "\n".join(new_sections)

            # Check if we already have a LEARNING INSIGHTS section
            if "LEARNING INSIGHTS" in updated_prompt:
                # Replace existing section
                updated_prompt = re.sub(
                    r"(# LEARNING INSIGHTS\n)(.+?)(\n# |\Z)",
                    r"\1" + updated_section + r"\3",
                    updated_prompt,
                    flags=re.DOTALL,
                )
            else:
                # Add new section at the end
                updated_prompt += "\n\n# LEARNING INSIGHTS\n" + updated_section

        # Save updated prompt
        try:
            os.makedirs("prompts", exist_ok=True)
            with open("prompts/fin_system_prompt.txt", "w") as f:
                f.write(updated_prompt)

            self.system_prompt_template = updated_prompt
            logger.info("Updated system prompt template with new patterns")
        except Exception as e:
            logger.error(f"Failed to save updated system prompt: {str(e)}")

    def _extract_patterns(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract useful patterns from the dataset"""
        patterns = []

        for conversation in dataset:
            messages = conversation.get("messages", [])
            tools_used = conversation.get("tools_used", [])
            led_to_action = conversation.get("led_to_action", False)

            # Skip conversations that are too short
            if len(messages) < 2:
                continue

            # Group messages into exchanges (user question + assistant response)
            exchanges = []
            for i in range(0, len(messages) - 1, 2):
                if (
                    i + 1 < len(messages)
                    and messages[i]["role"] == "user"
                    and messages[i + 1]["role"] == "assistant"
                ):
                    exchanges.append(
                        {
                            "user": messages[i]["content"],
                            "assistant": messages[i + 1]["content"],
                            "tools": [
                                t
                                for t in tools_used
                                if t.get("timestamp", "")
                                > messages[i].get("timestamp", "")
                            ],
                        }
                    )

            # Analyze each exchange for patterns
            for exchange in exchanges:
                user_query = exchange["user"]
                assistant_response = exchange["assistant"]
                tools = exchange["tools"]

                # Look for financial query patterns
                financial_patterns = self._identify_financial_patterns(
                    user_query, assistant_response, tools
                )

                # If this exchange led to user action, it's especially valuable
                if led_to_action:
                    financial_patterns.append(
                        {
                            "type": "action_trigger",
                            "query": user_query,
                            "response_elements": self._extract_response_elements(
                                assistant_response
                            ),
                        }
                    )

                patterns.extend(financial_patterns)

        # Deduplicate patterns
        unique_patterns = []
        seen_patterns = set()

        for pattern in patterns:
            pattern_key = f"{pattern['type']}:{pattern.get('query_type', '')}"
            if pattern_key not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(pattern_key)

        return unique_patterns

    @staticmethod
    def _identify_financial_patterns(query, response, tools):
        """Identify patterns in financial queries and responses"""
        patterns = []

        # Check for specific financial query types
        query_lower = query.lower()

        # Spending questions
        if re.search(
            r"(spend|spent|spending|cost|costs|paid|pay)", query_lower
        ) and re.search(r"(last|this) (month|week|year)", query_lower):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "historical_spending",
                    "example": query,
                }
            )

        # Budget questions
        if re.search(r"budget", query_lower) and re.search(
            r"(create|make|suggest|recommend)", query_lower
        ):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "budget_creation",
                    "example": query,
                }
            )

        # Forecast questions
        if re.search(r"(forecast|project|predict|outlook|future)", query_lower):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "financial_forecast",
                    "example": query,
                }
            )

        # What-if scenario questions
        if re.search(r"what (if|would happen|would it look like)", query_lower):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "what_if_scenario",
                    "example": query,
                }
            )

        # Tool effectiveness patterns
        for tool in tools:
            tool_name = tool.get("name", "")
            success = tool.get("success", False)

            if success:
                patterns.append(
                    {"type": "tool_success", "tool": tool_name, "query_context": query}
                )

        return patterns

    @staticmethod
    def _extract_response_elements(response):
        """Extract key elements from a successful response"""
        elements = []

        # Check for data points
        if re.search(r"\$[\d,.]+", response):
            elements.append("specific_amounts")

        # Check for time references
        if re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December|last month|this month|next month)",
            response,
        ):
            elements.append("time_references")

        # Check for comparisons
        if re.search(
            r"(more than|less than|compared to|versus|vs|increased|decreased)", response
        ):
            elements.append("comparisons")

        # Check for actionable advice
        if re.search(r"(recommend|suggest|advise|could|should|consider)", response):
            elements.append("actionable_advice")

        # Check for visualizations references
        if re.search(r"(chart|graph|visualization|diagram)", response):
            elements.append("visualization_references")

        return elements

    @staticmethod
    def _extract_example_exchanges(
        dataset: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract high-quality example exchanges from the dataset"""
        examples = []

        # Look for conversations with high ratings and that led to action
        for conversation in dataset:
            messages = conversation.get("messages", [])
            led_to_action = conversation.get("led_to_action", False)

            # Skip conversations that are too short
            if len(messages) < 2:
                continue

            # Only consider high-value conversations
            if not led_to_action:
                continue

            # Extract exchanges
            for i in range(0, len(messages) - 1, 2):
                if (
                    i + 1 < len(messages)
                    and messages[i]["role"] == "user"
                    and messages[i + 1]["role"] == "assistant"
                ):
                    examples.append(
                        {
                            "user": messages[i]["content"],
                            "assistant": messages[i + 1]["content"],
                        }
                    )

        # Take the top 10 most diverse examples
        if examples:
            # Very basic diversity check - just take different length questions
            examples.sort(key=lambda x: len(x["user"]))
            step = max(1, len(examples) // 10)
            diverse_examples = examples[::step][:10]
            return diverse_examples

        return []

    def _update_example_exchanges(self, new_examples: List[Dict[str, Any]]) -> None:
        """Update the example exchanges file with new examples"""
        if not new_examples:
            return

        # Combine with existing examples
        combined_examples = self.example_exchanges

        # Add new unique examples
        existing_user_queries = [ex["user"] for ex in combined_examples]
        for example in new_examples:
            if example["user"] not in existing_user_queries:
                combined_examples.append(example)
                existing_user_queries.append(example["user"])

        # Limit to a reasonable number
        if len(combined_examples) > 20:
            combined_examples = combined_examples[-20:]

        # Save to file
        try:
            os.makedirs("prompts", exist_ok=True)
            with open("prompts/fin_examples.json", "w") as f:
                json.dump(combined_examples, f, indent=2)

            self.example_exchanges = combined_examples
            logger.info(
                f"Updated example exchanges with {len(new_examples)} new examples"
            )
        except Exception as e:
            logger.error(f"Failed to save updated examples: {str(e)}")

    @staticmethod
    def _get_query_type_description(query_type: str) -> str:
        """Get a human-readable description for a query type"""
        descriptions = {
            "historical_spending": "Questions about past spending in specific categories or time periods",
            "budget_creation": "Requests to create or recommend budget allocations",
            "financial_forecast": "Questions about future financial projections",
            "what_if_scenario": "Hypothetical scenarios about financial changes",
            "savings_goals": "Questions about saving for specific goals",
            "investment_advice": "Questions about investment strategies",
            "expense_reduction": "Questions about reducing specific expenses",
        }

        return descriptions.get(query_type, query_type.replace("_", " ").title())
