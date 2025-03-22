import unittest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock, mock_open
from services.fin_learning_service import FinLearningService


class TestFinLearningService(unittest.TestCase):
    """Tests for the FinLearningService class"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create models and prompts directories
        os.makedirs(os.path.join(self.test_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "prompts"), exist_ok=True)
        
        # Set up environment variable for tests
        os.environ["ANTHROPIC_API_KEY"] = "test_api_key"
        
        # Patch the paths for config files
        self.model_path_patcher = patch(
            "services.fin_learning_service.os.path.join", 
            return_value=os.path.join(self.test_dir, "models/intents_data.json")
        )
        self.model_path_mock = self.model_path_patcher.start()
        
        # Disable logging during tests
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """Clean up after each test"""
        # Stop patchers
        self.model_path_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
        # Clear environment variables
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        
        # Re-enable logging
        logging.disable(logging.NOTSET)

    @patch("services.fin_learning_service.anthropic.Anthropic")
    @patch("services.fin_learning_service.spacy.load")
    @patch("services.fin_learning_service.AutoTokenizer.from_pretrained")
    @patch("services.fin_learning_service.AutoModelForTokenClassification.from_pretrained")
    @patch("services.fin_learning_service.pipeline")
    def test_initialization(self, mock_pipeline, mock_model, mock_tokenizer, mock_spacy_load, mock_anthropic):
        """Test FinLearningService initialization"""
        # Set up mocks
        mock_spacy_model = MagicMock()
        mock_spacy_load.return_value = mock_spacy_model
        
        # Create service instance
        service = FinLearningService()
        
        # Verify Anthropic client was initialized
        mock_anthropic.assert_called_once_with(api_key="test_api_key")
        
        # Verify NLP models were loaded
        mock_spacy_load.assert_called_once_with("en_core_web_md")
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        mock_pipeline.assert_called_once()
        
        # Verify intents data was loaded
        self.assertIsInstance(service.intents_data, dict)
        self.assertIn("forecast", service.intents_data)
        self.assertIn("budget", service.intents_data)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_load_intents_data_fallback(self, mock_load_nlp):
        """Test loading intents data falls back to default when file not found"""
        # Create service instance
        service = FinLearningService()
        
        # Verify default intents were loaded
        self.assertIn("forecast", service.intents_data)
        self.assertIn("budget", service.intents_data)
        self.assertIn("analysis", service.intents_data)
        
        # Verify each intent has examples
        for intent, examples in service.intents_data.items():
            self.assertIsInstance(examples, list)
            self.assertTrue(len(examples) > 0)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_update_intents_data(self, mock_load_nlp):
        """Test updating intents data with new examples"""
        # Create service instance
        service = FinLearningService()
        
        # Initial count of examples
        initial_count = sum(len(examples) for examples in service.intents_data.values())
        
        # New intents to add
        new_intents = {
            "forecast": ["predict my spending next quarter", "forecast expenses for next year"],
            "new_category": ["example of new intent category"]
        }
        
        # Update intents
        service._update_intents_data(new_intents)
        
        # Verify examples were added
        self.assertIn("forecast", service.intents_data)
        self.assertIn("new_category", service.intents_data)
        self.assertIn("predict my spending next quarter", service.intents_data["forecast"])
        
        # Verify count increased
        updated_count = sum(len(examples) for examples in service.intents_data.values())
        self.assertGreater(updated_count, initial_count)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_update_intents_data_saves_file(self, mock_json_dump, mock_file, mock_load_nlp):
        """Test that updating intents saves to file"""
        # Create service instance
        service = FinLearningService()
        
        # New intents to add
        new_intents = {
            "forecast": ["new forecast example"]
        }
        
        # Update intents
        service._update_intents_data(new_intents)
        
        # Verify file was opened for writing
        mock_file.assert_called_with("models/intents_data.json", "w")
        
        # Verify json was dumped
        mock_json_dump.assert_called_once()
        args, _ = mock_json_dump.call_args
        self.assertEqual(args[0], service.intents_data)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_extract_intent(self, mock_load_nlp):
        """Test intent extraction from text"""
        # Create service with mocked NLP
        service = FinLearningService()
        service.nlp = None
        
        # Test keyword-based fallback method
        intents = [
            ("forecast my spending for next month", "forecast"),
            ("create a budget for groceries", "budget"),
            ("analyze my spending patterns", "analysis"),
            ("compare this month to last month", "comparison"),
            ("recommend ways to save more", "recommendation"),
            ("what is my current balance", "information"),
            ("something completely different", "other")
        ]
        
        for text, expected_intent in intents:
            result = service._extract_intent(text)
            self.assertEqual(result, expected_intent, f"Failed to extract intent '{expected_intent}' from '{text}'")

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_extract_entities(self, mock_load_nlp):
        """Test financial entity extraction"""
        # Create service
        service = FinLearningService()
        
        # Mock the NLP and NER components to return None
        service.nlp = None
        service.ner_pipeline = None
        
        # Test regex-based fallback for amounts
        text = "I spent $50.25 on groceries last month"
        entities = service._extract_entities(text)
        
        # Verify amount extraction
        amount_entity = next((e for e in entities if e["type"] == "amount"), None)
        self.assertIsNotNone(amount_entity)
        self.assertIn("50.25", amount_entity["values"][0])
        
        # Test time period extraction
        text = "Show me spending for the last 3 months"
        entities = service._extract_entities(text)
        
        # Verify time period extraction
        time_entity = next((e for e in entities if e["type"] == "time_period"), None)
        self.assertIsNotNone(time_entity)
        self.assertTrue(any("last 3 months" in value for value in time_entity["values"]))
        
        # Test category extraction
        text = "Show me my groceries spending"
        entities = service._extract_entities(text)
        
        # Verify category extraction
        category_entity = next((e for e in entities if e["type"] == "category"), None)
        self.assertIsNotNone(category_entity)
        self.assertIn("groceries", category_entity["values"])

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_classify_query_type(self, mock_load_nlp):
        """Test classification of financial query types"""
        # Create service
        service = FinLearningService()
        service.nlp = None  # Force fallback to regex method
        
        # Test query types
        queries = [
            ("How much did I spend last month?", "historical_spending"),
            ("Create a budget for me", "budget_creation"),
            ("Forecast my expenses for next month", "financial_forecast"),
            ("What if I spend $500 more on marketing?", "what_if_scenario"),
            ("Where is my money going?", "category_analysis"),
            ("Help me save for a vacation", "saving_goal"),
            ("Tell me about my account", "general_inquiry")  # Default fallback
        ]
        
        for text, expected_type in queries:
            result = service._classify_query_type(text)
            self.assertEqual(result, expected_type, f"Failed to classify '{text}' as '{expected_type}'")

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    @patch("services.fin_learning_service.FinLearningService._extract_patterns_from_conversations")
    @patch("services.fin_learning_service.FinLearningService._extract_intents_from_conversations")
    @patch("services.fin_learning_service.FinLearningService._update_system_prompt")
    @patch("services.fin_learning_service.FinLearningService._extract_example_exchanges")
    @patch("services.fin_learning_service.FinLearningService._update_example_exchanges")
    @patch("services.fin_learning_service.FinLearningService._analyze_tool_usage")
    @patch("services.fin_learning_service.FinLearningService._update_tool_guidance")
    def test_process_learning_dataset(
        self, 
        mock_update_tool, 
        mock_analyze_tool, 
        mock_update_examples,
        mock_extract_examples,
        mock_update_prompt,
        mock_extract_intents,
        mock_extract_patterns,
        mock_load_nlp
    ):
        """Test processing a learning dataset"""
        # Set up mocks
        mock_extract_patterns.side_effect = [
            [{"type": "query_pattern", "intent": "forecast"}],  # success patterns
            [{"type": "query_pattern", "intent": "budget"}]     # failure patterns
        ]
        mock_extract_intents.return_value = {"forecast": ["new example"]}
        mock_extract_examples.return_value = [{"user": "query", "assistant": "response"}]
        mock_analyze_tool.return_value = [{"type": "tool_success_rate"}]
        
        # Create service
        service = FinLearningService()
        
        # Create test dataset
        dataset = {
            "helpful_conversations": [
                {
                    "messages": [
                        {"role": "user", "content": "forecast my spending"},
                        {"role": "assistant", "content": "Here's your forecast", "was_helpful": True, "feedback_rating": 5}
                    ]
                }
            ],
            "unhelpful_conversations": [
                {
                    "messages": [
                        {"role": "user", "content": "budget advice"},
                        {"role": "assistant", "content": "Bad response", "was_helpful": False, "feedback_rating": 2}
                    ]
                }
            ],
            "tool_usage": {
                "forecast_tool": {"total": 10, "success": 9}
            }
        }
        
        # Process dataset
        result = service.process_learning_dataset(dataset)
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["processed_conversations"], 2)
        
        # Verify method calls
        mock_extract_patterns.assert_any_call(dataset["helpful_conversations"], is_helpful=True)
        mock_extract_patterns.assert_any_call(dataset["unhelpful_conversations"], is_helpful=False)
        mock_extract_intents.assert_called_once_with(dataset["helpful_conversations"], dataset["unhelpful_conversations"])
        mock_update_prompt.assert_called_once()
        mock_extract_examples.assert_called_once_with(dataset["helpful_conversations"])
        mock_update_examples.assert_called_once()
        mock_analyze_tool.assert_called_once_with(dataset["tool_usage"])
        mock_update_tool.assert_called_once()

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_process_learning_dataset_error_handling(self, mock_load_nlp):
        """Test error handling in process_learning_dataset"""
        # Create service
        service = FinLearningService()
        
        # Test with empty dataset
        empty_dataset = {}
        result = service.process_learning_dataset(empty_dataset)
        self.assertEqual(result["status"], "error")
        
        # Test with dataset missing conversations
        invalid_dataset = {"tool_usage": {}}
        result = service.process_learning_dataset(invalid_dataset)
        self.assertEqual(result["status"], "error")

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_consolidate_tokens(self, mock_load_nlp):
        """Test token consolidation for entity extraction"""
        # Create service
        service = FinLearningService()
        
        # Test basic consolidation
        tokens = ["$", "50", ".25"]
        result = service._consolidate_tokens(tokens)
        self.assertEqual(result, ["$50.25"])
        
        # Test with entity spans
        tokens = ["The", "White", "House", "announced", "a", "new", "policy"]
        entity_spans = [{"start": 1, "end": 3}]
        result = service._consolidate_tokens(tokens, entity_spans)
        self.assertIn("White House", result)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_analyze_tool_usage(self, mock_load_nlp):
        """Test tool usage analysis"""
        # Create service
        service = FinLearningService()
        
        # Test tool usage data
        tool_usage = {
            "good_tool": {"total": 10, "success": 9},
            "ok_tool": {"total": 10, "success": 5},
            "bad_tool": {"total": 10, "success": 2},
            "insufficient_data": {"total": 2, "success": 0}
        }
        
        patterns = service._analyze_tool_usage(tool_usage)
        
        # Verify patterns
        self.assertEqual(len(patterns), 2)  # Should only include good_tool and bad_tool (meeting thresholds)
        
        # Verify high success rate pattern
        success_pattern = next((p for p in patterns if p["type"] == "tool_success_rate"), None)
        self.assertIsNotNone(success_pattern)
        self.assertEqual(success_pattern["tool"], "good_tool")
        self.assertAlmostEqual(success_pattern["success_rate"], 0.9)
        
        # Verify high failure rate pattern
        failure_pattern = next((p for p in patterns if p["type"] == "tool_failure_rate"), None)
        self.assertIsNotNone(failure_pattern)
        self.assertEqual(failure_pattern["tool"], "bad_tool")
        self.assertAlmostEqual(failure_pattern["failure_rate"], 0.8)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    @patch("services.fin_learning_service.FinLearningService._extract_intent")
    @patch("services.fin_learning_service.FinLearningService._classify_query_type")
    def test_analyze_tool_effectiveness(self, mock_classify, mock_extract_intent, mock_load_nlp):
        """Test tool effectiveness analysis"""
        # Set up mocks
        mock_extract_intent.return_value = "forecast"
        mock_classify.return_value = "financial_forecast"
        
        # Create service
        service = FinLearningService()
        
        # Test data
        user_query = "forecast my spending"
        assistant_response = "Here's your forecast"
        successful_tools = [{"name": "forecast_tool", "success": True}]
        
        # Test successful tool in helpful response
        patterns = service._analyze_tool_effectiveness(user_query, assistant_response, successful_tools, is_helpful=True)
        
        # Verify patterns
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern["type"], "tool_success")
        self.assertEqual(pattern["tool"], "forecast_tool")
        self.assertEqual(pattern["intent"], "forecast")
        self.assertEqual(pattern["query_type"], "financial_forecast")
        
        # Test unsuccessful tool in unhelpful response
        unsuccessful_tools = [{"name": "forecast_tool", "success": False}]
        patterns = service._analyze_tool_effectiveness(user_query, assistant_response, unsuccessful_tools, is_helpful=False)
        
        # Verify patterns
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern["type"], "tool_failure")
        self.assertEqual(pattern["tool"], "forecast_tool")

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    @patch("builtins.open", new_callable=mock_open, read_data="existing prompt")
    def test_update_tool_guidance(self, mock_file, mock_load_nlp):
        """Test updating tool guidance in system prompt"""
        # Create service with mock system prompt
        service = FinLearningService()
        service.system_prompt_template = "existing prompt"
        
        # Test tool patterns
        tool_patterns = [
            {"type": "tool_success_rate", "tool": "good_tool", "success_rate": 0.9, "sample_size": 10},
            {"type": "tool_failure_rate", "tool": "bad_tool", "failure_rate": 0.8, "sample_size": 10}
        ]
        
        # Update tool guidance
        service._update_tool_guidance(tool_patterns)
        
        # Verify file was opened for writing
        mock_file.assert_called_with("prompts/fin_system_prompt.txt", "w")
        
        # Verify prompt was updated
        self.assertIn("TOOL USAGE RECOMMENDATIONS", service.system_prompt_template)
        self.assertIn("good_tool", service.system_prompt_template)
        self.assertIn("bad_tool", service.system_prompt_template)
        
        # Test with empty patterns (should not change prompt)
        original_prompt = service.system_prompt_template
        service._update_tool_guidance([])
        self.assertEqual(service.system_prompt_template, original_prompt)

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_extract_time_periods_regex(self, mock_load_nlp):
        """Test regex-based time period extraction"""
        # Create service
        service = FinLearningService()
        
        # Test various time expressions
        texts_and_expected = [
            ("last 3 months", ["last 3 months"]),
            ("next week", ["next 1 week"]),
            ("previous 2 years", ["previous 2 years"]),
            ("this quarter", ["this 1 quarter"]),
            ("the coming month", ["coming 1 month"])
        ]
        
        for text, expected in texts_and_expected:
            entities = []
            service._extract_time_periods_regex(text, entities)
            
            # Verify time period was extracted
            self.assertEqual(len(entities), 1)
            time_entity = entities[0]
            self.assertEqual(time_entity["type"], "time_period")
            
            # Verify values match expected
            for expected_value in expected:
                self.assertTrue(
                    any(expected_value in value for value in time_entity["values"]),
                    f"Failed to find '{expected_value}' in {time_entity['values']}"
                )

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_extract_financial_categories(self, mock_load_nlp):
        """Test financial category extraction"""
        # Create service
        service = FinLearningService()
        service.nlp = None  # Force fallback to simple matching
        
        # Test various financial categories
        categories_to_test = [
            "groceries", "dining", "utilities", "rent", "mortgage", 
            "transportation", "entertainment", "healthcare", "income"
        ]
        
        for category in categories_to_test:
            text = f"Show me my {category} transactions"
            entities = []
            service._extract_financial_categories(text, entities)
            
            # Verify category was extracted
            self.assertEqual(len(entities), 1)
            category_entity = entities[0]
            self.assertEqual(category_entity["type"], "category")
            self.assertIn(category, category_entity["values"])

    @patch("services.fin_learning_service.FinLearningService._load_nlp_models")
    def test_keyword_based_intent(self, mock_load_nlp):
        """Test keyword-based intent extraction fallback"""
        # Create service
        service = FinLearningService()
        
        # Test various intents
        intents_to_test = {
            "forecast my spending for next month": "forecast",
            "create a budget for me": "budget",
            "analyze my spending patterns": "analysis",
            "compare this month to last month": "comparison",
            "recommend ways to save money": "recommendation",
            "what is my current balance": "information",
            "something completely different": "other"
        }
        
        for text, expected_intent in intents_to_test.items():
            result = service._keyword_based_intent(text)
            self.assertEqual(result, expected_intent, f"Failed to extract intent '{expected_intent}' from '{text}'")


def generate_learning_dataset():
    """Generate synthetic learning dataset for manual testing"""
    helpful_conversations = [
        {
            "messages": [
                {"role": "user", "content": "forecast my spending for next month"},
                {
                    "role": "assistant", 
                    "content": "Based on your historical spending, I predict you'll spend about $2,345 next month.", 
                    "was_helpful": True, 
                    "feedback_rating": 5,
                    "tools_used": [{"name": "forecast_cash_flow", "success": True}]
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "how much did I spend on groceries last month?"},
                {
                    "role": "assistant", 
                    "content": "You spent $432.18 on groceries last month.", 
                    "was_helpful": True, 
                    "feedback_rating": 4,
                    "tools_used": [{"name": "analyze_transactions", "success": True}]
                }
            ]
        }
    ]
    
    unhelpful_conversations = [
        {
            "messages": [
                {"role": "user", "content": "should I invest in cryptocurrency?"},
                {
                    "role": "assistant", 
                    "content": "I don't have enough information about your financial situation to advise on cryptocurrency investments.", 
                    "was_helpful": False, 
                    "feedback_rating": 2,
                    "tools_used": []
                }
            ]
        }
    ]
    
    tool_usage = {
        "forecast_cash_flow": {"total": 50, "success": 45},
        "analyze_transactions": {"total": 100, "success": 95},
        "create_budget": {"total": 30, "success": 25},
        "detect_anomalies": {"total": 20, "success": 5}
    }
    
    return {
        "helpful_conversations": helpful_conversations,
        "unhelpful_conversations": unhelpful_conversations,
        "tool_usage": tool_usage
    }


def manual_test():
    """Run manual tests with synthetic data"""
    print("\n===== Testing FinLearningService =====")
    
    # Initialize service
    print("Initializing FinLearningService...")
    service = FinLearningService()
    
    # Test intent extraction
    test_queries = [
        "forecast my spending for next month",
        "create a budget for me",
        "analyze my spending patterns",
        "compare my spending to last month",
        "recommend ways to save money",
        "what's my current balance"
    ]
    
    print("\nTesting intent extraction:")
    for query in test_queries:
        intent = service._extract_intent(query)
        print(f"  Query: '{query}' → Intent: '{intent}'")
    
    # Test entity extraction
    entity_test_queries = [
        "I spent $123.45 on groceries last month",
        "Show me spending for the next 3 weeks",
        "How much did I spend at Walmart?",
        "What's my rent payment for this month?"
    ]
    
    print("\nTesting entity extraction:")
    for query in entity_test_queries:
        entities = service._extract_entities(query)
        print(f"  Query: '{query}'")
        for entity in entities:
            print(f"    {entity['type']}: {entity['values']}")
    
    # Test query classification
    query_types = [
        "How much did I spend last month?",
        "Create a budget for me",
        "Forecast my expenses for next month",
        "What if I spend $500 more on marketing?",
        "Where is my money going?",
        "Help me save for a vacation"
    ]
    
    print("\nTesting query type classification:")
    for query in query_types:
        query_type = service._classify_query_type(query)
        print(f"  Query: '{query}' → Type: '{query_type}'")
    
    # Test process_learning_dataset
    print("\nTesting process_learning_dataset...")
    dataset = generate_learning_dataset()
    result = service.process_learning_dataset(dataset)
    
    print("Dataset processing result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n===== FinLearningService Testing Complete =====")


if __name__ == "__main__":
    # Uncomment to run unit tests
    # unittest.main()
    
    # Run manual tests
    manual_test()