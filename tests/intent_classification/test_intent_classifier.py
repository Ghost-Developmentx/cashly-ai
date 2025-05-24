import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.intent_classification import IntentClassifier, IntentService


class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = IntentClassifier()
        self.intent_service = IntentService()

    def test_basic_classification(self):
        """Test basic intent classification."""
        test_cases = [
            ("Show me my transactions", "transactions"),
            ("Connect my bank account", "accounts"),
            ("Create an invoice", "invoices"),
            ("Forecast my cash flow", "forecasting"),
            ("Help me budget", "budgets"),
            ("Analyze my spending", "insights"),
            ("Hello", "general"),
        ]

        for query, expected_intent in test_cases:
            result = self.classifier.classify_intent(query)
            self.assertEqual(
                result["intent"],
                expected_intent,
                f"Failed for query: '{query}' - expected {expected_intent}, got {result['intent']}",
            )

    def test_routing_service(self):
        """Test the intent routing service."""
        query = "Show me my transactions from last week"
        result = self.intent_service.classify_and_route(query)

        self.assertIn("classification", result)
        self.assertIn("routing", result)
        self.assertIn("recommended_assistant", result)
        self.assertEqual(result["recommended_assistant"], "transaction_assistant")

    def test_confidence_levels(self):
        """Test confidence level classification."""
        query = "Show me my recent transactions"
        result = self.intent_service.classify_and_route(query)

        confidence = result["classification"]["confidence"]
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
