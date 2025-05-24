import traceback
import json
from datetime import datetime

from flask import Flask, request, jsonify
import os
import logging
from flask_cors import CORS
import pandas as pd
from services.categorize_service import CategorizationService
from services.forecast_service import ForecastService
from services.budget_service import BudgetService
from services.insight_service import InsightService
from services.anomaly_service import AnomalyService
from services.fin.fin_service import FinService
from services.fin_learning_service import FinLearningService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app)

categorization_service = CategorizationService()
forecast_service = ForecastService()
budget_service = BudgetService()
insight_service = InsightService()
anomaly_service = AnomalyService()
fin_service = FinService()
fin_learning_service = FinLearningService()


# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime("%Y-%m-%d")
        return super(DateTimeEncoder, self).default(obj)


app.json_encoder = DateTimeEncoder

# Directory For Saving Trained Models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check to verify service is running"""
    return (
        jsonify(
            {"status": "healthy", "service": "cashly-ai-service", "version": "1.0.0"}
        ),
        200,
    )


@app.route("/categorize/transaction", methods=["POST"])
def categorize_transaction():
    """
    Categorize a transaction based on its description and amount

    Expected input:
    {
        "description": "AMAZON PAYMENT",
        "amount": -45.67,
        "date": "2025-03-10"
    }
    """
    try:
        data = request.json
        description = data.get("description", "")
        amount = data.get("amount", 0)
        date = data.get("date")

        logger.info(f"Categorizing transaction: {description}, ${amount}")

        result = categorization_service.categorize_transaction(
            description=description, amount=amount, date=date
        )

        logger.info(
            f"Categorization result: {result['category']} (confidence: {result['confidence']:.2f})"
        )

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error categorizing transaction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/cash_flow", methods=["POST"])
def forecast_cash_flow():
    """
    Forecast cash flow based on historical transaction data

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": 1000.00, "category": "income"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "forecast_days": 30
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        forecast_days = data.get("forecast_days", 30)

        if not user_id or not transactions:
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(
            f"Forecasting cash flow for user {user_id} for {forecast_days} days"
        )

        result = forecast_service.forecast_cash_flow(
            user_id=user_id, transactions=transactions, forecast_days=forecast_days
        )

        logger.info(
            f"Forecast completed with {len(result.get('forecast', []))} days predicted"
        )

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error forecasting cash flow: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate/budget", methods=["POST"])
def generate_budget():
    """
    Generate budget recommendations based on spending patterns

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "income": 5000.00
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        monthly_income = data.get("income", 0)

        if not user_id or not transactions:
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(
            f"Generating budget recommendations for user {user_id} with income ${monthly_income}"
        )

        result = budget_service.generate_budget(
            user_id=user_id, transactions=transactions, monthly_income=monthly_income
        )

        logger.info(
            f"Budget recommendations generated for {len(result.get('recommended_budget', {}).get('by_category', {}))} categories"
        )

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error generating budget recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/trends", methods=["POST"])
def analyze_trends():
    """
    Analyze financial trends and patterns

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "period": "3m"  # 1m, 3m, 6m, 1y
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        period = data.get("period", "3m")

        if not user_id or not transactions:
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(f"Analyzing trends for user {user_id} over period {period}")

        result = insight_service.analyze_trends(
            user_id=user_id, transactions=transactions, period=period
        )

        logger.info(
            f"Trend analysis completed with {len(result.get('insights', []))} insights generated"
        )

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect/anomalies", methods=["POST"])
def detect_anomalies():
    """
    Detect anomalous transactions

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "threshold": -0.5  # Optional anomaly score threshold
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        threshold = data.get("threshold")

        if not user_id or not transactions:
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(f"Detecting anomalies for user {user_id}")

        result = anomaly_service.detect_anomalies(
            user_id=user_id, transactions=transactions, threshold=threshold
        )

        logger.info(
            f"Anomaly detection completed with {len(result.get('anomalies', []))} anomalies found"
        )

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/cash_flow/scenario", methods=["POST"])
def forecast_cash_flow_scenario():
    """
    Forecast cash flow with scenario adjustments

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": 1000.00, "category": "income"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "forecast_days": 30,
        "adjustments": {
            "category_adjustments": {"1": 500, "2": -200},
            "income_adjustment": 1000,
            "expense_adjustment": 500,
            "recurring_transactions": [
                {"description": "New Salary", "amount": 5000, "frequency": "monthly"},
                ...
            ]
        }
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        forecast_days = data.get("forecast_days", 30)
        adjustments = data.get("adjustments", {})

        if not user_id or not transactions:
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(f"Generating scenario forecast for user {user_id}")

        result = forecast_service.forecast_cash_flow_scenario(
            user_id=user_id,
            transactions=transactions,
            forecast_days=forecast_days,
            adjustments=adjustments,
        )

        logger.info(f"Scenario forecast completed")

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error generating scenario forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/fin/query", methods=["POST"])
def fin_query():
    """
    Process a natural language query for the financial assistant.
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        query = data.get("query")
        transactions = data.get("transactions", [])
        conversation_history = data.get("conversation_history", [])
        user_context = data.get("user_context", {})

        if not user_id or not query:
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(f"📥 Fin query from user {user_id}: {query}")
        logger.info(
            f"📥 User context accounts: {len(user_context.get('accounts', []))}"
        )
        logger.info(f"📥 Transactions provided: {len(transactions)}")

        result = fin_service.process_query(
            user_id=user_id,
            query=query,
            transactions=transactions,
            conversation_history=conversation_history,
            user_context=user_context,
        )

        logger.info(f"📤 Fin response generated successfully")
        logger.info(f"📤 Response keys: {list(result.keys())}")
        logger.info(f"📤 Actions count: {len(result.get('actions', []))}")
        logger.info(f"📤 Tool results count: {len(result.get('tool_results', []))}")

        # Log detailed action information
        if result.get("actions"):
            for i, action in enumerate(result["actions"]):
                logger.info(
                    f"📤 Action {i}: type={action.get('type')}, has_data={bool(action.get('data'))}"
                )
                if action.get("data") and isinstance(action["data"], dict):
                    logger.info(
                        f"📤 Action {i} data keys: {list(action['data'].keys())}"
                    )
                    # Special logging for transactions and accounts
                    if action.get(
                        "type"
                    ) == "show_transactions" and "transactions" in action.get(
                        "data", {}
                    ):
                        logger.info(
                            f"📤 Action {i} transaction count: {len(action['data']['transactions'])}"
                        )
                    elif action.get(
                        "type"
                    ) == "show_accounts" and "accounts" in action.get("data", {}):
                        logger.info(
                            f"📤 Action {i} account count: {len(action['data']['accounts'])}"
                        )
        else:
            logger.warning("⚠️ No actions in the response!")

        # Log if tool_results exist but no actions
        if result.get("tool_results") and not result.get("actions"):
            logger.error("❌ Tool results present but no actions generated!")
            for i, tool_result in enumerate(result["tool_results"]):
                logger.info(f"🔧 Tool result {i}: tool={tool_result.get('tool')}")

        # Ensure we're returning the complete result
        logger.info(
            f"📤 Final response has response_text: {bool(result.get('response_text'))}"
        )
        logger.info(f"📤 Final response has actions: {bool(result.get('actions'))}")

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"❌ Error processing Fin query: {str(e)}")
        logger.error(f"❌ Full traceback:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/fin/accounts/status", methods=["POST"])
def get_account_status():
    """
    Get user account status for Fin queries

    Expected input:
    {
        "user_id": "user_123",
        "user_context": {
            "accounts": [...],
            "budgets": [...],
            ...
        }
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        user_context = data.get("user_context", {})

        if not user_id:
            return jsonify({"error": "Missing required parameters"}), 400

        accounts = user_context.get("accounts", [])

        result = {
            "account_count": len(accounts),
            "accounts": accounts,
            "total_balance": sum(acc.get("balance", 0) for acc in accounts),
            "has_accounts": len(accounts) > 0,
            "plaid_connected": any(acc.get("plaid_account_id") for acc in accounts),
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting account status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/fin/accounts/initiate_connection", methods=["POST"])
def initiate_account_connection():
    """
    Initiate a Plaid connection process from Fin

    Expected input:
    {
        "user_id": "user_123",
        "institution_preference": "major_bank" (optional)
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        institution_preference = data.get("institution_preference")

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        result = {
            "action": "initiate_plaid_connection",
            "user_id": user_id,
            "institution_preference": institution_preference,
            "message": "I'll help you connect your bank account securely through Plaid.",
            "next_step": "plaid_link_token",
            "instructions": [
                "Click the 'Connect Bank Account' button below",
                "Select your bank from the list",
                "Enter your online banking credentials",
                "Select which accounts to connect",
                "Complete the verification process",
            ],
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error initiating account connection: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/fin/learn", methods=["POST"])
def learn_from_feedback():
    """
    Learn from user feedback to improve Fin's capabilities.

    Expected input:
    {
        "dataset": {
            "helpful_conversations": [
                {
                    "id": 123,
                    "messages": [...],
                    "user_id": "user_123"
                },
                ...
            ],
            "unhelpful_conversations": [...],
            "tool_usage": {
                "forecast_cash_flow": {
                    "total": 15,
                    "success": 12,
                    "contexts": ["What if I spend $5000 on marketing?", ...],
                    "parameters": [{"amount": 5000, "category": "marketing"}, ...]
                },
                ...
            }
        }
    }
    """
    try:
        data = request.json
        dataset = data.get("dataset", {})

        if not dataset or not any(dataset.values()):
            return (
                jsonify({"status": "error", "message": "Empty dataset provided"}),
                400,
            )

        learning_service = FinLearningService()
        result = learning_service.process_learning_dataset(dataset)

        return jsonify(json.loads(json.dumps(result, cls=DateTimeEncoder))), 200

    except Exception as e:
        logger.error(f"Error in learning process: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
