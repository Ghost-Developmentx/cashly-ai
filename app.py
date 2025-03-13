from flask import Flask, request, jsonify
import os
import logging
from flask_cors import CORS
from services.categorize_service import CategorizationService
from services.forecast_service import ForecastService
from services.budget_service import BudgetService
from services.insight_service import InsightService
from services.anomaly_service import AnomalyService

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

        return jsonify(result), 200

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

        return jsonify(result), 200

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

        return jsonify(result), 200

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

        return jsonify(result), 200

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

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
