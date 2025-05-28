"""
API router that maps Flask routes to controller methods.
This provides a clean separation between routing and business logic.
"""

from flask import Blueprint, jsonify
from controllers.categorization_controller import CategorizationController
from controllers.forecast_controller import ForecastController
from controllers.fin_controller import FinController
from controllers.budget_controller import BudgetController
from controllers.insights_controller import InsightsController
from controllers.anomaly_controller import AnomalyController
from controllers.account_controller import AccountController
from controllers.learning_controller import LearningController


def create_api_routes() -> Blueprint:
    """
    Create and configure API routes blueprint

    Returns:
        Flask Blueprint with configured routes
    """
    api_bp = Blueprint("api", __name__)

    # Initialize controllers
    categorization_controller = CategorizationController()
    forecast_controller = ForecastController()
    fin_controller = FinController()
    budget_controller = BudgetController()
    insights_controller = InsightsController()
    anomaly_controller = AnomalyController()
    account_controller = AccountController()
    learning_controller = LearningController()

    # Health check route
    @api_bp.route("/health", methods=["GET"])
    def health_check():
        """Health check to verify service is running"""
        return (
            jsonify(
                {
                    "status": "healthy",
                    "service": "cashly-ai-service",
                    "version": "1.0.0",
                }
            ),
            200,
        )

    # Categorization routes
    @api_bp.route("/categorize/transaction", methods=["POST"])
    def categorize_transaction():
        """Categorize a transaction based on its description and amount"""
        response_data, status_code = categorization_controller.categorize_transaction()
        return jsonify(response_data), status_code

    # Forecasting routes
    @api_bp.route("/forecast/cash_flow", methods=["POST"])
    def forecast_cash_flow():
        """Forecast cash flow based on historical transaction data"""
        response_data, status_code = forecast_controller.forecast_cash_flow()
        return jsonify(response_data), status_code

    @api_bp.route("/forecast/cash_flow/scenario", methods=["POST"])
    def forecast_cash_flow_scenario():
        """Forecast cash flow with scenario adjustments"""
        response_data, status_code = forecast_controller.forecast_cash_flow_scenario()
        return jsonify(response_data), status_code

    # Fin conversational AI routes
    @api_bp.route("/fin/conversations/query", methods=["POST"])
    def fin_query():
        """Process a natural language query using OpenAI Assistants"""
        response_data, status_code = fin_controller.process_query()
        return jsonify(response_data), status_code

    @api_bp.route("/fin/health", methods=["GET"])
    def fin_health_check():
        """Health check for the OpenAI Assistants system"""
        response_data, status_code = fin_controller.health_check()
        return jsonify(response_data), status_code

    @api_bp.route("/fin/analytics", methods=["POST"])
    def fin_analytics():
        """Get analytics for recent queries and assistant usage"""
        response_data, status_code = fin_controller.get_analytics()
        return jsonify(response_data), status_code

    # Budget routes
    @api_bp.route("/generate/budget", methods=["POST"])
    def generate_budget():
        """Generate budget recommendations based on spending patterns"""
        response_data, status_code = budget_controller.generate_budget()
        return jsonify(response_data), status_code

    # Insights routes
    @api_bp.route("/analyze/trends", methods=["POST"])
    def analyze_trends():
        """Analyze financial trends and patterns"""
        response_data, status_code = insights_controller.analyze_trends()
        return jsonify(response_data), status_code

    # Anomaly detection routes
    @api_bp.route("/detect/anomalies", methods=["POST"])
    def detect_anomalies():
        """Detect anomalous transactions"""
        response_data, status_code = anomaly_controller.detect_anomalies()
        return jsonify(response_data), status_code

    # Account management routes
    @api_bp.route("/fin/accounts/status", methods=["POST"])
    def get_account_status():
        """Get user account status for Fin queries"""
        response_data, status_code = account_controller.get_account_status()
        return jsonify(response_data), status_code

    @api_bp.route("/fin/accounts/initiate_connection", methods=["POST"])
    def initiate_account_connection():
        """Initiate a Plaid connection process from Fin"""
        response_data, status_code = account_controller.initiate_account_connection()
        return jsonify(response_data), status_code

    # Learning routes
    @api_bp.route("/fin/learn", methods=["POST"])
    def learn_from_feedback():
        """Learn from user feedback to improve Fin's capabilities"""
        response_data, status_code = learning_controller.learn_from_feedback()
        return jsonify(response_data), status_code

    return api_bp
