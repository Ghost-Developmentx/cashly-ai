#!/usr/bin/env python
"""
Train initial models with synthetic data and save to MLflow.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.synthetic_data.training_data_generator import TrainingDataGenerator
from app.services.ml.model_manager import model_manager
from app.core.mlflow_config import mlflow_manager

from dotenv import load_dotenv

# Load AWS credentials from .env.docker
load_dotenv(".env.docker")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_all_models():
    """Train all models with synthetic data."""

    # Initialize MLflow
    mlflow_manager.initialize()

    # Generate training data
    logger.info("📊 Generating synthetic training data...")
    generator = TrainingDataGenerator()

    # Generate 2 years of data for better patterns
    training_data = generator.generate_all_training_data(
        num_days=730,
        num_users=1
    )

    # Train each model
    models_config = [
        {
            'type': 'categorizer',
            'data_key': 'categorization',
            'name': 'Transaction Categorizer'
        },
        {
            'type': 'forecaster',
            'data_key': 'forecasting',
            'name': 'Cash Flow Forecaster'
        },
        {
            'type': 'anomaly_detector',
            'data_key': 'anomaly',
            'name': 'Anomaly Detector'
        },
        {
            'type': 'budget_recommender',
            'data_key': 'budgeting',
            'name': 'Budget Recommender'
        },
        {
            'type': 'trend_analyzer',
            'data_key': 'trend',
            'name': 'Trend Analyzer'
        }
    ]

    results = {}

    for config in models_config:
        logger.info(f"\n🎯 Training {config['name']}...")

        try:
            # Get training data
            data = training_data[config['data_key']]

            # Train model
            result = await model_manager.train_model(
                model_type=config['type'],
                training_data=data
            )

            if result['success']:
                logger.info(f"✅ {config['name']} trained successfully!")
                logger.info(f"   Metrics: {result.get('metrics', {})}")
                logger.info(f"   Model URI: {result.get('model_uri', 'N/A')}")
            else:
                logger.error(f"❌ {config['name']} training failed: {result.get('error')}")

            results[config['type']] = result

        except Exception as e:
            logger.error(f"❌ Error training {config['name']}: {e}")
            results[config['type']] = {'success': False, 'error': str(e)}

    # Summary
    logger.info("\n📈 Training Summary:")
    logger.info("=" * 50)

    successful = sum(1 for r in results.values() if r.get('success'))
    logger.info(f"Models trained successfully: {successful}/{len(models_config)}")

    for model_type, result in results.items():
        status = "✅" if result.get('success') else "❌"
        logger.info(f"{status} {model_type}")

    return results


async def verify_models():
    """Verify that all models can be loaded properly."""
    logger.info("\n🔍 Verifying models can be loaded...")

    model_types = ['categorizer', 'forecaster', 'anomaly_detector', 'budget_recommender', 'trend_analyzer']

    for model_type in model_types:
        try:
            model = await model_manager.get_model(model_type, force_reload=True)

            # Check if the model is actually fitted/loaded
            if hasattr(model, 'model') and model.model is not None:
                logger.info(f"✅ {model_type} loaded successfully")
            elif hasattr(model, 'is_fitted') and model.is_fitted:
                logger.info(f"✅ {model_type} loaded successfully")
            else:
                logger.warning(f"⚠️  {model_type} loaded but not fitted")

        except Exception as e:
            logger.error(f"❌ Failed to load {model_type}: {e}")



async def test_predictions():
    """Test that models can make predictions."""
    logger.info("\n🧪 Testing model predictions...")

    # Generate small test dataset
    generator = TrainingDataGenerator()
    test_data = generator.generate_all_training_data(num_days=7)

    # Test categorizer
    try:
        categorizer = await model_manager.get_model('categorizer')
        test_transaction = test_data['categorization'].iloc[0]
        result = await categorizer.predict_with_confidence(
            test_data['categorization'].head(1)
        )
        logger.info(f"✅ Categorizer prediction: {result[0]['category']} "
                    f"(confidence: {result[0]['confidence']:.2f})")
    except Exception as e:
        logger.error(f"❌ Categorizer test failed: {e}")

    # Test forecaster
    try:
        forecaster = await model_manager.get_model('forecaster')
        forecast = forecaster.forecast(test_data['forecasting'], horizon=7)
        logger.info(f"✅ Forecaster predicted {len(forecast)} days")
    except Exception as e:
        logger.error(f"❌ Forecaster test failed: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train initial ML models")
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only verify existing models'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run prediction tests after training'
    )

    args = parser.parse_args()

    if not args.skip_training:
        # Train models
        asyncio.run(train_all_models())

    # Verify models
    asyncio.run(verify_models())

    if args.test:
        # Test predictions
        asyncio.run(test_predictions())

    logger.info("\n🎉 Model initialization complete!")
    logger.info("View your models at: http://localhost:5000")


if __name__ == "__main__":
    main()