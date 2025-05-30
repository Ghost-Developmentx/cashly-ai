#!/usr/bin/env python3
"""
Updated setup script using the enhanced seeding system.
This will create a robust intent classification system with lots of training data.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_environment():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print("✅ Loaded environment from .env file")
            return True
        else:
            print("❌ .env file not found")
            return False
    except ImportError:
        print("❌ python-dotenv not installed. Run: pip install python-dotenv")
        return False


def setup_logging():
    """Configure logging for the setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("enhanced_vector_setup.log"),
        ],
    )


def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing_vars = []
    print("Environment variables status:")
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"  ❌ {var}: Not set")
        else:
            # Show partial values for security
            if "KEY" in var or "PASSWORD" in var:
                display_value = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")

    if missing_vars:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease check your .env file contains:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return False

    print("\n✅ All required environment variables are set")
    return True


def test_database_connection():
    """Test the database connection."""
    try:
        from db.init import get_db_connection

        print("🔄 Testing database connection...")
        db = get_db_connection()
        if db.test_connection():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False


def test_openai_connection():
    """Test the OpenAI API connection."""
    try:
        from services.embeddings.openai_client import OpenAIEmbeddingClient

        print("🔄 Testing OpenAI API connection...")
        client = OpenAIEmbeddingClient()
        test_embedding = client.create_embedding("test connection")

        if test_embedding and len(test_embedding) > 0:
            print("✅ OpenAI API connection successful")
            print(f"  Embedding dimension: {len(test_embedding)}")
            return True
        else:
            print("❌ OpenAI API connection failed - no embedding returned")
            return False
    except Exception as e:
        print(f"❌ OpenAI API connection error: {e}")
        return False


def run_migrations():
    """Run database migrations."""
    try:
        from db.init import DatabaseInitializer

        print("🔄 Running database migrations...")
        initializer = DatabaseInitializer()

        if initializer.initialize():
            print("✅ Database migrations completed")
            return True
        else:
            print("❌ Database migrations failed")
            return False
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False


def seed_enhanced_database():
    """Seed the database with comprehensive conversation data."""
    try:
        from seed_data.enhanced_vector_seeder import EnhancedVectorSeeder

        print("🔄 Starting enhanced database seeding...")
        print("  This will create hundreds of sample conversations for training")

        seeder = EnhancedVectorSeeder()

        # Seed with multiple users for diversity
        user_ids = [f"training_user_{i}" for i in range(1, 8)]  # 7 users

        print(f"  Seeding for {len(user_ids)} different user profiles...")
        stats = seeder.seed_comprehensive_database(user_ids)

        print("✅ Enhanced database seeding completed!")
        print(f"  📊 Total conversations: {stats['total_seeded']}")
        print(f"  ❌ Failed: {stats['failed']}")
        print("  📈 By Intent:")
        for intent, count in stats["by_intent"].items():
            print(f"    {intent}: {count} conversations")

        return True

    except Exception as e:
        print(f"❌ Enhanced seeding error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_intent_classification():
    """Test the intent classification system with various queries."""
    try:
        from services.intent_classification.intent_service import IntentService

        print("🔄 Testing enhanced intent classification...")
        intent_service = IntentService()

        # Test queries that should now work well
        test_queries = [
            "Show me all my invoices",
            "Can you show me my invoice list?",
            "What's my account balance?",
            "Show me recent transactions",
            "Create a budget for next month",
        ]

        successful_classifications = 0

        for query in test_queries:
            try:
                result = intent_service.classify_and_route(query)
                intent = result["classification"]["intent"]
                confidence = result["classification"]["confidence"]
                method = result["classification"]["method"]

                print(f"  '{query}'")
                print(f"    -> {intent} ({confidence:.1%}) via {method}")

                if confidence > 0.6:  # Good confidence
                    successful_classifications += 1

            except Exception as e:
                print(f"  ❌ Error classifying '{query}': {e}")

        success_rate = successful_classifications / len(test_queries)
        print(f"\n✅ Intent classification test completed")
        print(
            f"  📊 Success rate: {success_rate:.1%} ({successful_classifications}/{len(test_queries)})"
        )

        return success_rate >= 0.6

    except Exception as e:
        print(f"❌ Intent classification test error: {e}")
        return False


def verify_vector_database():
    """Verify the vector database has sufficient data."""
    try:
        from seed_data.enhanced_vector_seeder import EnhancedVectorSeeder

        print("🔄 Verifying vector database...")
        seeder = EnhancedVectorSeeder()

        verification_results = seeder.verify_seeding()

        print("✅ Vector database verification completed")
        print("  🔍 Sample searches:")

        good_results = 0
        for query, result in verification_results.items():
            if "error" not in result:
                found = result["found_similar"]
                expected = result["expected_intent"]
                top_intent = result["top_intent"]
                similarity = result["top_similarity"]

                print(f"    '{query}'")
                print(
                    f"      -> Found {found} similar, top: {top_intent} ({similarity:.1%})"
                )

                if found >= 3 and similarity > 0.7:
                    good_results += 1
            else:
                print(f"    ❌ '{query}' -> {result['error']}")

        success_rate = good_results / len(verification_results)
        print(f"  📊 Verification success rate: {success_rate:.1%}")

        return success_rate >= 0.8

    except Exception as e:
        print(f"❌ Vector database verification error: {e}")
        return False


def main():
    """Main setup function with enhanced seeding."""
    print("🚀 Setting up Enhanced Cashly Vector Database...")
    print("=" * 60)

    # Step 0: Load environment variables FIRST
    print("\n0️⃣ Loading Environment Variables")
    if not load_environment():
        print("⚠️ Continuing without .env file - using system environment variables")

    setup_logging()

    # Step 1: Check environment
    print("\n1️⃣ Checking Environment")
    if not check_environment():
        return False

    # Step 2: Test database connection
    print("\n2️⃣ Testing Database Connection")
    if not test_database_connection():
        return False

    # Step 3: Test OpenAI connection
    print("\n3️⃣ Testing OpenAI Connection")
    if not test_openai_connection():
        return False

    # Step 4: Run migrations
    print("\n4️⃣ Running Database Migrations")
    if not run_migrations():
        return False

    # Step 5: Enhanced database seeding
    print("\n5️⃣ Enhanced Database Seeding")
    if not seed_enhanced_database():
        return False

    # Step 6: Test intent classification
    print("\n6️⃣ Testing Intent Classification")
    if not test_intent_classification():
        print("  ⚠️ Intent classification had issues, but setup completed")

    print("\n" + "=" * 60)
    print("🎉 Enhanced Vector Database Setup Complete!")
    print("\n🚀 Next Steps:")
    print("  1. Restart your Flask application: python app.py")
    print("  2. Test the query: 'Show me all my invoices'")
    print("  3. Check logs for: 🎯 Intent: invoices")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
