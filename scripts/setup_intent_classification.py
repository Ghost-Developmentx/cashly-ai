import os
import sys
import subprocess


def install_dependencies():
    """Install required dependencies."""
    print("Installing intent classification dependencies...")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements_intent.txt"]
        )
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "models/intent_classifier",
        "data/training",
        "data/conversations",
        "logs/intent_classification",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def initialize_models():
    """Initialize intent classification models."""
    print("Initializing intent classification models...")

    try:
        sys.path.insert(0, os.getcwd())
        from services.intent_classification import IntentClassifier

        # Initialize classifier (this will create a default model)
        classifier = IntentClassifier()
        print("✅ Intent classifier initialized successfully")

        # Test with a sample query
        test_result = classifier.classify_intent("Show me my transactions")
        print(
            f"✅ Test classification successful: {test_result['intent']} ({test_result['confidence']:.2%})"
        )

        return True
    except Exception as e:
        print(f"❌ Failed to initialize models: {e}")
        return False


def main():
    """Main setup function."""
    print("Setting up Intent Classification Service for Cashly...")
    print("=" * 50)

    # Step 1: Create directories
    print("\n1. Creating directories...")
    create_directories()

    # Step 2: Install dependencies
    print("\n2. Installing dependencies...")
    if not install_dependencies():
        print("Setup failed at dependency installation")
        return

    # Step 3: Initialize models
    print("\n3. Initializing models...")
    if not initialize_models():
        print("Setup failed at model initialization")
        return

    print("\n✅ Intent Classification Service setup complete!")
    print("\nNext steps:")
    print(
        "1. Run 'python scripts/prepare_training_data.py' to process your conversation data"
    )
    print("2. Update your .env file with intent classification settings")
    print("3. Integrate with your existing Fin service")


if __name__ == "__main__":
    main()
