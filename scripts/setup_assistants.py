import os
import sys
import subprocess
from dotenv import load_dotenv, set_key

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ["openai", "python-dotenv"]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def setup_environment():
    """Set up environment variables."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to your .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False

    print("✅ OpenAI API key found")
    return True


def create_assistants():
    """Create all Cashly assistants."""
    try:
        from services.openai_assistants.assistant_factory import AssistantFactory

        print("Creating Cashly assistants...")
        factory = AssistantFactory()

        # Create all assistants
        assistant_ids = factory.create_all_assistants()

        if not assistant_ids:
            print("❌ No assistants were created")
            return False

        print(f"✅ Created {len(assistant_ids)} assistants")

        # Update .env file with assistant IDs
        env_file = "../.env"
        for assistant_type, assistant_id in assistant_ids.items():
            env_var = f"{assistant_type.upper()}_ASSISTANT_ID"
            set_key(env_file, env_var, assistant_id)
            print(f"  {assistant_type}: {assistant_id}")

        return True

    except Exception as e:
        print(f"❌ Error creating assistants: {e}")
        return False


def test_assistants():
    """Test that assistants are working."""
    try:
        from services.openai_assistants.assistant_manager import AssistantManager

        print("\nTesting assistant manager...")
        manager = AssistantManager()

        # Health check
        health = manager.health_check()

        if health["status"] == "healthy":
            print("✅ All assistants are healthy")
            return True
        else:
            print(f"⚠️  Assistant health check: {health['status']}")

            if health["missing_assistants"]:
                print(f"Missing assistants: {', '.join(health['missing_assistants'])}")

            for assistant_type, status in health["assistants"].items():
                if status.get("status") == "error":
                    print(f"❌ {assistant_type}: {status.get('error')}")
                else:
                    print(f"✅ {assistant_type}: {status.get('status')}")

            return len(health["assistants"]) > 0

    except Exception as e:
        print(f"❌ Error testing assistants: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "services/openai_assistants",
        "services/openai_assistants/assistants",
        "services/openai_assistants/tools",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_init_files():
    """Create __init__.py files."""
    init_files = [
        "services/openai_assistants/__init__.py",
        "services/openai_assistants/assistants/__init__.py",
        "services/openai_assistants/tools/__init__.py",
    ]

    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# OpenAI Assistants for Cashly\n")
            print(f"✅ Created {init_file}")


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 OPENAI ASSISTANTS SETUP COMPLETE!")
    print("=" * 60)

    print("\nCreated Assistants:")
    print("1. Transaction Assistant - Handles transaction CRUD and analysis")
    print("2. Account Assistant - Manages account information and balances")
    print("3. Connection Assistant - Handles Plaid and Stripe Connect setup")
    print("4. Invoice Assistant - Creates and manages invoices")
    print("5. Forecasting Assistant - Generates cash flow predictions")
    print("6. Budget Assistant - Creates budget recommendations")
    print("7. Insights Assistant - Analyzes trends and detects anomalies")

    print("\nNext Steps:")
    print("1. Test the assistants with: python test_assistants.py")
    print("2. Integrate with your Rails backend")
    print("3. Update your Fin service to use the new routing")
    print("4. Deploy and test with real user queries")

    print("\nYour .env file has been updated with assistant IDs.")
    print("Keep these IDs secure and back them up!")


def main():
    """Main setup function."""
    print("🚀 SETTING UP OPENAI ASSISTANTS FOR CASHLY")
    print("=" * 50)

    # Check dependencies
    print("1. Checking dependencies...")
    check_dependencies()

    # Create directories
    print("\n2. Creating directories...")
    create_directories()
    create_init_files()

    # Set up environment
    print("\n3. Checking environment...")
    if not setup_environment():
        return False

    # Create assistants
    print("\n4. Creating assistants...")
    if not create_assistants():
        return False

    # Test assistants
    print("\n5. Testing assistants...")
    if not test_assistants():
        print("⚠️  Some assistants may have issues, but setup is complete")

    # Display next steps
    display_next_steps()

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
