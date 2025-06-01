import json
import sys
import os
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import ConversationDataProcessor


def load_conversation_data(file_path: str) -> List[Dict]:
    """Load conversation data from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Conversation file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return []


def load_tool_usage_data(file_path: str) -> Dict:
    """Load tool usage statistics from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Tool usage file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return {}


def main():
    """Main function to process training data."""
    processor = ConversationDataProcessor()

    # File paths (adjust based on your data location)
    conversations_file = "data/conversations.json"
    tool_usage_file = "data/tool_usage_stats.json"
    output_file = "data/training/intent_training_data.csv"

    print("Loading conversation data...")
    conversations = load_conversation_data(conversations_file)
    print(f"Loaded {len(conversations)} conversations")

    print("Loading tool usage data...")
    tool_usage = load_tool_usage_data(tool_usage_file)
    print(f"Loaded tool usage data for {len(tool_usage)} tools")

    print("Processing training data...")
    df = processor.create_training_dataset(
        conversations=conversations, tool_usage=tool_usage, output_file=output_file
    )

    print(f"Created training dataset with {len(df)} samples")

    # Analyze the dataset
    print("\nDataset Analysis:")
    analysis = processor.analyze_intent_distribution(df)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Unique intents: {analysis['unique_intents']}")
    print("\nIntent distribution:")
    for intent, count in analysis["intent_distribution"].items():
        percentage = analysis["intent_percentages"][intent]
        print(f"  {intent}: {count} samples ({percentage:.1f}%)")

    # Augment if needed
    if analysis["min_samples_per_intent"] < 10:
        print("\nAugmenting training data...")
        augmented_df = processor.augment_training_data(df, target_samples_per_intent=15)

        # Save augmented data
        augmented_output = output_file.replace(".csv", "_augmented.csv")
        augmented_df.to_csv(augmented_output, index=False)
        print(f"Saved augmented dataset to {augmented_output}")

    print("Training data preparation complete!")


if __name__ == "__main__":
    main()
