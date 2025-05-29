# Cashly AI

Cashly AI is an intelligent financial assistant platform that helps users manage their finances through natural language interactions. It uses advanced AI techniques to understand user queries, classify intents, and route requests to specialized assistants for various financial tasks.

## Features

- **Intent Classification**: Automatically classifies user queries into different financial categories
- **Specialized Financial Assistants**: Dedicated assistants for:
  - Transactions
  - Accounts
  - Invoices
  - Bank Connections
  - Payment Processing
  - Forecasting
  - Budgets
  - Financial Insights
- **Context-Aware Responses**: Considers user context and conversation history for better understanding
- **Hybrid Classification**: Uses both traditional ML and embedding-based approaches for intent classification
- **Confidence-Based Routing**: Routes queries to specialized assistants based on confidence levels
- **Fallback Mechanisms**: Provides alternative options when intent classification confidence is low
- **Vector Database**: Stores and retrieves semantic embeddings for improved understanding
- **Continuous Learning**: Improves over time with new conversation data

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- PostgreSQL with pgvector extension

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cashly-ai.git
   cd cashly-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your API keys and configuration.

5. Start the PostgreSQL database with pgvector:
   ```bash
   make docker-up
   ```

6. Initialize the database schema:
   ```bash
   make db-setup
   ```

## Configuration

The application is configured through environment variables in the `.env` file:

- **OpenAI Configuration**: API keys, model selection, and parameters
- **Assistant IDs**: IDs for specialized OpenAI assistants
- **Service Configuration**: URLs and API keys for external services
- **Database Configuration**: PostgreSQL connection parameters

## Usage

### Starting the Application

Run the Flask application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

### API Endpoints

- `/api/chat`: Send user messages and get AI responses
- `/api/intent/classify`: Classify user intent without generating a response
- `/api/analytics/intent`: Get analytics about intent classification

### Example Request

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me my recent transactions",
    "user_id": "user123"
  }'
```

## Project Structure

```
cashly-ai/
├── app.py                  # Main application entry point
├── config/                 # Configuration modules
├── controllers/            # API controllers
├── db/                     # Database models and connections
├── models/                 # ML model storage
├── routes/                 # API route definitions
├── scripts/                # Utility scripts
├── services/               # Business logic services
│   ├── embeddings/         # Vector embedding services
│   ├── intent_classification/ # Intent classification services
│   └── ...
├── tests/                  # Test suite
├── util/                   # Utility functions
├── docker-compose.yml      # Docker configuration
├── Makefile                # Build and run commands
└── requirements.txt        # Python dependencies
```

## Development

### Database Management

- Start database: `make docker-up`
- Stop database: `make docker-down`
- View logs: `make docker-logs`
- Access database shell: `make db-shell`
- Clean database: `make docker-clean`

### Running Tests

```bash
python -m pytest tests/
```

## Security Notes

- Never commit `.env` files with real API keys
- Rotate API keys regularly
- Use environment-specific configurations for development, testing, and production

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]