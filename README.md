# AI Co-Scientist System

A multi-agent system built on Claude Sonnet 4.0 that assists scientists in generating novel research hypotheses through iterative reasoning, debate, and tournament-based selection.

## Features

- **Multi-Agent Architecture**: Six specialized AI agents working together
- **Real-time Streaming**: See AI reasoning character-by-character
- **Enhanced Tournament System**: Multi-dimensional hypothesis evaluation
- **Literature Integration**: Smart search across multiple scientific databases with fallback support
- **Expert-in-the-Loop**: Scientists can provide feedback and guide the system
- **Distributed Processing**: Redis/Celery for scalable, fault-tolerant execution

## Quick Start

### Prerequisites

- Docker and Docker Compose
- **Required API keys:**
  - Anthropic (Claude Sonnet 4.0)
  - OpenAI (embeddings)
- **Optional API keys for enhanced literature search:**
  - Serper (Google Scholar search)
  - Semantic Scholar (academic papers)
  - Perplexity (recent research)

**Note:** The system works without external literature APIs using a built-in fallback system.

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CoScientist_v2
   ```

2. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys (see LITERATURE_SETUP.md for details)
   ```

3. **Start the system**
   ```bash
   # Development mode (no persistence)
   docker compose -f ops/docker-compose.yml up --build
   
   # With persistence (PostgreSQL + Chroma)
   export ENABLE_PERSISTENCE=true
   docker compose -f ops/docker-compose.yml --profile persistence up --build
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Usage

### Simple Mode
One-click research hypothesis generation:
1. Enter your research goal
2. Click "Start Research"
3. Watch the AI agents work in real-time
4. Review generated hypotheses

### Advanced Mode
Step-by-step control over the research process:
1. Configure research preferences
2. Control agent execution
3. Inject custom hypotheses
4. Provide expert feedback

## Architecture

```
┌─────────────────────────────────────────────────┐
│ SUPERVISOR AGENT                               │
│ • Orchestrates the entire process              │
│ • Schedules specialized agents                 │
└─────────────────────────────────────────────────┘
│
▼
┌────────── Multi-Agent Loop ──────────┐
│ 1 Generation → produce hypotheses    │
│ 2 Reflection → review quality        │
│ 3 Ranking → tournaments (Enhanced)   │
│ 4 Evolution → refine hypotheses      │
│ 5 Proximity → similarity clustering  │
│ 6 Meta-Review → system feedback      │
└──────────────────────────────────────┘
│
▼
Context Memory (PostgreSQL + Chroma)
│
▼
Research Overview / Final Results
```

## API Endpoints

- `POST /v1/run-simple` - One-shot research execution
- `WS /ws/auto-run` - Streaming endpoint for real-time updates
- `WS /ws/advanced` - Bidirectional step-by-step control
- `GET /v1/hypothesis/{id}` - Get hypothesis details
- `POST /v1/feedback` - Submit expert feedback

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm run test
```

## Configuration

Key environment variables:
- `ENABLE_PERSISTENCE` - Enable PostgreSQL + Chroma (default: false)
- `AUTH_ENABLED` - Enable authentication (default: false)
- `OTEL_ENABLED` - Enable observability (default: false)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub. 