# AI Co-Scientist System

A multi-agent system built on Claude Sonnet 4.0 that assists scientists in generating novel research hypotheses through iterative reasoning and tournament-based selection.

## Features

- **Multi-Agent Architecture**: Six specialized AI agents working together (Supervisor, Generation, Reflection, Ranking, Evolution, Proximity, Meta-Review)
- **Real-time Streaming**: See AI reasoning character-by-character via WebSocket
- **Literature Integration**: Smart search across multiple scientific databases with fallback support
- **Multi-dimensional Hypothesis Evaluation**: Scoring on novelty, feasibility, impact, and testability
- **Tournament-based Selection**: Enhanced ranking system with Elo ratings
- **Hypothesis Evolution**: Iterative refinement through multiple generations

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
   git clone https://github.com/niashwin/CoScientist_v2.git
   cd CoScientist_v2
   ```

2. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Start the system**
   ```bash
   # Development mode (in-memory storage)
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
4. Review generated hypotheses with detailed scoring

### Advanced Mode (Planned)
Step-by-step control over the research process - currently in development.

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
│ 3 Ranking → tournaments             │
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

### Implemented Endpoints
- `POST /v1/run-simple` - One-shot research execution with streaming
- `WS /ws/auto-run` - WebSocket streaming for real-time updates
- `GET /v1/session/{session_id}` - Get session details and status
- `GET /v1/hypothesis/{id}` - Get detailed hypothesis information
- `GET /v1/sessions` - List recent research sessions
- `GET /v1/health` - Health check endpoint
- `GET /v1/stats` - System statistics
- `GET /v1/literature-logs/{session_id}` - Literature search logs

### Development Endpoints
- `POST /v1/feedback` - Submit expert feedback (basic implementation)
- `WS /ws/advanced` - Advanced mode WebSocket (in development)
- `POST /v1/sessions/{session_id}/pause` - Pause session (placeholder)
- `POST /v1/sessions/{session_id}/resume` - Resume session (placeholder)

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
# Frontend tests (configured with Vitest)
cd frontend
npm run test

# Backend tests (test infrastructure exists but tests need to be written)
cd backend
pytest
```

## Configuration

Key environment variables:
- `ENABLE_PERSISTENCE` - Enable PostgreSQL + Chroma (default: false)
- `ANTHROPIC_API_KEY` - Required for Claude Sonnet 4.0
- `OPENAI_API_KEY` - Required for embeddings
- `SERPER_API_KEY` - Optional for enhanced literature search
- `SEMANTIC_SCHOLAR_API_KEY` - Optional for academic papers
- `PERPLEXITY_API_KEY` - Optional for recent research

## Future Roadmap

The following features are planned for future development:

### Advanced Features (In Development)
- **Expert-in-the-Loop**: Scientists can provide feedback and guide the system
- **Distributed Processing**: Redis/Celery for scalable, fault-tolerant execution
- **Advanced Mode**: Full step-by-step control over the research process
- **Authentication System**: User management and session persistence

### Planned Integrations
- **Scientific Tools**: AlphaFold, ChEMBL, PubChem, UniProt integration
- **Enhanced Tournament System**: More sophisticated multi-dimensional evaluation
- **Observability**: OpenTelemetry integration for monitoring
- **Comprehensive Testing**: Full test suite for backend and frontend

### Future Enhancements
- **Hypothesis Injection**: Allow users to inject custom hypotheses
- **Real-time Collaboration**: Multi-user research sessions
- **Export Capabilities**: Research reports and hypothesis summaries
- **Advanced Analytics**: Research session insights and patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests (when test infrastructure is complete)
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub. 