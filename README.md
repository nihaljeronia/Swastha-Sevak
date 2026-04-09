# Swastha Sevak

Agentic AI medical triage chatbot scaffold for rural India.

## What has been done so far

- Created the initial FastAPI scaffold with async startup lifespan.
- Added `app/core/config.py` for Pydantic settings and database configuration.
- Added `app/db/` with SQLAlchemy async session, database models, and CRUD stubs.
- Added `app/webhook/` routes, schemas, and service layer stubs.
- Added `app/nlp/`, `app/ml/`, and `app/agents/` stub modules for future model loading and triage logic.
- Added `app/tasks/` with `followup.py` and `alerts.py` as async BackgroundTasks stubs.
- Added async pytest scaffolding in `tests/` with `conftest.py` and a webhook route test.
- Removed Docker and Celery/Redis support from the scaffold.
- Updated configuration and environment example for local PostgreSQL usage.

## Current setup instructions

1. Install PostgreSQL locally.
2. Create a PostgreSQL database named `swasthya_sahayak`.
3. Copy `.env.example` to `.env` and update your secrets.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Project structure

- `app/` — main application package
- `app/webhook/` — webhook endpoint and request schema
- `app/db/` — database models, session, CRUD operations
- `app/nlp/` — NLP model wrapper stubs
- `app/ml/` — ML classifier stub
- `app/tasks/` — BackgroundTasks stubs for follow-up and alerts
- `tests/` — pytest async tests

## Notes

This repository currently contains scaffold code with minimal logic. The next steps are to implement the webhook processing pipeline, patient message persistence, NLP and translation integration, and agent state flow.
