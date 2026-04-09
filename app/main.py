from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agents.graph import TriageAgent
from app.core.config import settings
from app.ml.classifier import DiseaseClassifier
from app.nlp.models import NLPModelManager
from app.webhook.routes import router as webhook_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nlp_manager = NLPModelManager()
    await app.state.nlp_manager.load_models()
    app.state.classifier = DiseaseClassifier()
    await app.state.classifier.load_model()
    app.state.agent = TriageAgent()
    await app.state.agent.load_model()
    yield
    await app.state.nlp_manager.shutdown()
    await app.state.classifier.shutdown()
    await app.state.agent.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(webhook_router, prefix="/webhook", tags=["webhook"])

    @app.get("/")
    async def root():
        return {"message": "Swastha Sevak API", "status": "running"}

    return app


app = create_app()
