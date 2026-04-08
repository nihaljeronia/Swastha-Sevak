import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from app.main import create_app


@pytest.fixture
async def app():
    return create_app()


@pytest.fixture
async def async_client(app):
    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client
