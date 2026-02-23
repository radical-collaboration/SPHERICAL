"""Tests for server module."""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

# Import server module functions (mocking dependencies)
with patch.dict("sys.modules", {"src.inference.inference_service": MagicMock()}):
    from src.inference.server import create_app, health_handler, info_handler, root_handler


class TestServerHandlers:
    """Tests for server HTTP handlers."""

    @pytest.mark.asyncio
    async def test_root_handler(self):
        """Test root endpoint returns service info."""
        request = MagicMock()

        response = await root_handler(request)

        assert response.status == 200
        data = response.body.decode()
        assert "Inference Server" in data
        assert "health" in data
        assert "generate" in data

    @pytest.mark.asyncio
    async def test_health_handler_no_service(self):
        """Test health endpoint when service not initialized."""
        request = MagicMock()

        with patch("server.inference_service", None):
            response = await health_handler(request)

        assert response.status == 503

    @pytest.mark.asyncio
    async def test_info_handler_no_service(self):
        """Test info endpoint when service not initialized."""
        request = MagicMock()

        with patch("server.inference_service", None):
            response = await info_handler(request)

        assert response.status == 503


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_application(self):
        """Test that create_app returns an aiohttp Application."""
        app = create_app()

        assert isinstance(app, web.Application)

    def test_create_app_has_routes(self):
        """Test that app has required routes."""
        app = create_app()

        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]

        assert "/" in routes
        assert "/health" in routes
        assert "/info" in routes
        assert "/generate" in routes

    def test_create_app_has_lifecycle_handlers(self):
        """Test that app has startup and shutdown handlers."""
        app = create_app()

        assert len(app.on_startup) > 0
        assert len(app.on_shutdown) > 0
