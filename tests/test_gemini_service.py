from unittest.mock import AsyncMock, MagicMock
from google.genai.errors import APIError
import pytest

from rag_lib.gemini_service import GeminiModel, GeminiService


class DummyAPIError(APIError):
    """Custom test subclass to guarantee code and message attribute accessibility."""

    def __init__(self, message: str, code: int) -> None:
        self._message = message
        self._code = code

    @property
    def message(self) -> str:  # type: ignore
        return self._message

    @property
    def code(self) -> int:  # type: ignore
        return self._code

    def __str__(self) -> str:
        return self._message


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock()
    return client


@pytest.fixture
def service(mock_client: MagicMock) -> GeminiService:
    return GeminiService(
        model_name=GeminiModel.GEMINI_3_5_FLASH,
        google_api_key="test-key",
        session_secret="test-secret",
        client=mock_client,
    )


@pytest.mark.asyncio
async def test_generate_gemini_response_api_error(
    service: GeminiService, mock_client: MagicMock
) -> None:

    # Set the side effect to our concrete APIError subclass instance
    mock_client.aio.models.generate_content.side_effect = DummyAPIError(
        message="API Rate limit exceeded", code=429
    )

    result = await service.generate_gemini_response(
        prompt="Hello", session_id="session-123"
    )

    assert result["status"] == "error"
    assert result["code"] == 429
    assert result["message"] == "API Rate limit exceeded"
