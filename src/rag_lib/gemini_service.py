from enum import StrEnum
import hashlib
import hmac
from typing import Any

from google.genai import Client, types
from google.genai.errors import APIError


class GeminiModel(StrEnum):
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


class GeminiService:
    def __init__(
        self,
        model_name: str | GeminiModel,
        google_api_key: str,
        session_secret: str,
        client: Client | None = None,
    ) -> None:
        valid_models = set(GeminiModel)
        if model_name not in valid_models:
            allowed = ", ".join(repr(m.value) for m in GeminiModel)
            raise ValueError(
                f"Invalid model_name '{model_name}'. Expected one of: {allowed}"
            )

        self.__google_api_key: str = google_api_key
        self.__session_secret: str = session_secret
        self._model_name: GeminiModel = GeminiModel(model_name)
        self._client: Client = client or Client(api_key=self.__google_api_key)

    def _generate_session_signature(self, data: str) -> str:
        return hmac.new(
            self.__session_secret.encode("utf-8"),
            data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def generate_gemini_response(
        self, prompt: str, session_id: str
    ) -> dict[str, Any]:
        session_signature: str = self._generate_session_signature(session_id)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_name.value,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )

            return {
                "status": "success",
                "content": response.text,
                "session_id": session_id,
                "signature": session_signature,
            }

        except APIError as e:
            # Safely extract status code from APIError across SDK variations
            code = getattr(e, "code", getattr(e, "status_code", 500))
            message = getattr(e, "message", str(e))
            return {
                "status": "error",
                "code": code,
                "message": message,
                "session_id": session_id,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected runtime error: {str(e)}",
                "session_id": session_id,
            }
