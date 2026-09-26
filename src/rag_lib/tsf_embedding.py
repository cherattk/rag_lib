import asyncio
from sentence_transformers import SentenceTransformer


class TSF_Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:

        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    # ---------------------------------------
    #
    # ---------------------------------------
    def _get_model(self) -> SentenceTransformer:
        """
        Lazy-load and cache the SentenceTransformer embedding model instance.
        """
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ---------------------------------------
    #
    # ---------------------------------------
    async def generate_embedding(self, chunk_list: list[str]) -> list[list[float]]:
        if not chunk_list:
            return []

        model = self._get_model()

        """
            Compute vector embeddings for a list of text chunks asynchronously using a thread pool.
        """
        embeddings = await asyncio.to_thread(
            model.encode,
            inputs=chunk_list,
            convert_to_tensor=False,  # Returns a NumPy array
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Converts NumPy array to list of lists
        return embeddings.tolist()
