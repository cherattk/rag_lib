import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sentence_transformers import SentenceTransformer
from rag_lib.tsf_embedding import TSF_Embedding


# --- Fixtures ---
@pytest.fixture
def mock_model():
    """Fixture to mock the SentenceTransformer model."""
    mock_model = MagicMock(spec=SentenceTransformer)
    np_array = np.array([[1.0, 2.0], [3.0, 4.0]])  # NumPy Array
    mock_model.encode = MagicMock(return_value=np_array)
    return mock_model


@pytest.fixture
def tsf_embedding(mock_model):
    """Fixture to create an instance of your class with a mocked model."""
    service = TSF_Embedding(model_name="all-MiniLM-L6-v2")
    with patch.object(service, "_get_model", return_value=mock_model):
        yield service


class TestEmbeddingLogic:
    @pytest.mark.asyncio
    async def test_generate_embedding_call_to_thread(
        self, tsf_embedding, mocker, mock_model
    ) -> None:

        mock_thread = mocker.patch(
            "rag_lib.tsf_embedding.asyncio.to_thread",
            new_callable=mocker.AsyncMock,
        )

        # IMPORTANT: asyncio.to_thread() returns the result of model.encode()
        mock_thread.return_value = mock_model.return_value

        await tsf_embedding.generate_embedding(["chunk_text"])

        mock_thread.assert_called_once_with(
            mock_model.encode,
            inputs=["chunk_text"],
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @pytest.mark.asyncio
    async def test_generate_embedding_call_get_model(self, tsf_embedding) -> None:

        await tsf_embedding.generate_embedding(["chunk1"])

        tsf_embedding._get_model.assert_called_once()

        # Assert that _get_model() is called without arguments
        args, kwargs = tsf_embedding._get_model.call_args

        assert len(args) == 0, f"Expected len(args) == 0"

        assert kwargs == {}, f"Expected kwargs to be empty"

    @pytest.mark.asyncio
    async def test_generate_embedding_call_model_encode(
        self, tsf_embedding, mock_model
    ) -> None:

        await tsf_embedding.generate_embedding(["chunk1"])

        mock_model.encode.assert_called_once()

        args, kwargs = mock_model.encode.call_args

        assert (
            len(args) == 0
        ), f"Expected len(args) == 0, model.encode() must be called with keyword argument only"

        assert kwargs["inputs"] == ["chunk1"], f"Expected inputs to be ['chunk1']"

        assert (
            kwargs["convert_to_tensor"] == False
        ), f"Expected convert_to_tensor to be False"

        assert (
            kwargs["normalize_embeddings"] == True
        ), f"Expected normalize_embeddings to be False"

        assert (
            kwargs["show_progress_bar"] == False
        ), f"Expected show_progress_bar to be False"


class TestEmbeddingInputOutput:
    @pytest.mark.asyncio
    async def test_generate_embedding_empty_output(self, tsf_embedding) -> None:
        output = await tsf_embedding.generate_embedding([])
        assert output == []

    @pytest.mark.asyncio
    async def test_generate_embedding_valid_output(self, tsf_embedding, mock_model):
        """
        Test that generate_embedding() returns model.encode() output as lists.
        """

        chunk_list = ["chunk1", "chunk2"]

        output = await tsf_embedding.generate_embedding(chunk_list)

        expected_embedding = mock_model.encode.return_value

        assert isinstance(output, list)
        assert output == expected_embedding.tolist()

        assert len(output) == len(chunk_list)

        assert all(isinstance(embedding, list) for embedding in output)

        assert all(
            isinstance(value, float) for embedding in output for value in embedding
        )
