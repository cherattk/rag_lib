import pytest
from unittest.mock import MagicMock
from t5_inference import T5Inference
from transformers import T5ForConditionalGeneration, T5Tokenizer

# from transformers.tokenization_utils_base import BatchEncoding
from torch import tensor

from pytest_mock import mocker


@pytest.fixture
def mock_model():
    model = MagicMock(spec=T5ForConditionalGeneration)
    model.generate.return_value = tensor([[1.0, 2.0, 3.0]])
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=T5Tokenizer)

    tokenizer.return_value = {"input_ids": tensor([[4.0, 5.0, 6.0]])}

    tokenizer.decode.return_value = "test response"

    return tokenizer


@pytest.fixture
def mock_t5(mocker, mock_model, mock_tokenizer):
    service = T5Inference()  # IMPORTANT Use default value to Mock
    mocker.patch.object(
        service, "_get_model", return_value=(mock_model, mock_tokenizer)
    )
    return service


# class TestFail:
#     def test_fail(self):
#         assert False, "This test failed intentionally to verify the workflow."


# --- Tests for Initialization ---
class TestInitialization:
    def test_t5_inference_initialization(self):
        service = T5Inference()
        assert service._model is None
        assert service._tokenizer is None
        assert service._model_name == "t5-small"

    def test_t5_inference_custom_model_name(self):
        service = T5Inference(model_name="custom-model")
        assert service._model_name == "custom-model"


class TestDefaultPrompt:
    def test_default_prompt(self, mock_t5):

        question_value = "test question"
        context_value = "test context"

        # default prompt format for t5-small model
        expected_prompt = (
            "question:" + " " + question_value + " " + "context:" + " " + context_value
        )
        prompt = mock_t5.default_prompt(context_value, question_value)
        # prompt = mock_t5.default_prompt("context", "question")
        assert prompt == expected_prompt


class TestExecuteInference:

    def test_execute_inference_call_tokenizer(
        self, mock_t5, mock_tokenizer, mock_model
    ):
        expected_inference_output = mock_tokenizer.decode.return_value

        result = mock_t5._execute_inference(
            "test prompt",
            mock_model,
            mock_tokenizer,
        )

        mock_tokenizer.assert_called_once_with(
            "test prompt",
            return_tensors="pt",
            truncation=True,
            max_length=mock_t5._max_length,
        )

        assert result == expected_inference_output

    def test_execute_inference_call_model_generate(
        self, mock_t5, mock_tokenizer, mock_model
    ):

        mock_t5._execute_inference("test prompt", mock_model, mock_tokenizer)

        mock_model.generate.assert_called_once()

    def test_execute_inference_failure(self, mock_tokenizer):
        mock_tokenizer.side_effect = RuntimeError("Tokenization failed")
        service = T5Inference()
        with pytest.raises(RuntimeError):
            service._execute_inference("test prompt", MagicMock(), mock_tokenizer)


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_generate_answer_call_to_thread(
        self, mock_t5, mocker, mock_model
    ) -> None:

        mock_thread = mocker.patch(
            "t5_inference.asyncio.to_thread",
            new_callable=mocker.AsyncMock,
        )

        # IMPORTANT: asyncio.to_thread() returns the result of _execute_inference
        mock_thread.return_value = "test answer"

        await mock_t5.generate_answer("test Prompt")

        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_call_get_model(self, mock_t5):

        await mock_t5.generate_answer("test prompt")
        mock_t5._get_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_call_execute_inference(
        self, mocker, mock_t5, mock_model, mock_tokenizer
    ):

        # Add mocked _execute_inference()
        mocker.patch.object(mock_t5, "_execute_inference")

        await mock_t5.generate_answer("test prompt")
        mock_t5._execute_inference.assert_called_once_with(
            "test prompt", mock_model, mock_tokenizer
        )

    @pytest.mark.asyncio
    async def test_generate_answer_invalid_model(self, mocker):
        mocker.patch.object(
            T5Inference, "_get_model", side_effect=RuntimeError("Model loading failed")
        )
        service = T5Inference()
        result = await service.generate_answer(prompt="test prompt")

        assert result is None


class TestGetModel:
    def test_t5_inference_get_model_outputs(self, mock_tokenizer, mock_model, mocker):

        mock_from_pretrained_model = mocker.patch(
            "transformers.T5ForConditionalGeneration.from_pretrained",
            return_value=mock_model,
        )

        mock_from_pretrained_tokenizer = mocker.patch(
            "transformers.T5Tokenizer.from_pretrained",
            return_value=mock_tokenizer,
        )

        service = T5Inference()

        model, tokenizer = service._get_model()

        assert model is mock_model
        assert tokenizer is mock_tokenizer

        mock_from_pretrained_model.assert_called_once_with(service._model_name)
        mock_from_pretrained_tokenizer.assert_called_once_with(service._model_name)

    @pytest.mark.asyncio
    async def test_generate_answer_empty_prompt(self, mock_t5):
        result = await mock_t5.generate_answer("")

        assert result is None
