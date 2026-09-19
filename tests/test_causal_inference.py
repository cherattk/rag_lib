import pytest
from torch import tensor
from torch import float32
from pytest_mock import mocker
from unittest.mock import MagicMock
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

# from transformers.tokenization_utils_base import BatchEncoding


from rag_lib.causal_inference import CausalInference


@pytest.fixture
def mock_causal_model():
    model = MagicMock(spec=PreTrainedModel)

    # 2. Add the dynamic generate method hook onto the mock instance
    model.generate = MagicMock()

    # 3. Apply your fake tensor output values
    model.generate.return_value = tensor([[1.0, 2.0, 3.0]])
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizerFast)

    # 1. Create a mock for the BatchEncoding object returned by calling tokenizer(...)
    mock_batch_encoding = MagicMock()
    mock_batch_encoding.__getitem__.side_effect = lambda key: {
        "input_ids": tensor([[4, 5, 6]]),
        "attention_mask": tensor([[1, 1, 1]]),
    }[key]
    mock_batch_encoding.keys.return_value = ["input_ids", "attention_mask"]

    # 2. Mock .to() on the BatchEncoding object so it returns itself (chainable)
    mock_batch_encoding.to.return_value = mock_batch_encoding

    # Tokenizer
    tokenizer.eos_token_id = 1
    tokenizer.apply_chat_template.return_value = (
        "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
    )

    # 3. Make tokenizer(...) return this BatchEncoding mock
    tokenizer.return_value = mock_batch_encoding

    # 4. Mock decode
    tokenizer.decode.return_value = "test response"

    return tokenizer


@pytest.fixture
def mock_service(mocker):
    service = CausalInference("test-name")  # IMPORTANT Use default value to Mock
    mocker.patch.object(service, "_init_model")
    return service


# class TestFail:
#     def test_fail(self):
#         assert False, "This test failed intentionally to verify the workflow."


class TestInitialization:
    def test_init_CausalInference_empty_model_name(self):
        # 1. Arrange: Define the invalid inputs you want to test
        invalid_inputs = ["", " ", "   "]

        # 2. Act & Assert: Verify that each invalid input raises a ValueError
        for invalid_name in invalid_inputs:
            with pytest.raises(ValueError) as exc_info:
                CausalInference(model_name=invalid_name)

            # 3. Optional: Verify that the exact error message is correct
        assert "model_name cannot be empty" in str(exc_info.value)  # type: ignore

    def test_init_model(self, mock_causal_model, mock_tokenizer, mocker):

        mock_causal_model.to.return_value = mock_causal_model

        mock_causal_model_from_pretrained = mocker.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_causal_model,
        )

        mock_tokenizer_from_pretrained = mocker.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        )

        service = CausalInference(model_name="test_model_name")

        service._init_model()

        # tokenizer
        mock_tokenizer_from_pretrained.assert_called_once_with(service.model_name)

        # model
        mock_causal_model_from_pretrained.assert_called_once_with(
            service.model_name, torch_dtype=float32
        )  # float32 if cpu device

        assert service.model is mock_causal_model
        assert service.tokenizer is mock_tokenizer


class TestPrompTemplate:

    def test_generate_prompt_call_tokenizer_apply_chat_template(
        self, mock_service, mock_tokenizer
    ):

        input_prompt = "Test Message"

        # Act
        mock_service.generate_prompt(input_prompt, mock_tokenizer)

        # Assert
        expected_messages = [{"role": "user", "content": "Test Message"}]
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            expected_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


class TestConfiGenerator:
    def test_changin_model_name_reset_model_and_tokenizer(self, mock_service):

        mock_service.config_generator(model_name="new_model")
        assert mock_service.model is None
        assert mock_service.tokenizer is None

    def test_config_generator(self):

        service = CausalInference(model_name="test-model")
        # Arrange
        expected_default_config = {
            "model_name": "test-model",
            "max_length": 2048,
            "max_new_tokens": 256,
            "do_sample": False,
        }

        # Test 1
        default_config = service.config_generator()

        assert default_config == expected_default_config

        # Set All Fields at once
        new_config = service.config_generator(
            model_name="model_name",
            max_length=111,
            max_new_tokens=222,
            do_sample=False,
        )
        assert new_config == {
            "model_name": "model_name",
            "max_length": 111,
            "max_new_tokens": 222,
            "do_sample": False,
        }

        # Test 3 : set config field one at time
        service.config_generator(model_name="test_model")
        assert service.model_name == "test_model"

        service.config_generator(max_length=10)
        assert service.max_length == 10

        service.config_generator(max_new_tokens=20)
        assert service.max_new_tokens == 20

        service.config_generator(do_sample=True)
        assert service.do_sample == True


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_generate_answer_call_to_thread(
        self, mock_service, mocker, mock_causal_model, mock_tokenizer
    ) -> None:

        mock_thread = mocker.patch(
            "rag_lib.causal_inference.asyncio.to_thread",
            new_callable=mocker.AsyncMock,
        )

        # IMPORTANT: asyncio.to_thread() returns the result of _execute_inference
        mock_thread.return_value = "test answer"

        # IMPORTANT : _model and _tokenizer must be mocked HERE not in mock_service fixture
        mocker.patch.object(mock_service, "_model", mock_causal_model)
        mocker.patch.object(mock_service, "_tokenizer", mock_tokenizer)

        await mock_service.generate_answer("test Prompt")

        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_call_init_model(self, mock_service):

        await mock_service.generate_answer("test prompt")
        mock_service._init_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_call_execute_inference(
        self, mocker, mock_service, mock_causal_model, mock_tokenizer
    ):
        # IMPORTANT : _model and _tokenizer must be mocked HERE not in mock_service fixture
        mocker.patch.object(mock_service, "_model", mock_causal_model)
        mocker.patch.object(mock_service, "_tokenizer", mock_tokenizer)

        # Add mocked _execute_inference()
        mocker.patch.object(mock_service, "_execute_inference")

        await mock_service.generate_answer("test prompt")
        mock_service._execute_inference.assert_called_once_with(
            "test prompt", mock_causal_model, mock_tokenizer
        )

    @pytest.mark.asyncio
    async def test_generate_answer_empty_prompt(self, mock_service):

        result = await mock_service.generate_answer("")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_answer_invalid_model(self, mocker):
        mocker.patch.object(
            CausalInference,
            "_init_model",
            side_effect=RuntimeError("Model loading failed"),
        )
        service = CausalInference(model_name="test-name")
        result = await service.generate_answer(prompt="test prompt")

        assert result is None


class TestExecuteInference:

    def test_execute_inference_call_tokenizer(
        self, mock_service, mock_tokenizer, mock_causal_model
    ):
        expected_inference_output = mock_tokenizer.decode.return_value

        result = mock_service._execute_inference(
            "test prompt",
            mock_causal_model,
            mock_tokenizer,
        )

        mock_tokenizer.assert_called_once_with(
            "test prompt",
            return_tensors="pt",
            truncation=True,
            max_length=mock_service.max_length,
        )

        assert result == expected_inference_output

    def test_execute_inference_call_model_generate(
        self, mock_service, mock_tokenizer, mock_causal_model
    ):

        mock_service._execute_inference(
            "test prompt", mock_causal_model, mock_tokenizer
        )

        mock_causal_model.generate.assert_called_once()

    def test_execute_inference_failure(self, mock_tokenizer):
        mock_tokenizer.side_effect = RuntimeError("Tokenization failed")
        service = CausalInference(model_name="test-name")
        with pytest.raises(RuntimeError):
            service._execute_inference("test prompt", MagicMock(), mock_tokenizer)
