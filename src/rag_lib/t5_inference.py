from torch import Tensor, inference_mode
from transformers import BatchEncoding, T5ForConditionalGeneration, T5Tokenizer
from threading import Lock
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5Inference:
    """A class for performing inference using the T5 model.

    Attributes:
        model_name (str): Name of the T5 model to use (default: "t5-small").
        _model: The loaded T5 model instance.
        _tokenizer: The loaded T5 tokenizer instance.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        max_length: int = 512,
    ) -> None:
        """Initialize the T5Inference class.

        Args:
            model_name (str): Name of the T5 model to use.
        """
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._max_length = max_length
        self._model: T5ForConditionalGeneration | None = None
        self._tokenizer: T5Tokenizer | None = None
        self._lock = Lock()

    def default_prompt(self, context: str, question: str):
        return f"question: {question} context: {context}"

    def _get_model(self) -> tuple[T5ForConditionalGeneration, T5Tokenizer]:
        logger.info("Loading model and tokenizer...")
        try:
            with self._lock:
                if self._model is None:
                    self._model = T5ForConditionalGeneration.from_pretrained(
                        self._model_name
                    )
                if self._tokenizer is None:
                    self._tokenizer = T5Tokenizer.from_pretrained(self._model_name)

                logger.info("Model and tokenizer loaded successfully.")

                return self._model, self._tokenizer  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    def _execute_inference(
        self, prompt: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer
    ) -> str | None:
        try:

            inputs: BatchEncoding = tokenizer(
                prompt,
                return_tensors="pt",  # the tokenizer returns a BatchEncoding.
                truncation=True,
                max_length=self._max_length,
            )
            with inference_mode():
                outputs: Tensor = model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=self._do_sample,
                    return_dict_in_generate=False,  # Force to return Tensor
                )

                # Single Sequence deconding
                token_ids: Tensor = outputs[0]  # 1-dimension tensor
                return tokenizer.decode(token_ids, skip_special_tokens=True)  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    async def generate_answer(self, prompt: str) -> str | None:

        if not prompt.strip():
            return None
        try:
            model, tokenizer = self._get_model()
            result = await asyncio.to_thread(
                self._execute_inference, prompt, model, tokenizer
            )
            return result if result else None

        except RuntimeError as e:
            logger.error(f"generate_answer failed : {e}")
        except Exception as e:
            logger.error(f"generate_answer failed : {e}")

        return None
