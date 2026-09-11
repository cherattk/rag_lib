import logging
import asyncio
from typing import Optional
from threading import Lock

from torch import Tensor, inference_mode, device, cuda, backends
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

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
        self._max_length = max_length
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample

        # Concurrency & Cache control
        self._lock = Lock()
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerFast] = None

        # Automatically select the fastest hardware accelerator available
        self.device = device(
            "cuda"
            if cuda.is_available()
            else "mps" if backends.mps.is_available() else "cpu"
        )

    def default_prompt(self, context: str, question: str):
        return f"question: {question} context: {context}"

    def _get_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        try:
            with self._lock:
                if self._model is None or self._tokenizer is None:
                    logger.info(f"Loading '{self._model_name}' on {self.device}...")

                    # AutoTokenizer automatically maps to T5TokenizerFast
                    # legacy=False avoids warnings on older t5-small checkouts
                    tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name, legacy=False
                    )

                    # AutoModel handles t5-small, flan-t5, or any other Seq2Seq LM
                    model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(
                        self.device
                    )

                    # Assign only after both successfully loaded
                    self._tokenizer = tokenizer
                    self._model = model

                    logger.info("Model and tokenizer loaded successfully.")

                return self._model, self._tokenizer  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    def _execute_inference(
        self, prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
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
