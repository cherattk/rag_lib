import logging
import asyncio
from typing import Optional
from threading import Lock

import torch
from torch import Tensor, inference_mode, device, cuda, backends
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalInference:
    def __init__(self, model_name: str) -> None:
        if model_name.strip() == "":
            raise ValueError("model_name cannot be empty")

        self._model_name = model_name.strip()
        self._max_length: int = (
            2048  # Raised max_length for Gemma's prompt + response context
        )
        self._max_new_tokens: int = 256
        self._do_sample = False

        self._lock = Lock()
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerFast] = None

        self.device = device(
            "cuda"
            if cuda.is_available()
            else "mps" if backends.mps.is_available() else "cpu"
        )

    @property
    def model(self) -> PreTrainedModel | None:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast | None:
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @property
    def do_sample(self) -> bool:
        return self._do_sample

    # --------------------------------------------------
    #
    # --------------------------------------------------
    def config_generator(
        self,
        model_name: str = "",
        max_length: int = 0,
        max_new_tokens: int = 0,
        do_sample: bool = False,
    ) -> dict[str, str | int | bool]:

        if model_name.strip() != "" and model_name != self._model_name:
            self._model_name = model_name
            self._model = None
            self._tokenizer = None

        if max_length > 0:
            self._max_length = max_length

        if max_new_tokens > 0:
            self._max_new_tokens = max_new_tokens

        if do_sample != self._do_sample:
            self._do_sample = do_sample

        return {
            "model_name": self._model_name,
            "max_length": self._max_length,
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
        }

    # --------------------------------------------------
    #
    # --------------------------------------------------
    def _init_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        try:
            with self._lock:
                if self._model is None or self._tokenizer is None:
                    logger.info(f"Loading '{self._model_name}' on {self.device}...")

                    tokenizer = AutoTokenizer.from_pretrained(self._model_name)

                    # Use bfloat16/float16 for optimal performance on CUDA/MPS
                    dtype = (
                        torch.bfloat16
                        if self.device.type in ["cuda", "mps"]
                        else torch.float32
                    )

                    model = AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        torch_dtype=dtype,
                    ).to(
                        self.device  # type: ignore
                    )

                    self._tokenizer = tokenizer
                    self._model = model

                    logger.info("Model and tokenizer loaded successfully.")

                return self._model, self._tokenizer  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    # --------------------------------------------------
    #
    # --------------------------------------------------
    def generate_prompt(self, prompt: str, tokenizer: PreTrainedTokenizerFast) -> str:
        # Apply chat template for Gemma
        messages = [{"role": "user", "content": prompt}]
        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return template  # type: ignore

    # --------------------------------------------------
    #
    # --------------------------------------------------
    def _execute_inference(
        self, prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
    ) -> str | None:
        try:

            inputs: BatchEncoding = tokenizer(
                prompt,  # type: ignore
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
            ).to(self.device)

            input_length = inputs["input_ids"].shape[1]

            with inference_mode():
                outputs: Tensor = model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=self._do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Slice out prompt tokens to return ONLY generated response text
                generated_tokens = outputs[0][input_length:]
                return tokenizer.decode(generated_tokens, skip_special_tokens=True)  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    # --------------------------------------------------
    #
    # --------------------------------------------------
    async def generate_answer(self, prompt: str) -> str | None:
        if not prompt.strip():
            return None
        try:
            self._init_model()
            if self._model is not None and self._tokenizer is not None:
                result = await asyncio.to_thread(
                    self._execute_inference, prompt, self._model, self._tokenizer
                )
                return result if result else None
            return None

        except Exception as e:
            logger.error(f"generate_answer failed : {e}")

        return None
