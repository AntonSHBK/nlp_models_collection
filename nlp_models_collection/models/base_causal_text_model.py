from typing import Union, List, Optional, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
    Pipeline
)
from .base_model import BaseModelLoader


class BaseCausalTextModelLoader(BaseModelLoader):
    """
    Базовый класс для автогрегрессионных текстовых моделей (GPT, DeepSeek, Mistral, LLaMA и т.п.)
    """

    def _load_model(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            torch_dtype = self.params.get("torch_dtype", "auto")
            # **self.params
        )
        model.to(self.device)
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            # **self.params
        )

        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        tokenizer.model_max_length = self.params.get("model_max_length", tokenizer.model_max_length)
        return tokenizer

    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.get("task", "text-generation")
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            # **self.params
        )

    def _load_processor(self) -> Optional[Any]:
        return None

    def generate(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        return self._generate(input_text, **kwargs)

    def tokenize(self, text: Union[str, List[str]], **kwargs) -> Any:
        return self._tokenize(text, **kwargs)

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> Union[str, List[str]]:
        return self._decode(token_ids, **kwargs)


    def _generate(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        tokenize_kwargs = kwargs.get("tokenize_kwargs", {})
        generation_kwargs = kwargs.get("generation_kwargs", {})
        decode_kwargs = kwargs.get("decode_kwargs", {})

        inputs = self._tokenize(input_text, **tokenize_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        return self._decode(output_ids, **decode_kwargs)

    def _tokenize(self, text: Union[str, List[str]], **kwargs) -> dict:
        padding = kwargs.pop("padding", True)
        truncation = kwargs.pop("truncation", True)

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            **kwargs
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def _decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> Union[str, List[str]]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True, **kwargs)
