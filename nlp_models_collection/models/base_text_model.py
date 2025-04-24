from typing import Any, Optional, List, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
    Pipeline
)
from .base_model import BaseModelLoader


class BaseTextModelLoader(BaseModelLoader):
    """
    Базовый класс для загрузки текстовых моделей Hugging Face (например, T5, BART).
    """

    def _load_model(self) -> PreTrainedModel:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )
        model.to(self.device)
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )
        return tokenizer

    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.get("task", "text2text-generation")
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def _load_processor(self) -> Optional[object]:
        return None

    def generate(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        return self._generate(input_text, **kwargs)

    def tokenize(self, text: Union[str, List[str]], **kwargs) -> Any:
        return self._tokenize(text, **kwargs)

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> Union[str, List[str]]:
        return self._decode(token_ids, **kwargs)

    def _generate(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        inputs = self._tokenize(input_text, **kwargs.get("tokenize_kwargs", {}))
        gen_kwargs = kwargs.get("generation_kwargs", {})

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        return self._decode(output_ids, **kwargs.get("decode_kwargs", {}))

    def _tokenize(self, text: Union[str, List[str]], **kwargs) -> dict:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            **kwargs
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def _decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> Union[str, List[str]]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True, **kwargs)