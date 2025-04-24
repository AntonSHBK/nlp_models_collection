import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin


class BaseModelLoader(ABC):
    """
    Базовый класс для загрузки моделей Hugging Face.
    """

    def __init__(
        self, 
        model_name: str, 
        cache_dir: Path = Path("cache_dir"), 
        **kwargs
    ):
        if not model_name:
            raise ValueError("Не указано название модели (model_name).")

        self.model_name: str = model_name
        self.cache_dir: Path = cache_dir
        self.device: str = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.params = kwargs

        self.hf_token: Optional[str] = kwargs.get("hf_token", None)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._pipeline: Optional[Pipeline] = None
        self._processor: Optional[ProcessorMixin] = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    @property
    def pipeline(self) -> Optional[Pipeline]:
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    @property
    def processor(self) -> Optional[ProcessorMixin]:
        if self._processor is None:
            self._processor = self._load_processor()
        return self._processor

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        pass

    @abstractmethod
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def _load_pipeline(self) -> Optional[Pipeline]:
        pass

    @abstractmethod
    def _load_processor(self) -> Optional[ProcessorMixin]:
        pass
