import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import AutoModel, AutoTokenizer, Pipeline, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin


class HuggingFaceModelLoader(ABC):
    """
    Базовый класс для загрузки моделей Hugging Face с ленивой (lazy) загрузкой.
    """

    def __init__(
        self, 
        model_name: str, 
        cache_dir: Path = Path("cache_dir"), 
        device: Optional[str] = None,
        **kwargs
    ):
        if not model_name:
            raise ValueError("Не указано название модели (model_name).")

        self.model_name: str = model_name
        self.cache_dir: Path = cache_dir
        self.device: str = device or kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.params = kwargs

        self.hf_token: Optional[str] = kwargs.get("hf_token", os.getenv("HF_TOKEN"))

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ленивые (lazy) загрузки
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._pipeline: Optional[Pipeline] = None
        self._processor: Optional[ProcessorMixin] = None

    @property
    def model(self) -> PreTrainedModel:
        """Ленивая загрузка модели"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Ленивая загрузка токенизатора"""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    @property
    def pipeline(self) -> Optional[Pipeline]:
        """Ленивая загрузка пайплайна"""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    @property
    def processor(self) -> Optional[ProcessorMixin]:
        """Ленивая загрузка процессора"""
        if self._processor is None:
            self._processor = self._load_processor()
        return self._processor

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Абстрактный метод для загрузки модели."""
        pass

    @abstractmethod
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Абстрактный метод для загрузки токенизатора."""
        pass

    @abstractmethod
    def _load_pipeline(self) -> Optional[Pipeline]:
        """Абстрактный метод для загрузки пайплайна."""
        pass

    @abstractmethod
    def _load_processor(self) -> Optional[ProcessorMixin]:
        """Абстрактный метод для загрузки процессора."""
        pass
