from typing import Optional

from transformers import (
    AutoModel,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
    Pipeline,
    pipeline
)
from .base_model import BaseModelLoader


class BaseMultimodalModelLoader(BaseModelLoader):
    """
    Базовый класс для мультимодальных моделей (работающих с текстом и изображениями).
    """

    def _load_model(self) -> PreTrainedModel:
        model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            **self.params
        )
        model.to(self.device)
        return model

    def _load_tokenizer(self):
        return None

    def _load_processor(self) -> Optional[ProcessorMixin]:
        return AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            **self.params
        )

    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.get("task", "multimodal")
        return pipeline(
            task=task,
            model=self.model,
            processor=self.processor,
            device=0 if self.device == "cuda" else -1,
        )
