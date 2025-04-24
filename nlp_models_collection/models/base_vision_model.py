from typing import Optional

from transformers import (
    AutoModelForImageClassification,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
    pipeline,
    Pipeline
)
from .base_model import BaseModelLoader


class BaseVisionModelLoader(BaseModelLoader):
    """
    Базовый класс для загрузки моделей обработки изображений Hugging Face.
    """

    def _load_model(self) -> PreTrainedModel:
        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
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
        )

    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.get("task", "image-classification")
        return pipeline(
            task=task,
            model=self.model,
            processor=self.processor,
            device=self.device,
        )
