from typing import Optional

import torch
from transformers import (
    AutoModelForImageClassification,
    pipeline,
    Pipeline
)

from .base_vision_model import BaseVisionModelLoader


class ImageClassificationModelLoader(BaseVisionModelLoader):
    """
    Класс для загрузки моделей классификации изображений (например, ResNet, ViT, Swin).
    """

    def _load_model(self):
        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            torch_dtype="auto",
        )
        model.to(self.device)
        model.eval()
        return model
    
    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.pop("task", "image-classification")
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=None,
            processor=self.processor,
            device=self.device,
        )

    def predict(self, image, top_k: int = 1, **kwargs):
        """
        Предсказывает логиты и вероятности. Возвращает top_k классов.
        """
        inputs = self.processor(images=image, return_tensors="pt", **kwargs).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        topk = torch.topk(probs, k=top_k, dim=-1)
        scores = topk.values.squeeze(0).cpu().tolist()
        ids = topk.indices.squeeze(0).cpu().tolist()

        id2label = self.model.config.id2label
        labels = [id2label[str(i)] if str(i) in id2label else str(i) for i in ids]

        return [
            {"label": label, "score": score, "id": idx}
            for label, score, idx in zip(labels, scores, ids)
        ]

    def predict_label(self, image, **kwargs):
        """
        Возвращает только самую вероятную метку.
        """
        preds = self.predict(image, top_k=1, **kwargs)
        return preds[0]["label"]
