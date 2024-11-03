from transformers import pipeline
from PIL import Image
from base_model import LlamaModelLoader

class LlamaVisionModelLoader(LlamaModelLoader):
    def __init__(self, model_id=None, cache_dir=None, use_gpu=None, hf_token=None):
        # Инициализация параметров для мультимодальной модели
        super().__init__(model_id=model_id, cache_dir=cache_dir, use_gpu=use_gpu, hf_token=hf_token)
        self.vision_pipeline = None
        self._load_vision_model()
    
    def _load_vision_model(self):
        """Загружает модель для мультимодальных запросов (изображение + текст)."""
        try:
            # Инициализация пайплайна для мультимодальных запросов
            self.vision_pipeline = pipeline(
                "vqa", model=self.model_id, device=0 if self.device == "cuda" else -1, use_auth_token=self.hf_token
            )
            print(f"Мультимодальная модель {self.model_id} успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке мультимодальной модели: {e}")
    
    def generate_image_description(self, image_path, question="What is in the image?"):
        """Генерирует описание изображения на основе заданного вопроса."""
        if not self.vision_pipeline:
            raise ValueError("Мультимодальный пайплайн не инициализирован.")
        
        # Загрузка изображения
        image = Image.open(image_path)
        
        # Генерация ответа на вопрос
        response = self.vision_pipeline(image, question)
        return response[0]["generated_text"]

class LlamaTextModelLoader(LlamaModelLoader):
    def __init__(self, model_id="meta-llama/Llama-3.2-3B", cache_dir=None, use_gpu=None, hf_token=None):
        # Передача параметров для текстовой модели
        super().__init__(model_id=model_id, cache_dir=cache_dir, use_gpu=use_gpu, hf_token=hf_token)
    
    def generate_text(self, prompt, max_length=100):
        """Генерирует текст на основе входного промпта с использованием Llama-3.2-3B."""
        return super().generate_text(prompt, max_length=max_length)