import os
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoTokenizer, AutoModel, pipeline
import torch


load_dotenv()


class ModelLoader(ABC):
    """
    Абстрактный базовый класс для загрузки моделей с Hugging Face.
    """
    def __init__(self, model_name, cache_dir="./data/cache", use_gpu=True, hf_token=None):
        if not model_name:
            raise ValueError("Не указано название модели model_name")

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.use_gpu = use_gpu

        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("Необходимо указать токен доступа Hugging Face (hf_token).")

        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Загружает модель."""
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, use_auth_token=self.hf_token
            ).to(self.device)
            print(f"Модель {self.model_name} загружена на устройство {self.device}")
            self._load_tokenizer()
            self._load_pipeline()
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели {self.model_name}: {e}")

    @abstractmethod
    def _load_tokenizer(self):
        """Абстрактный метод для загрузки токенайзера."""
        pass

    @abstractmethod
    def _load_pipeline(self):
        """Абстрактный метод для загрузки пайплайна."""
        pass


class TextGenerationModelLoader(ModelLoader):
    """
    Класс для загрузки любой модели генерации текста, совместимой с Hugging Face Transformers.

    Поддерживаемые модели:
    - GPT-2, GPT-3
    - BERT-based generators (например, BERT2BERT)
    - BLOOM
    - T5 (например, google/t5-large)
    - Meta Llama (например, meta-llama/Llama-3.1-8B-Instruct)
    - И другие модели, поддерживающие задачу text-generation.
    
    Параметры:
    - model_name (str): Название модели на Hugging Face (например, "meta-llama/Llama-3.1-8B-Instruct").
    - cache_dir (str): Путь для кэширования модели.
    - use_gpu (bool): Если True, использует GPU при наличии.
    - hf_token (str): Токен для доступа к Hugging Face.
    """
    def _load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, use_auth_token=self.hf_token
            )
            print(f"Токенайзер для {self.model_name} успешно загружен.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке токенайзера: {e}")

    def _load_pipeline(self):
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            print(f"Пайплайн для {self.model_name} успешно инициализирован.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации пайплайна: {e}")

    def generate_text(self, prompt, max_length=100):
        if not self.pipeline:
            raise ValueError("Пайплайн не инициализирован.")
        output = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return output[0]["generated_text"]


class ImageToTextModelLoader(ModelLoader):
    """
    Класс для загрузки моделей, поддерживающих мультимодальную задачу 'image-to-text', например описание изображений.

    Поддерживаемые модели:
    - Meta Llama Vision (например, meta-llama/Llama-3.2-11B-Vision-Instruct)
    - BLIP
    - GIT
    
    Параметры:
    - model_name (str): Название модели на Hugging Face (например, "meta-llama/Llama-3.2-11B-Vision-Instruct").
    - cache_dir (str): Путь для кэширования модели.
    - use_gpu (bool): Если True, использует GPU при наличии.
    - hf_token (str): Токен для доступа к Hugging Face.
    """
    def _load_tokenizer(self):
        pass  # Не всегда требуется для image-to-text

    def _load_pipeline(self):
        try:
            self.pipeline = pipeline(
                "image-to-text",
                model=self.model,
                device=0 if self.device == "cuda" else -1
            )
            print(f"Пайплайн для {self.model_name} успешно инициализирован для image-to-text.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации пайплайна: {e}")

    def generate_image_description(self, image_path):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        image = Image.open(image_path)
        output = self.pipeline(image)
        return output[0]["generated_text"]


class VQAModelLoader(ModelLoader):
    """
    Класс для загрузки моделей, поддерживающих визуально-вопросно-ответную задачу (VQA).

    Поддерживаемые модели:
    - Meta Llama Vision (например, meta-llama/Llama-3.2-11B-Vision-Instruct)
    - BLIP
    - OFA
    
    Параметры:
    - model_name (str): Название модели на Hugging Face (например, "meta-llama/Llama-3.2-11B-Vision-Instruct").
    - cache_dir (str): Путь для кэширования модели.
    - use_gpu (bool): Если True, использует GPU при наличии.
    - hf_token (str): Токен для доступа к Hugging Face.
    """
    def _load_tokenizer(self):
        pass  # Не всегда требуется для VQA

    def _load_pipeline(self):
        try:
            self.pipeline = pipeline(
                "vqa",
                model=self.model,
                device=0 if self.device == "cuda" else -1
            )
            print(f"Пайплайн для {self.model_name} успешно инициализирован для VQA.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации пайплайна: {e}")

    def answer_question(self, image_path, question):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        image = Image.open(image_path)
        output = self.pipeline(image, question)
        return output[0]["generated_text"]

if __name__ == "__main__":
    # Пример использования TextGenerationModelLoader
    text_model = TextGenerationModelLoader(model_name="meta-llama/Llama-3.1-8B-Instruct")
    prompt = "Describe the capabilities of Llama models."
    print("Сгенерированный текст:", text_model.generate_text(prompt))

    # Пример использования ImageToTextModelLoader
    image_model = ImageToTextModelLoader(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")
    image_path = "path_to_image.jpg"
    print("Описание изображения:", image_model.generate_image_description(image_path))

    # Пример использования VQAModelLoader
    vqa_model = VQAModelLoader(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")
    question = "What objects are visible in the image?"
    print("Ответ на вопрос по изображению:", vqa_model.answer_question(image_path, question))
