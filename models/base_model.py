import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch

# Загрузим параметры из файла .env
load_dotenv()

class LlamaModelLoader:
    def __init__(self, model_id=None, cache_dir=None, use_gpu=None, hf_token=None):
        # Загрузка параметров из .env файла, если они не указаны при инициализации
        self.model_id = model_id or os.getenv("MODEL_ID", "meta-llama/Llama-3.2-vision-instruct-11B")
        self.cache_dir = cache_dir or os.getenv("CACHE_DIR", "./cache")
        self.use_gpu = use_gpu if use_gpu is not None else os.getenv("USE_GPU", "false").lower() == "true"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")  # Токен для доступа к Hugging Face
        
        # Инициализация устройства: GPU, если доступен и разрешен, или CPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        
        # Загрузка модели и токенизатора
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Загружает модель и токенизатор с учетом кэширования и устройства."""
        try:
            # Загрузка модели и токенизатора с использованием токена
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir, use_auth_token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, cache_dir=self.cache_dir, use_auth_token=self.hf_token
            ).to(self.device)
            print(f"Модель {self.model_id} загружена на устройство {self.device}")

            # Инициализация пайплайна для текстовой генерации
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")

    def generate_text(self, prompt, max_length=100):
        """Генерирует текст на основе входного промпта."""
        if not self.pipeline:
            raise ValueError("Пайплайн не инициализирован.")
        
        # Генерация текста
        output = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return output[0]["generated_text"]
