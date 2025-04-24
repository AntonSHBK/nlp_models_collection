# Базовые классы (используются для наследования и не предназначены для прямого использования)
from .base_model import BaseModelLoader
from .base_text_model import BaseTextModelLoader
from .base_causal_text_model import BaseCausalTextModelLoader
from .base_vision_model import BaseVisionModelLoader
from .base_multimodal_model import BaseMultimodalModelLoader

# Модели для задач text-to-text (например, T5, FLAN-T5, BART)
from .text_to_text import TextToTextModelLoader

# Модели для классификации текста (например, BERT, RoBERTa, DistilBERT)
from .text_classification import TextClassificationModelLoader

# Модели для генерации текстовых эмбеддингов (например, sentence-transformers, MiniLM)
from .text_embedding import TextEmbeddingModelLoader

# Модели автогрегрессионного типа (например, GPT, DeepSeek, Mistral)
from .deepseek_model import DeepSeekModelLoader

# Модели классификации изображений (например, ViT, ResNet, Swin)
from .image_classification import ImageClassificationModelLoader

# Модели генерации подписей к изображениям (например, BLIP, GIT, Donut)
from .image_captioning import ImageCaptioningModelLoader

# Мультимодальные модели для вопрос-ответа по изображению (например, BLIP2, OFA)
from .multimodal_qa import MultimodalQAModelLoader
