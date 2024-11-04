# Model Loaders Documentation

Документация для классов загрузчиков моделей, предназначенных для генерации текста, обработки изображений и визуального вопросно-ответного моделирования. Все классы расположены в файле `base_model.py`.

## Оглавление

1. [Общий базовый класс ModelLoader](#общий-базовый-класс-modelloader)
2. [TextGenerationModelLoader](#textgenerationmodelloader)
3. [ImageToTextModelLoader](#imagetotextmodelloader)
4. [VQAModelLoader](#vqamodelloader)
5. [Примеры использования](#примеры-использования)

---

## Общий базовый класс ModelLoader

Класс `ModelLoader` является базовым для всех специфичных загрузчиков моделей. Он определяет общие методы для загрузки моделей и инициализации пайплайнов, а также поддерживает использование GPU и кэширование моделей.

### Параметры конструктора

- `model_name` (str): **Обязательный параметр**. Название модели на Hugging Face.
- `cache_dir` (str): Путь для кэширования модели (по умолчанию: `./data/cache`).
- `use_gpu` (bool): Если `True`, использует GPU при наличии (по умолчанию: `True`).
- `hf_token` (str): Токен для доступа к Hugging Face. Может быть передан в `.env` или через аргументы.

### Основные методы

- `_load_model()`: Загружает модель с учетом кэширования и устройства (CPU или GPU).
- `_load_tokenizer()`: Заглушка для загрузки токенайзера (переопределяется в производных классах).
- `_load_pipeline()`: Заглушка для загрузки пайплайна (переопределяется в производных классах).

---

## TextGenerationModelLoader

Класс `TextGenerationModelLoader` предназначен для работы с любыми моделями генерации текста, поддерживаемыми библиотекой `transformers`.

### Поддерживаемые модели

- GPT-2, GPT-3
- BERT-based generators (например, BERT2BERT)
- BLOOM
- T5 (например, google/t5-large)
- Meta Llama (например, meta-llama/Llama-3.1-8B-Instruct)

### Основные методы

- `generate_text(prompt, max_length=100)`: Генерирует текст на основе входного промпта.
  - **prompt** (str): Текстовый запрос для генерации текста.
  - **max_length** (int): Максимальная длина сгенерированного текста (по умолчанию: `100`).

### Пример использования

```python
model = TextGenerationModelLoader(model_name="meta-llama/Llama-3.1-8B-Instruct")
print(model.generate_text("Describe the capabilities of Llama models."))
```

---

## ImageToTextModelLoader

Класс `ImageToTextModelLoader` предназначен для мультимодальных задач, таких как генерация текстовых описаний для изображений.

### Поддерживаемые модели

- Meta Llama Vision (например, meta-llama/Llama-3.2-11B-Vision-Instruct)
- BLIP
- GIT

### Основные методы

- `generate_image_description(image_path)`: Генерирует текстовое описание для изображения.
  - **image_path** (str): Путь к изображению для анализа.

### Пример использования

```python
image_model = ImageToTextModelLoader(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")
print(image_model.generate_image_description("path_to_image.jpg"))
```

---

## VQAModelLoader

Класс `VQAModelLoader` предназначен для визуально-вопросно-ответного моделирования (VQA) и поддерживает задачи, где требуется ответить на текстовый вопрос по изображению.

### Поддерживаемые модели

- Meta Llama Vision (например, meta-llama/Llama-3.2-11B-Vision-Instruct)
- BLIP
- OFA

### Основные методы

- `answer_question(image_path, question)`: Отвечает на вопрос по изображению.
  - **image_path** (str): Путь к изображению для анализа.
  - **question** (str): Вопрос, на который модель должна ответить.

### Пример использования

```python
vqa_model = VQAModelLoader(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")
question = "What objects are visible in the image?"
print(vqa_model.answer_question("path_to_image.jpg", question))
```

---

## Примеры использования

```python
# Text Generation Example
text_model = TextGenerationModelLoader(model_name="meta-llama/Llama-3.1-8B-Instruct")
print(text_model.generate_text("Describe the capabilities of Llama models."))

# Image to Text Example
image_model = ImageToTextModelLoader(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")
print(image_model.generate_image_description("path_to_image.jpg"))

# Visual Question Answering Example
vqa_model = VQAModelLoader(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")
question = "What objects are visible in the image?"
print(vqa_model.answer_question("path_to_image.jpg", question))
```