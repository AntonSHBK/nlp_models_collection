# Text and Multimodal Model Loader

## Описание

Этот проект предоставляет универсальные классы для загрузки и использования моделей генерации текста и мультимодальных моделей, таких как **Llama**, **GIT**, **T5**, **BERT-based generators**, и **BLIP**. Классы используют библиотеку [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) и поддерживают задачи генерации текста, описания изображений, визуально-вопросно-ответные системы (VQA) и многое другое.

## Требования

- Python 3.7+
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation)
- [Torch](https://pytorch.org/get-started/locally/)
- [dotenv](https://pypi.org/project/python-dotenv/) для работы с переменными окружения

## Установка

1. Клонируйте репозиторий и перейдите в его директорию:
   ```bash
   git clone https://github.com/AntonSHBK/multimodal_nlp_models.git
   cd multimodal_nlp_models
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Создайте файл `.env` в корневой директории и добавьте токен Hugging Face:
   ```plaintext
   HF_TOKEN=your_huggingface_token_here
   ```

---

## Документация

### Описание моделей и классов

Подробное описание классов и инструкцию по их использованию можно найти в [docs/model_loaders_documentation.md](./docs/model_loaders_documentation.md).

### Сравнительный анализ мультимодальных моделей

Сравнительный анализ моделей и их применимость для различных задач доступен в файле [docs/multimodal_model.md](./docs/multimodal_model.md).

---

## Описание классов

Все классы находятся в файле `base_model.py`:

### `TextGenerationModelLoader`

Класс для загрузки любой модели генерации текста. Поддерживаемые модели:
- GPT-2, GPT-3
- BLOOM
- Meta Llama (например, `meta-llama/Llama-3.1-8B-Instruct`)
- T5 (например, `t5-base`)
- BERT-based generators (например, `google/bert2bert_L-24_wmt_de_en`)

### `ImageToTextModelLoader`

Класс для загрузки моделей, поддерживающих задачу `image-to-text` (генерация описания изображения). Поддерживаемые модели:
- Meta Llama Vision (например, `meta-llama/Llama-3.2-11B-Vision-Instruct`)
- BLIP
- GIT (например, `microsoft/git-base`)

### `VQAModelLoader`

Класс для загрузки моделей, поддерживающих визуально-вопросно-ответные задачи (VQA). Поддерживаемые модели:
- Meta Llama Vision (например, `meta-llama/Llama-3.2-11B-Vision-Instruct`)
- BLIP
- OFA