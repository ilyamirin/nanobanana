# nanobanana

Инструменты для генерации изображений с помощью модели `google/nano-banana` через Replicate API. Первая цель — настроить клиента и проверить подключение.

## Быстрый старт

1) Установите зависимости:
```
pip install -r requirements.txt
```

2) Скопируйте пример переменных окружения и укажите ваш токен Replicate:
```
cp .env.example .env
# Откройте .env и вставьте свой токен в REPLICATE_API_TOKEN
```

3) Проверьте подключение к Replicate и доступ к модели `google/nano-banana`:
```
python scripts/check_replicate_connection.py
```
При успешном подключении увидите что-то вроде:
```
Connection OK ✅
Model: google/nano-banana
Latest version: <версия>
```

## Конфигурация
- Токен хранится в переменной окружения `REPLICATE_API_TOKEN` (подгружается из файла `.env` через `python-dotenv`).
- Файл `.env` добавлен в `.gitignore`, чтобы не коммитить секреты. Пример находится в `.env.example`.

## Структура
- `nanobanana_app/config.py` — загрузка конфигурации/токена.
- `nanobanana_app/replicate_client.py` — инициализация клиента Replicate и доступ к модели `google/nano-banana`.
- `scripts/check_replicate_connection.py` — скрипт проверки подключения.

## Дальше
Следующим шагом будет добавление интерфейса на Streamlit для генерации изображений с помощью `nano-banana`, используя уже настроенный клиент Replicate.

## Streamlit UI

Интерфейс Streamlit позволяет настраивать ВСЕ доступные параметры модели `google/nano-banana`:
- Параметры автоматически подтягиваются из OpenAPI-схемы модели (если доступна).
- Для любых дополнительных/неизвестных параметров есть редактор JSON.
- Результаты генерации сохраняются в папку `outputs/` и одновременно отображаются в UI.

Запуск:
```
streamlit run streamlit_app.py
```

Требования:
- Установлены зависимости из `requirements.txt`.
- Переменная `REPLICATE_API_TOKEN` задана (через `.env` или окружение).

Поведение и советы:
- Поля `prompt`/`text` выводятся первыми для удобства.
- Если API модели не раскрывает схему входов, используйте блок "JSON‑параметры".
- Выходные изображения скачиваются в `outputs/` (эта папка добавлена в `.gitignore`).
