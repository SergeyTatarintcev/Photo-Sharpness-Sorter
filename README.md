# Photo Sharpness Sorter

Скрипт на Python для автоматической сортировки фотографий (JPG, PNG, RAW: NEF, CR2, ARW) по уровню резкости.

## Как работает

- Фото с резкостью < 75 — в папку **"на удаление меньше 75"**
- 75 ≤ резкость < 150 — в папку **"на разбор 75-150"**
- 150 ≤ резкость < 450 — в папку **"на разбор 150-450"**
- 450 и выше — в папку **"отлично более 450"**

## Установка

1. Склонируйте репозиторий или скачайте скрипт.
2. Установите зависимости:
    ```
    pip install opencv-python rawpy numpy
    ```

## Использование

1. Запустите скрипт:
    ```
    python sharpness_sorter.py
    ```
2. Введите путь к папке с фотографиями.
3. Скрипт рассортирует фотографии по подпапкам в зависимости от резкости.

## Требования

- Python 3.7+
- opencv-python
- rawpy
- numpy

## Примечания

- Поддерживаются форматы: JPG, JPEG, PNG, NEF, CR2, ARW.
- Для работы с RAW требуется установка библиотеки `rawpy`.
# Важно, путь к фото должен быть на латинице
---

**Автор:** [github.com/SergeyTatarintcev]  
