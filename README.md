# Photo Sharpness Sorter

Скрипт на Python для **автоматической сортировки фотографий (JPG, PNG, RAW: NEF, CR2, ARW) по уровню резкости**.

---

## Как работает

- Фото с резкостью **< 75** — в папку **"на удаление меньше 75"**
- **75 ≤ резкость < 150** — в папку **"на разбор 75-150"**
- **150 ≤ резкость < 450** — в папку **"на разбор 150-450"**
- **450 и выше** — в папку **"отлично более 450"**

---

## Рекомендуемый рабочий процесс

1. **Сортировка** фотографий по резкости с помощью данного скрипта.
2. **Лёгкий цветокоррект**:  
    - Обработайте файлы после сортировки для базовой цветокоррекции (например, с помощью [RawTherapee](https://rawtherapee.com/) или [Darktable](https://www.darktable.org/)).
3. **Выравнивание горизонта** (можно делать вручную в той же программе или любом графическом редакторе).
4. **Экспортируйте** итоговые файлы для отправки на стоки.

---

## Установка

1. Склонируйте репозиторий или скачайте скрипт.
2. Установите зависимости:
    ```
    pip install opencv-python rawpy numpy
    ```

---

## Использование

1. Запустите скрипт:
    ```
    python sharpness_sorter.py
    ```
2. Введите путь к папке с фотографиями (рекомендуется использовать папки с путём только на латинице!).
3. Скрипт рассортирует фотографии по подпапкам в зависимости от уровня резкости.

---

## Требования

- Python 3.7+
- opencv-python
- rawpy
- numpy

---

## Примечания

- Поддерживаются форматы: JPG, JPEG, PNG, NEF, CR2, ARW.
- Для работы с RAW требуется установка библиотеки `rawpy`.
- **Важно:** путь к фото должен быть только на латинице (английскими буквами).

---

**Автор:** [github.com/SergeyTatarintcev](https://github.com/SergeyTatarintcev)

