import os
from PIL import Image, ImageEnhance
import numpy as np
import rawpy

def open_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        image = Image.open(path).convert('RGB')
        raw_mode = False
    elif ext in ['.nef', '.cr2', '.arw']:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(output_bps=8, use_camera_wb=True, no_auto_bright=True)
            image = Image.fromarray(rgb)
        raw_mode = True
    else:
        image = None
        raw_mode = None
    return image, raw_mode

def soft_autocorrect(image):
    # Чуть похолодней — убавить красный, добавить синий (очень деликатно)
    arr = np.array(image).astype(np.int16)
    arr[:,:,0] = np.clip(arr[:,:,0] - 5, 0, 255)  # красный -5
    arr[:,:,2] = np.clip(arr[:,:,2] + 5, 0, 255)  # синий +5
    image = Image.fromarray(arr.astype(np.uint8))

    # Контраст +10%
    image = ImageEnhance.Contrast(image).enhance(1.10)
    # Цвет +10% (чуть больше)
    image = ImageEnhance.Color(image).enhance(1.10)
    return image

def check_bad_shot(image, dark_thr=60, bright_thr=210):
    brightness = np.array(image).mean()
    if brightness < dark_thr:
        return 'dark'
    elif brightness > bright_thr:
        return 'bright'
    return None

input_folder = input('Путь к папке с фото: ').strip()
folder_jpg = os.path.join(input_folder, "colour_JPG")
folder_raw = os.path.join(input_folder, "colour_RAW")
folder_bad = os.path.join(input_folder, "bad_shots")
os.makedirs(folder_jpg, exist_ok=True)
os.makedirs(folder_raw, exist_ok=True)
os.makedirs(folder_bad, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.nef', '.cr2', '.arw')):
        in_path = os.path.join(input_folder, filename)
        try:
            image, is_raw = open_image(in_path)
            if image is None:
                print(f"{filename}: не удалось открыть файл")
                continue

            bad = check_bad_shot(image)
            out_name = os.path.splitext(filename)[0] + ".jpg"
            if bad:
                out_path = os.path.join(folder_bad, out_name)
                image.save(out_path, quality=95)
                print(f"{filename}: плохой снимок ({'очень темный' if bad=='dark' else 'пересвечен'}), отправлен в bad_shots")
                continue

            enhanced = soft_autocorrect(image)
            if is_raw:
                out_path = os.path.join(folder_raw, out_name)
                enhanced.save(out_path, quality=95)
                print(f"{filename}: обработан RAW, сохранён как JPG в 'colour_RAW'")
            else:
                out_path = os.path.join(folder_jpg, out_name)
                enhanced.save(out_path, quality=95)
                print(f"{filename}: обработан JPG/PNG, сохранён в 'colour_JPG'")

        except Exception as e:
            print(f"{filename}: ошибка {e}")

print('Готово! Все фото обработаны и отсортированы по папкам.')
