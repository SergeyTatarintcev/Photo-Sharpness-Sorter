import os
from PIL import Image, ImageEnhance
import numpy as np
import rawpy

DARK_THRESHOLD = 60
BRIGHT_THRESHOLD = 210

def open_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        image = Image.open(path).convert('RGB')
        raw_mode = False
    elif ext in ['.nef', '.cr2', '.arw']:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(output_bps=8)
            image = Image.fromarray(rgb)
        raw_mode = True
    else:
        image = None
        raw_mode = None
    return image, raw_mode

def auto_enhance(image):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    brightness = np.array(image).mean()
    if brightness < 90:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.4)
    elif brightness > 180:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.85)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.12)
    arr = np.asarray(image).astype(np.float32)
    avgR = np.mean(arr[:,:,0])
    avgG = np.mean(arr[:,:,1])
    avgB = np.mean(arr[:,:,2])
    avg = (avgR + avgG + avgB) / 3
    arr[:,:,0] *= (avg / avgR)
    arr[:,:,1] *= (avg / avgG)
    arr[:,:,2] *= (avg / avgB)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    return image

def check_bad_shot(image):
    brightness = np.array(image).mean()
    if brightness < DARK_THRESHOLD:
        return 'dark'
    elif brightness > BRIGHT_THRESHOLD:
        return 'bright'
    return None

input_folder = input('Путь к корневой папке с фото: ').strip()
folder_jpg = os.path.join(input_folder, "цветокор JPG")
folder_raw = os.path.join(input_folder, "цветокор RAW")
folder_bad = os.path.join(input_folder, "bad_shots")
os.makedirs(folder_jpg, exist_ok=True)
os.makedirs(folder_raw, exist_ok=True)
os.makedirs(folder_bad, exist_ok=True)

supported_ext = ('.jpg', '.jpeg', '.png', '.nef', '.cr2', '.arw')

for dirpath, dirnames, filenames in os.walk(input_folder):
    # чтобы не обрабатывать папки-результаты
    if os.path.abspath(dirpath) in map(os.path.abspath, [folder_jpg, folder_raw, folder_bad]):
        continue

    for filename in filenames:
        if filename.lower().endswith(supported_ext):
            in_path = os.path.join(dirpath, filename)
            try:
                image, is_raw = open_image(in_path)
                if image is None:
                    print(f"{filename}: не удалось открыть файл")
                    continue

                bad = check_bad_shot(image)
                basename = os.path.splitext(os.path.relpath(in_path, input_folder))[0]
                safe_basename = basename.replace(os.sep, "_")
                if bad:
                    out_path = os.path.join(folder_bad, safe_basename + ".jpg")
                    image.save(out_path, quality=95)
                    print(f"{filename}: плохой снимок ({'очень темный' if bad=='dark' else 'пересвечен'}), отправлен в bad_shots")
                    continue

                enhanced = auto_enhance(image)
                if is_raw:
                    out_path = os.path.join(folder_raw, safe_basename + ".jpg")
                    enhanced.save(out_path, quality=95)
                    print(f"{filename}: обработан RAW, сохранён как JPG в 'цветокор RAW'")
                else:
                    out_path = os.path.join(folder_jpg, safe_basename + ".jpg")
                    enhanced.save(out_path, quality=95)
                    print(f"{filename}: обработан JPG/PNG, сохранён в 'цветокор JPG'")

            except Exception as e:
                print(f"{filename}: ошибка {e}")

print('Готово! Все улучшенные фото рассортированы по папкам.')
