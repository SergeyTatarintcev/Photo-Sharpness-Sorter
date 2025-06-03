import os
import shutil
import cv2
import rawpy
import numpy as np

def get_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif ext in ['.nef', '.cr2', '.arw']:
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(output_bps=8)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            print(f"{os.path.basename(path)}: Не удалось прочитать RAW-файл: {e}")
            image = None
    else:
        image = None
    return image

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Папки для сортировки
FOLDERS = [
    ("на удаление меньше 75", 0, 75),
    ("на разбор 75-150", 75, 150),
    ("на разбор 150-450", 150, 450),
    ("отлично более 450", 450, float('inf')),
]

folder = input("Введите путь к папке с фотографиями: ").strip()

# Создание папок для сортировки
dest_paths = {}
for name, _, _ in FOLDERS:
    dest = os.path.join(folder, name)
    if not os.path.exists(dest):
        os.makedirs(dest)
    dest_paths[name] = dest

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".nef", ".cr2", ".arw")):
        file_path = os.path.join(folder, filename)
        image = get_image(file_path)
        if image is None:
            print(f"{filename}: Не удалось прочитать файл")
            continue
        score = variance_of_laplacian(image)
        found = False
        for name, min_val, max_val in FOLDERS:
            if min_val <= score < max_val:
                dst_path = os.path.join(dest_paths[name], filename)
                shutil.move(file_path, dst_path)
                print(f"{filename}: {name} (резкость: {score:.2f}) — перемещён")
                found = True
                break
        if not found:
            print(f"{filename}: Не попал ни в одну категорию (резкость: {score:.2f})")

print("\nСортировка завершена!")
