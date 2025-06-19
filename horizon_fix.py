import os
import cv2
import numpy as np
from PIL import Image

def detect_horizon_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return 0.0  # Не найдено — не поворачиваем
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.rad2deg(theta)
        # Фильтруем горизонтальные линии (80-100 и 260-280 градусов)
        if 80 < angle < 100 or 260 < angle < 280:
            if angle > 180:
                angle -= 180
            angles.append(angle - 90)
    if len(angles) == 0:
        return 0.0
    avg_angle = np.mean(angles)
    return avg_angle

def rotate_and_crop_to_16x9(pil_img, angle):
    # 1. Резервируем запас по сторонам
    w, h = pil_img.size
    scale = 1.08 if abs(angle) > 0.2 else 1.0  # Увеличиваем только если есть наклон
    nw, nh = int(w*scale), int(h*scale)
    pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
    # 2. Поворот
    rotated = pil_img.rotate(-angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
    # 3. Кроп по центру до пропорций 16:9
    rw, rh = rotated.size
    target_ratio = 16/9
    if rw / rh > target_ratio:
        # Изображение шире — обрезаем по ширине
        new_w = int(rh * target_ratio)
        new_h = rh
    else:
        # Изображение выше — обрезаем по высоте
        new_w = rw
        new_h = int(rw / target_ratio)
    left = (rw - new_w) // 2
    top = (rh - new_h) // 2
    crop_box = (left, top, left + new_w, top + new_h)
    cropped = rotated.crop(crop_box)
    return cropped

def process_one_image(img_path, out_path):
    image_cv = cv2.imread(img_path)
    if image_cv is None:
        print(f"{os.path.basename(img_path)}: не удалось открыть файл.")
        return
    angle = detect_horizon_angle(image_cv)
    pil_img = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    if abs(angle) > 0.2:
        fixed = rotate_and_crop_to_16x9(pil_img, angle)
        fixed.save(out_path, quality=95)
        print(f"{os.path.basename(img_path)}: исправлен угол ({angle:.2f}°), сохранён.")
    else:
        # Если наклон минимальный — просто кроп под 16:9
        w, h = pil_img.size
        target_ratio = 16/9
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            new_h = h
        else:
            new_w = w
            new_h = int(w / target_ratio)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        crop_box = (left, top, left + new_w, top + new_h)
        cropped = pil_img.crop(crop_box)
        cropped.save(out_path, quality=95)
        print(f"{os.path.basename(img_path)}: почти горизонт, просто кроп под 16:9.")

def process_folder(input_folder):
    output_folder = os.path.join(input_folder, "horizon_fixed")
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)
            process_one_image(in_path, out_path)
    print("Все фото обработаны! Смотри в папке 'horizon_fixed'.")

if __name__ == "__main__":
    folder = input("Введите путь к папке с фото: ").strip()
    process_folder(folder)
