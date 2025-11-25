import cv2
import numpy as np

# Загружаем изображение
image_path = 'car.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

# Ресайз к фиксированной ширине для стабильности (сохраняем пропорции)
width = 800  # Можно изменить
height = int(width * image.shape[0] / image.shape[1])
resized = cv2.resize(image, (width, height))
cv2.imwrite('stage_resized.jpg', resized)

# Преобразуем в grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imwrite('stage_gray.jpg', gray)

# Улучшаем контраст с CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
cv2.imwrite('stage_contrast.jpg', enhanced)

# Bilateral filter для снижения шума, сохраняя края
blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
cv2.imwrite('stage_denoised.jpg', blurred)

# Адаптивный threshold для бинаризации (лучше для разного освещения)
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=11, C=2)
cv2.imwrite('stage_binary.jpg', binary)

# Детекция краёв Canny (с авто-порогами)
edges = cv2.Canny(binary, 50, 150)
cv2.imwrite('stage_edges.jpg', edges)

# Находим контуры
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Фильтруем контуры: крупные (area > 1% от изображения) и в нижней половине
img_area = height * width
candidates = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 0.01 * img_area:  # Минимум 1% площади
        # Центроид контура
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cy = int(M['m01'] / M['m00'])
            if cy > height * 0.5:  # Нижняя половина
                candidates.append((area, cnt))

# Сортируем по площади (самые крупные) и берём топ-2 (для 2 колёс)
candidates.sort(reverse=True, key=lambda x: x[0])
selected = [cnt for _, cnt in candidates[:2]]

# Рисуем эллипсы и центры
output = resized.copy()
for cnt in selected:
    if len(cnt) >= 5:  # Минимум для fitEllipse
        ellipse = cv2.fitEllipse(cnt)
        # Рисуем эллипс (зелёный)
        cv2.ellipse(output, ellipse, (0, 255, 0), 4)
        # Центр (красный)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        cv2.circle(output, center, 2, (0, 0, 255), 3)

# Сохраняем результат
cv2.imwrite('car_with_detected_wheels.jpg', output)
print("Колёса выделены и сохранены как 'car_with_detected_wheels.jpg'")
print("Промежуточные этапы сохранены как stage_*.jpg")