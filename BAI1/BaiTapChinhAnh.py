import cv2
import numpy as np
import os

# Đọc ảnh
image = cv2.imread('anh1.jpg', cv2.IMREAD_GRAYSCALE)

# Tạo thư mục lưu ảnh nếu chưa tồn tại
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. Ảnh âm tính
negative_image = 255 - image
cv2.imwrite(os.path.join(output_folder, 'negative_image.jpg'), negative_image)

# 2. Tăng độ tương phản với CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_image = clahe.apply(image)
cv2.imwrite(os.path.join(output_folder, 'contrast_image.jpg'), contrast_image)

# 3. Biến đổi log
log_image = np.uint8(np.log1p(image) / np.log1p(np.max(image)) * 255)
cv2.imwrite(os.path.join(output_folder, 'log_image.jpg'), log_image)

# 4. Cân bằng Histogram
equalized_image = cv2.equalizeHist(image)
cv2.imwrite(os.path.join(output_folder, 'equalized_image.jpg'), equalized_image)

print(f"Ảnh đã được lưu vào thư mục: {output_folder}")