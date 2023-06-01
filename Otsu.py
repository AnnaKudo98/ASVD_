import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh RGB-D
image = cv2.imread('D:/Anna/Project/Medical Materials/archive/train/images/0554.png', cv2.IMREAD_UNCHANGED)

# Tách kênh depth
depth = image[:, :, 2]

# Áp dụng bộ lọc Gaussian để loại bỏ nhiễu
depth = cv2.GaussianBlur(depth, (5, 5), 0)

# Tính toán ngưỡng phân đoạn bằng phương pháp Otsu
otsu_threshold, _ = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Phân đoạn ảnh depth thành 3 lớp khác nhau
depth_layer1 = np.where(depth < otsu_threshold/2, 255, 0).astype('uint8')
depth_layer2 = np.where((depth >= otsu_threshold/2) & (depth < otsu_threshold*2/3), 255, 0).astype('uint8')
depth_layer3 = np.where(depth >= otsu_threshold*2/3, 255, 0).astype('uint8')

# Hiển thị ảnh ban đầu, vết thương và độ sâu từng lớp lên cùng một hình ảnh
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

rgb = image[:, :, :3]
ax[0].imshow(rgb[:, :, ::-1])
ax[0].set_title('RGB')

ax[1].imshow(depth, cmap='gray')
ax[1].set_title('Depth')

ax[2].imshow(depth_layer1, cmap='gray')
ax[2].set_title('Depth Layer 1')

ax[3].imshow(depth_layer2, cmap='gray')
ax[3].set_title('Depth Layer 2')

plt.show()
