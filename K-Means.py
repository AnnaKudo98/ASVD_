import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Đọc ảnh vết thương sâu
image = cv2.imread("C:/Users/user/Desktop/12.jpg")

# Chuyển đổi ảnh sang không gian màu Lab
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Tách kênh L* (độ sáng)
l_channel = lab_image[:,:,0]

# Chuyển đổi ảnh thành mảng 1D
pixels = l_channel.reshape(-1, 1)

# Sử dụng thuật toán K-Means để phân lớp
num_classes = 8  # Số lớp độ sâu
kmeans = KMeans(n_clusters=num_classes, random_state=42)
kmeans.fit(pixels)

# Gán nhãn cho từng điểm ảnh
labels = kmeans.labels_

# Chuyển đổi nhãn thành hình ảnh kết quả
segmented_image = labels.reshape(l_channel.shape).astype(np.uint8)

# Hiển thị ảnh gốc và ảnh kết quả
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(segmented_image, cmap='gray')
ax[1].set_title('Segmented Image')
plt.show()
