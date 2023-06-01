import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh vết thương sâu
image = cv2.imread('D:/Anna/Project/Medical Materials/archive/train/images/0554.png')

# Chuyển đổi không gian màu từ BGR sang RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Định nghĩa các giới hạn màu cho từng lớp
lower_depth1 = np.array([0, 0, 0])
upper_depth1 = np.array([100, 100, 100])
lower_depth2 = np.array([101, 101, 101])
upper_depth2 = np.array([255, 255, 255])
# Thêm các lớp khác tùy theo nhu cầu của bạn

# Tạo các mask cho từng lớp dựa trên giới hạn màu
mask_depth1 = cv2.inRange(rgb_image, lower_depth1, upper_depth1)
mask_depth2 = cv2.inRange(rgb_image, lower_depth2, upper_depth2)
# Thêm các mask cho các lớp khác tùy theo nhu cầu của bạn

# Áp dụng mask vào ảnh gốc
segmented_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_depth1)
segmented_image = cv2.bitwise_or(segmented_image, segmented_image, mask=mask_depth2)
# Thêm các phép toán bitwise_or cho các mask và lớp khác tùy theo nhu cầu của bạn

# Hiển thị ảnh gốc và ảnh đã phân lớp
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('Ảnh gốc')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Ảnh đã phân lớp')

plt.show()
