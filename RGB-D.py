'''import cv2
import numpy as np

# Load RGB-D image
image = cv2.imread("D:/Anna/Project/Medical Materials/archive/train/images/0554.png", cv2.IMREAD_UNCHANGED)

# Extract depth map from RGB-D image
depth_map = image[:,:,2]

# Extract region of interest containing wound
roi_mask = np.zeros_like(depth_map, dtype=np.uint8)
roi_mask[100:200, 100:200] = 255
roi_depth_map = np.where(roi_mask, depth_map, 0)

# Compute mean depth of wound
wound_depth = np.mean(roi_depth_map[roi_depth_map != 0])

# Compute distance to wound based on camera parameters
camera_height = 1.5 # meters
camera_fov = 60 # degrees
image_height = image.shape[0] # pixels
wound_height = 0.02 # meters
distance_to_wound = (wound_height * camera_height) / (wound_depth * np.tan(camera_fov/2))

print("Distance to wound:", distance_to_wound)
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh RGB-D
image = cv2.imread('D:/Anna/Project/Medical Materials/archive/train/images/0554.png', cv2.IMREAD_UNCHANGED)

# Thêm một kênh màu giả để biểu diễn kênh màu xanh dương
depth_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
depth_map[:, :, :2] = image[:, :, :2]
depth_map[:, :, 2] = image[:, :, 2]

# Tách các kênh màu
rgb = depth_map[:, :, :3]
depth = depth_map[:, :, 2]

# Áp dụng bộ lọc Gaussian để loại bỏ nhiễu
depth = cv2.GaussianBlur(depth, (5, 5), 0)

# Tính toán độ sâu và tạo các lớp khác nhau của vết thương
thresh1 = cv2.threshold(depth, 100, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(depth, 200, 255, cv2.THRESH_BINARY)[1]
thresh3 = cv2.threshold(depth, 250, 255, cv2.THRESH_BINARY)[1]

# Hiển thị ảnh ban đầu, vết thương và độ sâu từng lớp lên cùng một hình ảnh
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

ax[0].imshow(rgb)
ax[0].set_title('RGB')

ax[1].imshow(depth_map, cmap='jet')
ax[1].set_title('Depth')

ax[2].imshow(thresh1, cmap='gray')
ax[2].set_title('Depth Layer 1')

ax[3].imshow(thresh2, cmap='gray')
ax[3].set_title('Depth Layer 2')

plt.show()
