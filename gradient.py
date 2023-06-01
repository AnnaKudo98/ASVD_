import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_depth_map(depth_map):
    # Tạo bản đồ màu gradient
    colormap = plt.get_cmap('jet')

    # Ánh xạ giá trị độ sâu sang màu tương ứng trên bản đồ màu gradient
    colored_depth_map = colormap(depth_map)

    return colored_depth_map

# Đường dẫn đến hình ảnh vết thương
image_path = "path/to/your/image.jpg"

# Độ sâu của vết thương (ví dụ)
depth_map = np.array([[0.1, 0.3, 0.5],
                      [0.2, 0.4, 0.6],
                      [0.3, 0.6, 0.9]])

# Đọc hình ảnh vết thương
image = cv2.imread(image_path)

# Chuyển đổi hình ảnh sang không gian màu RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Thay đổi kích thước độ sâu để phù hợp với kích thước hình ảnh vết thương
depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

# Thể hiện độ sâu của vết thương dưới dạng mảng nhiệt
colored_depth_map = visualize_depth_map(depth_map_resized)

# Kết hợp hình ảnh vết thương và bản đồ màu độ sâu
blended_image = cv2.addWeighted(image_rgb, 0.7, colored_depth_map, 0.3, 0)

# Hiển thị hình ảnh kết quả
plt.imshow(blended_image)
plt.axis('off')
plt.show()
