import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_color_matrix(color):
    color_matrix = np.array(color)
    for i in range(len(color_matrix)):
        if np.linalg.norm(color_matrix[i]) != 0:
            color_matrix[i] /= np.linalg.norm(color_matrix[i])
        else:
            color_matrix[i] = np.zeros(3)
    return color_matrix

def color_deconvolution(image, color_matrix_list):
    channels = cv2.split(image)
    height, width = channels[0].shape[:2]
    resized_color_matrix = []
    for color_matrix in color_matrix_list:
        resized_matrix = cv2.resize(color_matrix, (3, 2))
        resized_color_matrix.append(resized_matrix.astype(np.float32))
    result_channels = []
    for i in range(len(channels)):
        if i < len(resized_color_matrix):
            transformed_channel = cv2.warpAffine(channels[i], resized_color_matrix[i], (width, height))
            result_channels.append(transformed_channel)
        else:
            result_channels.append(channels[i])
    result_image = cv2.merge(result_channels)
    return result_image

def visualize_depth_map(depth_map):
    colormap = plt.get_cmap('jet')
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    colored_depth_map = colormap(normalized_depth_map)
    colored_depth_map_rgb = (colored_depth_map[:, :, :3] * 255).astype(np.uint8)
    return colored_depth_map_rgb

# Đường dẫn đến hình ảnh vết thương
image_path = "C:/Users/user/Desktop/12.jpg"

# Đọc hình ảnh vết thương
image = cv2.imread(image_path)

# Ma trận chuyển đổi cho từng lớp vết thương
hematoxylin = np.array([[0.650, 0.704, 0.286]])
eosin = np.array([[0.072, 0.990, 0.105]])
dab = np.array([[0.268, 0.570, 0.776]])

hematoxylin_matrix = calculate_color_matrix(hematoxylin)
eosin_matrix = calculate_color_matrix(eosin)
dab_matrix = calculate_color_matrix(dab)

hematoxylin_image = color_deconvolution(image, [hematoxylin_matrix])
eosin_image = color_deconvolution(image, [eosin_matrix])
dab_image = color_deconvolution(image, [dab_matrix])

# Độ sâu của vết thương (ví dụ)
depth_map = np.array([[0.1, 0.3, 0.5],
                      [0.2, 0.4, 0.6],
                      [0.3, 0.6, 0.9]])

# Chuyển đổi hình ảnh vết thương sang không gian màu RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Thay đổi kích thước độ sâu để phù hợp với kích thước hình ảnh vết thương
depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

# Thể hiện độ sâu của vết thương cho từng lớp màu
hematoxylin_depth_map = depth_map_resized * hematoxylin_image
eosin_depth_map = depth_map_resized * eosin_image
dab_depth_map = depth_map_resized * dab_image

# Áp dụng bản đồ màu nhiệt cho từng lớp màu và độ sâu
colored_hematoxylin_depth_map = visualize_depth_map(hematoxylin_depth_map)
colored_eosin_depth_map = visualize_depth_map(eosin_depth_map)
colored_dab_depth_map = visualize_depth_map(dab_depth_map)

# Kết hợp hình ảnh vết thương và bản đồ màu độ sâu
blended_hematoxylin_image = cv2.addWeighted(image_rgb, 0.7, colored_hematoxylin_depth_map, 0.3, 0)
blended_eosin_image = cv2.addWeighted(image_rgb, 0.7, colored_eosin_depth_map, 0.3, 0)
blended_dab_image = cv2.addWeighted(image_rgb, 0.7, colored_dab_depth_map, 0.3, 0)

# Hiển thị kết quả
plt.subplot(2, 2, 1), plt.imshow(image_rgb), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(blended_hematoxylin_image), plt.title('Hematoxylin Depth Map')
plt.subplot(2, 2, 3), plt.imshow(blended_eosin_image), plt.title('Eosin Depth Map')
plt.subplot(2, 2, 4), plt.imshow(blended_dab_image), plt.title('DAB Depth Map')
plt.show()
