import cv2
import numpy as np

def tensor_light_compensation(image):
    # Chuyển đổi hình ảnh thành ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tăng cường độ tương phản sử dụng phương pháp Equalize Histogram
    enhanced_image = cv2.equalizeHist(gray_image)

    # Tạo ảnh màu giống ảnh xám
    enhanced_image = cv2.merge((enhanced_image, enhanced_image, enhanced_image))

    return enhanced_image

def convert_tensor_to_image(tensor):
    # Chuyển đổi tensor thành ảnh
    image = tensor.astype(np.uint8)
    return image

def determine_wound_depth(segmented_image):
    # Kiểm tra nếu ảnh là hình ảnh đen trắng
    if len(segmented_image.shape) == 2:
        gray_image = segmented_image
    else:
        # Chuyển đổi ảnh phân đoạn thành ảnh xám
        gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Đếm số pixel trắng
    white_pixel_count = np.count_nonzero(gray_image == 255)

    # Tính độ nông sâu dựa trên số pixel trắng
    depth = white_pixel_count / (segmented_image.shape[0] * segmented_image.shape[1])

    return depth

def extract_features(image):
    # Áp dụng các phương pháp xử lý hình ảnh để trích xuất thông tin
    # Ví dụ: làm mờ, phát hiện biên, phân đoạn, phân loại pixel v.v.

    # Làm mờ hình ảnh
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Phát hiện biên hình ảnh
    edges = cv2.Canny(blurred_image, 100, 200)

    # Phân đoạn hình ảnh
    _, segmented_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

    # Đo diện tích vết thương
    area = np.count_nonzero(segmented_image == 255)

    # Tính độ sâu dựa trên diện tích
    depth = area / (segmented_image.shape[0] * segmented_image.shape[1])

    return depth, area

def hosvd_wound_feature_extraction(image):
    # Áp dụng Tensor Light Compensation
    enhanced_image = tensor_light_compensation(image)

    # Áp dụng HOSVD trên enhanced_image
    tensor = np.array(enhanced_image)

    # Trích xuất đặc trưng từ tensor
    depth, area = extract_features(tensor)

    return depth, area


# Đường dẫn đến hình ảnh cần xử lý
image_path = "D:/Anna/Project/Medical Materials/archive/train/images/0554.png"
#"D:/Anna/Project/Medical Materials/archive/train/images/0554.png"
#"C:/Users/user/Desktop/12.jpg"
# Đọc hình ảnh
image = cv2.imread(image_path)

# Áp dụng HOSVD để trích xuất đặc trưng sau Tensor Light Compensation
extracted_features = hosvd_wound_feature_extraction(image)

# In kết quả

print(extracted_features)

# Hiển thị ảnh đã xử lý
enhanced_image = tensor_light_compensation(image)
cv2.imshow("Processed Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

