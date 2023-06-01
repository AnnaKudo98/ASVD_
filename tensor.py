import cv2

def tensor_light_compensation(image):
    # Chuyển đổi hình ảnh thành ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tăng cường độ tương phản sử dụng phương pháp Equalize Histogram
    enhanced_image = cv2.equalizeHist(gray_image)

    # Chuyển đổi ảnh xám trở lại thành ảnh màu
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    return enhanced_image

# Đường dẫn đến hình ảnh cần xử lý
image_path = "D:/Anna/Project/Medical Materials/archive/train/images/0554.png"

# Đọc hình ảnh
image = cv2.imread(image_path)

# Áp dụng Tensor Light Compensation
enhanced_image = tensor_light_compensation(image)

# Hiển thị ảnh gốc và ảnh đã được tăng cường độ tương phản
cv2.imshow("Original Image", image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
