import cv2
import numpy as np
import matplotlib.pyplot as plt

def linear_contrast_adjustment(image, alpha, beta):
    adjusted_image = np.clip(alpha * image + beta, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)
    return adjusted_image

def histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    return equalized_image

def automatic_color_correction(image,output_path):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes = [clahe.apply(channel) for channel in lab_planes]
    lab_image = cv2.merge(lab_planes)
    corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, corrected_image)  # Lưu ảnh đã qua xử lý
    return corrected_image


def apply_image_processing(image, alpha, beta):
    adjusted_image = linear_contrast_adjustment(image, alpha, beta)
    equalized_image = histogram_equalization(adjusted_image)
    corrected_image = automatic_color_correction(equalized_image)
    return corrected_image

# Đọc ảnh gốc
image = cv2.imread('C:/Users/user/Desktop/1406.png')
output_path = 'C:/Users/user/Desktop/processed_image3.jpg'
# Thiết lập giá trị alpha và beta cho linear contrast adjustment
alpha = 1.5
beta = 30

# Áp dụng các phương pháp xử lý ảnh
processed_image = automatic_color_correction(image,output_path)

'''# Hiển thị ảnh gốc và ảnh đã xử lý
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Hiển thị hình ảnh trước và sau điều chỉnh trên cùng một cửa sổ
fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
axs[1].set_title('Corrected Image')
axs[1].axis('off')

plt.show()