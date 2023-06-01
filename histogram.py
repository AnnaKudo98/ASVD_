import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    # Chuyển đổi ảnh sang không gian màu xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng phương pháp histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng histogram equalization
    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
    plt.show()


# Đọc ảnh
image = cv2.imread('C:/Users/user/Desktop/12.jpg')

# Áp dụng histogram equalization cho ảnh
histogram_equalization(image)

