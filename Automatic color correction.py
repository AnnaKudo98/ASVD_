import cv2
import numpy as np
import matplotlib.pyplot as plt
def automatic_color_correction(image):
    # Chuyển đổi ảnh sang không gian màu LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Tách các kênh màu
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Áp dụng Histogram Equalization trên kênh L
    l_channel_eq = cv2.equalizeHist(l_channel)

    # Kết hợp các kênh màu đã điều chỉnh
    lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))

    # Chuyển đổi ảnh trở lại không gian màu BGR
    result_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)

    return result_image

# Đọc ảnh
image = cv2.imread('C:/Users/user/Desktop/12.jpg')

# Áp dụng phương pháp tự động điều chỉnh màu sắc
corrected_image = automatic_color_correction(image)

'''# Hiển thị ảnh gốc và ảnh đã điều chỉnh
cv2.imshow('Original Image', image)
cv2.imshow('Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Hiển thị hình ảnh trước và sau điều chỉnh trên cùng một cửa sổ
fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
axs[1].set_title('Corrected Image')
axs[1].axis('off')

plt.show()