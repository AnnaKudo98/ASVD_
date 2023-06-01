import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def linear_contrast_adjustment(image, alpha, beta):
    # Áp dụng phương pháp Linear Contrast Adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Cập nhật nội dung trên trục
    ax[1].imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()


# Đọc ảnh
image = cv2.imread('C:/Users/user/Desktop/12.jpg')

# Tạo cửa sổ và trục để hiển thị ảnh và thanh trượt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Hiển thị ảnh gốc ban đầu
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

# Tạo thanh trượt cho giá trị alpha
ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_alpha = Slider(ax_alpha, 'Alpha', 0.1, 5.0, valinit=1.0)

# Tạo thanh trượt cho giá trị beta
ax_beta = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_beta = Slider(ax_beta, 'Beta', 0.0, 100.0, valinit=0.0)


# Hàm xử lý khi giá trị trượt thay đổi
def update(val):
    alpha = slider_alpha.val
    beta = slider_beta.val
    linear_contrast_adjustment(image, alpha, beta)


slider_alpha.on_changed(update)
slider_beta.on_changed(update)

plt.show()
