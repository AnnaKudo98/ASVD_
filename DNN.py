import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Đọc ảnh vết thương sâu
image = cv2.imread('D:/Anna/Project/Medical Materials/archive/train/images/0554.png', cv2.IMREAD_GRAYSCALE)

# Xử lý và chuẩn hóa ảnh
image = cv2.resize(image, (224, 224))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

# Tạo mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Tải trọng số của mô hình đã được huấn luyện trước
model.load_weights('path_to_pretrained_weights.h5')

# Dự đoán độ sâu của vết thương
depth = model.predict(image)[0][0]

# In kết quả độ sâu
print("Độ sâu của vết thương là:", depth)

# Hiển thị ảnh và vết thương đã qua xử lý
plt.imshow(image[0], cmap='gray')
plt.title('Vết thương sâu')
plt.show()
