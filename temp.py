

'''###In kích thước ảnh
import cv2

# Load ảnh
img = cv2.imread('D:/Anna/Project/Medical Materials/archive/train/images/0554.png')

# Lấy kích thước của ảnh
height, width, channels = img.shape

# In kích thước của ảnh
print("Kích thước của ảnh là: {}x{}x{}".format(width, height, channels))
'''
'''
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load ảnh vết thương sâu
img = cv2.imread('D:/Anna/Project/Medical Materials/archive/train/images/0554.png')

# Chuyển đổi ảnh thành mảng 2 chiều
img_arr = img.reshape((-1, 3))

# Tạo ra mô hình KMeans với số lượng clusters là 2
kmeans = KMeans(n_clusters=6, n_init=10)

# Áp dụng KMeans vào dữ liệu ảnh
kmeans.fit(img_arr)

# Gán các nhãn cho từng điểm ảnh
labels = kmeans.labels_

# Reshape labels thành kích thước của ảnh gốc
labels = labels.reshape((img.shape[:2]))

# Tạo mask cho vùng vết thương
mask = np.zeros_like(img)
mask[labels == 0] = [255, 255, 255]

# Áp dụng mask lên ảnh gốc
result = cv2.bitwise_and(img, mask)

# Hiển thị ảnh kết quả
cv2.imshow("Result", result)
cv2.waitKey(0)
'''
'''
import cv2
import numpy as np

def calculate_wound_depth(image_path):
    # Đọc hình ảnh vết thương
    img = cv2.imread(image_path)

    # Chuyển đổi không gian màu từ BGR sang HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Xác định vùng màu đỏ (có thể điều chỉnh ngưỡng màu sắc phù hợp)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    # Tính toán tổng số điểm ảnh trong vùng màu đỏ
    red_pixels = cv2.countNonZero(mask)

    # Tính toán tỷ lệ điểm ảnh màu đỏ trên tổng số điểm ảnh
    total_pixels = img.shape[0] * img.shape[1]
    red_ratio = red_pixels / total_pixels

    # In kết quả
    print("Tổng số điểm ảnh:", total_pixels)
    print("Số điểm ảnh màu đỏ:", red_pixels)
    print("Tỷ lệ điểm ảnh màu đỏ:", red_ratio)

    # Dựa trên tỷ lệ điểm ảnh màu đỏ, ước tính độ sâu của vết thương
    if red_ratio < 0.1:
        depth = "Hướng dẫn vết thương"
    elif red_ratio >= 0.1 and red_ratio < 0.3:
        depth = "Vết thương hơi sâu"
    else:
        depth = "Vết thương sâu"

    return depth

# Đường dẫn đến hình ảnh vết thương
image_path = "C:/Users/user/Desktop/12.jpg"

# Tính toán độ sâu của vết thương
wound_depth = calculate_wound_depth(image_path)

# In kết quả độ sâu của vết thương
print("Độ sâu của vết thương:", wound_depth)
'''
'''
import cv2
import numpy as np

# Đường dẫn đến các hình ảnh của vết thương từ các góc độ khác nhau
image_paths = ["C:/Users/user/Desktop/1406.png", 'C:/Users/user/Desktop/1406 - Copy.png', 'C:/Users/user/Desktop/1406 - Copy - Copy.png']  # Thay đổi tên và đường dẫn tương ứng với hình ảnh của bạn

# Đọc các hình ảnh và chuyển đổi sang không gian màu xám
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Xác định điểm đặc trưng trên hình ảnh đầu tiên (ví dụ: sử dụng phương pháp SIFT)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(images[0], None)

# Tính toán độ sâu cho từng điểm ảnh trên các hình ảnh còn lại
depths = []
for image in images[1:]:
    # Xác định điểm đặc trưng trên hình ảnh hiện tại
    keypoints_curr, descriptors_curr = sift.detectAndCompute(image, None)

    # Ánh xạ các điểm đặc trưng từ hình ảnh đầu tiên sang hình ảnh hiện tại
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_curr, descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    src_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Tính toán độ sâu dựa trên khoảng cách giữa các điểm đặc trưng
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3, 0.99)
    depth = 1 / np.linalg.norm(F[:, 2])
    depths.append(depth)

# Tính toán độ sâu trung bình của vết thương từ các hình ảnh khác nhau
average_depth = sum(depths) / len(depths)

# In kết quả độ sâu
print("Độ sâu của vết thương:", average_depth)'''
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Bước 1: Xác định điểm đặc trưng trên ảnh sử dụng phương pháp SIFT
def detect_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

left_image = cv2.imread('C:/Users/user/Desktop/1406.png')
right_image = cv2.imread('C:/Users/user/Desktop/1406 - Copy (2).png')

keypoints_left, descriptors_left = detect_keypoints(left_image)
keypoints_right, descriptors_right = detect_keypoints(right_image)

# Bước 2: So khớp điểm đặc trưng và tính toán ma trận cơ sở cấu trúc
def match_keypoints(descriptors_left, descriptors_right):
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors_left, descriptors_right)
    return matches

matches = match_keypoints(descriptors_left, descriptors_right)

# Lấy các điểm tương ứng
points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Sử dụng RANSAC để ước tính ma trận cơ sở cấu trúc F
F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC)

# Bước 3: Tính toán độ sâu từ ma trận cơ sở cấu trúc
def calculate_depth(F, points_left, points_right, image_shape):
    depth = np.zeros(image_shape[:2])
    F = F[:3, :3]  # Chỉ lấy ma trận 3x3 từ ma trận F

    for i in range(len(points_left)):
        x2, y2 = points_right[i, 0]
        x1, y1 = points_left[i, 0]
        x1_int, y1_int = int(x1), int(y1)
        if y1_int < depth.shape[0] and x1_int < depth.shape[1]:
            A = np.array([[x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]])
            if A.shape[1] != F.shape[1]:
                continue  # Bỏ qua điểm không phù hợp kích thước
            depth[y1_int, x1_int] = np.linalg.solve(F, A.T)[2, 0]
            print("Điểm ({}, {}): Độ sâu = {}".format(x1_int, y1_int, depth[y1_int, x1_int]))
    return depth


depth = calculate_depth(F, points_left, points_right, left_image.shape)

# Bước 4: Hiển thị kết quả
rows, cols = left_image.shape[:2]
X = np.arange(cols)
Y = np.arange(rows)
X, Y = np.meshgrid(X, Y)

# Hiển thị đồ thị 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vẽ các điểm
ax.plot_surface(X, Y, depth[:rows, :cols], cmap='viridis')

# Đặt tên cho các trục
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')

# Hiển thị đồ thị
plt.show()

'''

import cv2
print(cv2.__version__)


