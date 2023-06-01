import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ImageProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Digital Image")
        self.master.geometry("800x400")
        self.image = None
        self.processed_image = None

        # create widgets
        self.button_frame = tk.Frame(self.master)
        self.select_button = tk.Button(self.button_frame, text="Select", command=self.open_image)
        self.process_button = tk.Button(self.button_frame, text="Process", command=self.process_image)
        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_image)
        self.original_image_label = tk.Label(self.master)
        self.processed_image_label = tk.Label(self.master)

        # place widgets
        self.select_button.pack(side="left", padx=10, pady=10)
        self.process_button.pack(side="left", pady=10, padx=10)
        self.save_button.pack(side="left", padx=10, pady=10)
        self.button_frame.pack(side="top", pady=10, padx=10)
        self.original_image_label.pack(side="left", padx=10, pady=10)
        self.processed_image_label.pack(side="right", pady=10, padx=10)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            original_photo = ImageTk.PhotoImage(self.image)
            self.original_image_label.config(image=original_photo)
            self.original_image_label.image = original_photo
            # Gọi phương thức process_image để xử lý ảnh
            self.process_image()

    def process_image(self):
        if self.image is not None:
            # Chuyển đổi ảnh sang không gian màu Lab
            lab_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2LAB)

            # Tách kênh L* (độ sáng)
            l_channel = lab_image[:,:,0]

            # Chuyển đổi ảnh thành mảng 1D
            pixels = l_channel.reshape(-1, 1)

            # Sử dụng thuật toán K-Means để phân lớp
            num_classes = 8  # Số lớp độ sâu
            kmeans = KMeans(n_clusters=num_classes, random_state=42)
            kmeans.fit(pixels)

            # Gán nhãn cho từng điểm ảnh
            labels = kmeans.labels_

            # Chuyển đổi nhãn thành hình ảnh kết quả
            segmented_image = labels.reshape(l_channel.shape).astype(np.uint8)

            # Lưu hình ảnh đã xử lý vào biến self.processed_image
            self.processed_image = Image.fromarray(segmented_image)

            # Chuyển đổi không gian màu từ Lab sang RGB
            rgb_image = self.processed_image.convert("RGB")

            # Hiển thị ảnh kết quả
            self.display_processed_image(segmented_image)

    def display_processed_image(self, processed_image):
        # Hiển thị ảnh kết quả bằng matplotlib
        plt.imshow(processed_image, cmap='gray')
        plt.axis('off')
        plt.show()

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                # Tạo một đối tượng Figure mới
                fig = plt.figure()

                # Tạo một đối tượng Axes và hiển thị hình ảnh kết quả trên đó
                ax = fig.add_subplot(111)
                ax.imshow(self.processed_image, cmap='gray')
                ax.axis('off')

                # Lưu hình ảnh với định dạng màu chính xác
                fig.savefig(file_path, format='jpeg')

                # Đóng đối tượng Figure để giải phóng bộ nhớ
                plt.close(fig)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()