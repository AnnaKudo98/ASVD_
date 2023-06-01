import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("ASVD")
        self.master.geometry("1000x600")
        self.image_path = None
        self.block_size = 8

        self.button_frame = tk.Frame(self.master)
        self.select_button = tk.Button(self.button_frame, text="Select", command=self.open_image)
        self.process_button = tk.Button(self.button_frame, text="Process", command=self.process_image)
        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_image)

        self.original_image_label = tk.Label(self.master)
        self.processed_image_label = tk.Label(self.master)

        self.select_button.pack(side="left", padx=10, pady=10)
        self.process_button.pack(side="left", padx=10, pady=10)
        self.save_button.pack(side="left", padx=10, pady=10)
        self.button_frame.pack(side="top", pady=10)
        self.original_image_label.pack(side="left", padx=10, pady=10)
        self.processed_image_label.pack(side="right", padx=10, pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)
            original_photo = ImageTk.PhotoImage(self.image.resize((400, 400)))
            self.original_image_label.config(image=original_photo)
            self.original_image_label.image = original_photo

    def process_image(self):
        if self.image_path:
            self.calculate_mean_std()
            img_svd = self.calculate_asvd_block()
            processed_photo = ImageTk.PhotoImage(Image.fromarray(img_svd.astype(np.uint8)).resize((400, 400)))
            self.processed_image_label.config(image=processed_photo)
            self.processed_image_label.image = processed_photo

    def save_image(self):
        if self.image_path:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                processed_image = Image.fromarray(self.calculate_asvd_block().astype(np.uint8))
                processed_image.save(file_path)

    def calculate_asvd_block(self):
        img = cv2.imread(self.image_path).astype(np.float32)
        B, G, R = cv2.split(img)

        meanB = np.mean(B)
        meanG = np.mean(G)
        meanR = np.mean(R)

        stdB = np.std(B)
        stdG = np.std(G)
        stdR = np.std(R)

        m, n = B.shape
        Ga = np.random.normal(self.mi_Ga, self.sigma_Ga, (m, n))

        B_svd = self.adaptive_svd(B, Ga, meanB, stdB)
        G_svd = self.adaptive_svd(G, Ga, meanG, stdG)
        R_svd = self.adaptive_svd(R, Ga, meanR, stdR)

        img_svd = cv2.merge((B_svd, G_svd, R_svd))
        img_svd = img_svd.astype(np.uint8)

        return img_svd

    def adaptive_svd(self, channel, Ga, mi, sigma):
        B_normalized = (channel - mi) / sigma
        u, s, vt = np.linalg.svd(B_normalized)

        xi = (mi / sigma) / 3
        s_new = xi/3 * s

        channel_svd = u @ np.diag(s_new) @ vt
        channel_svd = (channel_svd * sigma) + mi

        return channel_svd

    def calculate_mean_std(self):
        img = cv2.imread(self.image_path).astype(np.float32)
        B, G, R = cv2.split(img)

        meanB = np.mean(B)
        meanG = np.mean(G)
        meanR = np.mean(R)

        stdB = np.std(B)
        stdG = np.std(G)
        stdR = np.std(R)

        self.mi_Ga = np.mean([meanB, meanG, meanR])
        self.sigma_Ga = np.mean([stdB, stdG, stdR])

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
