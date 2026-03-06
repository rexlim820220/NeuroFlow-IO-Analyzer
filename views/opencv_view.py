import cv2
import tkinter as tk
from PIL import Image, ImageTk
from views.base_view import BaseView
from tkinter import filedialog, messagebox

class OpenCVPage(BaseView):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "OpenCV processing")

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="1. Read and Open image", command=self.process_open_image).pack(side='top', padx=5)
        tk.Button(btn_frame, text="2. Apply grayscale filter", command=self.process_to_gray).pack(side='left', padx=5)
        tk.Button(btn_frame, text="3. Apply Gaussian blur", command=self.process_to_blur).pack(side='left', padx=5)
        tk.Button(btn_frame, text="4. Apply Canny process", command=self.process_to_canny).pack(side='bottom', padx=5)

        self.img_container = tk.Frame(self, bg="#333")
        self.img_container.pack(fill="both", expand=True, padx=20, pady=20)

        self.panel = tk.Label(self.img_container, text="Image is not loaded yet", bg="#333", fg="white")
        self.panel.pack(expand=True)

        self.current_cv_image = None

    def cv2_to_tkinter(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        pil_img.thumbnail((600, 400))
        return ImageTk.PhotoImage(pil_img)

    def process_open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path: return

        self.current_cv_image = cv2.imread(path)
        if self.current_cv_image is None:
            messagebox.showerror("Error", "Fail to load Image")
            return

        rgb_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        pil_img.thumbnail((500, 500))

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.panel.configure(image=self.tk_img, text="")
        self.panel.image = self.tk_img

    def process_to_gray(self):
        if self.tk_img is None:
            messagebox.showwarning("Warning", "Please read and process grayscale image")
            return

        gray_img = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2GRAY)
        display_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

        tk_img = self.cv2_to_tkinter(display_img)
        self.panel.configure(image=tk_img, text="")
        self.panel.image = tk_img

    def process_to_blur(self):
        if self.current_cv_image is None:
            messagebox.showwarning("Warning", "Please read and process grayscale image")
            return

        gray = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        display_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        tk_img = self.cv2_to_tkinter(display_img)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img

    def process_to_canny(self):
        if self.current_cv_image is None:
            messagebox.showwarning("Warning", "Please read and process grayscale image")
            return

        gray = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 100)

        display_img = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        tk_img = self.cv2_to_tkinter(display_img)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img
