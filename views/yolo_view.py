import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from views.base_view import BaseView
from core.yolo_inference import YOLOLogic

class YoloPage(BaseView):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "YOLO model demonstration")

        self.yolo = YOLOLogic()

        tk.Button(self, text="Select image and perform YOLO prediction", command=self.run_inference).pack(pady=10)

        self.res_label = tk.Label(self, text="Status: object detecting, please wait...")
        self.res_label.pack()

        self.canvas_label = tk.Label(self, bg="black")
        self.canvas_label.pack(expand=True, fill="both", padx=20, pady=20)

    def run_inference(self):
        path = filedialog.askopenfilename()
        if not path: return

        rgb_img, status = self.yolo.predict(path)

        color = "green" if "OK" in status else "red"
        self.res_label.config(text=f"prediction outcome: {status}", fg=color)

        img = Image.fromarray(rgb_img)
        img.thumbnail((600, 400))
        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas_label.config(image=self.tk_img)
