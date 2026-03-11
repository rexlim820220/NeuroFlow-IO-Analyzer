import tkinter as tk
from views.base_view import BaseView

class StartPage(BaseView):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Main Menu")

        self.btn_cv = tk.Button(self, text="進入OpenCV實驗", command=lambda: controller.show_frame("OpenCVPage"))
        self.btn_cv.pack(pady=10)

        self.btn_yolo = tk.Button(self, text="YOLO模型物件偵測", command=lambda: controller.show_frame("YoloPage"))
        self.btn_yolo.pack(pady=10)

        self.btn_mt = tk.Button(self, text="進入multi-thread實驗", command=lambda: controller.show_frame("MultiTaskPage"))
        self.btn_mt.pack(pady=10)

        self.btn_exit = tk.Button(self, text="離開主程式", command=self.destroy)
        self.btn_exit.pack(pady=15)
