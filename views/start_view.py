import tkinter as tk
from views.base_view import BaseView

class StartPage(BaseView):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Main Menu")

        self.btn_cv = tk.Button(self, text="進入OpenCV實驗", command=lambda: controller.show_frame("OpenCVPage"))
        self.btn_cv.pack(pady=10)

