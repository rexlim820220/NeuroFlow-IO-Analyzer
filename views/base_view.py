import tkinter as tk

class BaseView(tk.Frame):
    def __init__(self, parent, controller, title_text):
        super().__init__(parent)
        self.controller = controller
        tk.Label(self, text=title_text, font=("Arial", 16)).pack(pady=20)
        tk.Button(self, text="返回主選單", command=lambda: controller.show_frame("StartPage")).pack(side="bottom", pady=20)

