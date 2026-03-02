import tkinter as tk
from views.start_view import StartPage
from views.opencv_view import OpenCVPage

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("研發整合系統 v1.0")
        self.geometry("1000x700")

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}

        for PageClass in (StartPage, OpenCVPage):
            name = PageClass.__name__
            frame = PageClass(parent=container, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        self.frames[page_name].tkraise()

