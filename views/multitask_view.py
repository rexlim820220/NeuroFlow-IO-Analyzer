import threading
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from views.base_view import BaseView
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MultiTaskPage(BaseView):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Multiprocess vs Multihreading")

        self.ctrl_frame = tk.Frame(self)
        self.ctrl_frame.pack(side = "top", fill = "x", pady=10)

        self.canvas_container = tk.Frame(self, bg="gray")
        self.canvas_container.pack(side="top", fill="both", expand=True, padx=20)

        self.btn_run = tk.Button(self.ctrl_frame, text="Start Experiment", command=self.start_test_thread)
        self.btn_run.pack(side="left", padx=10)

        self.status_label = tk.Label(self, text="Click button to start...")
        self.status_label.pack()

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)

    def start_test_thread(self):
        self.btn_run.config(state="disable")
        self.status_label.config(text="Start experiment, please wait...")

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        t = threading.Thread(target=self.execute_lab)
        t.daemon = True
        t.start()

    def execute_lab(self):
        try:
            import time
            import multiprocessing

            def cpu_task():
                count = 0
                for i in range(10**7): count += i
            def io_task():
                time.sleep(1)

            def run_test(mode, task_func, n=2):
                start = time.perf_counter()
                if mode == "Sequential":
                    for _ in range(n):
                        task_func()
                else:
                    workers = []
                    start = time.perf_counter()
                    for _ in range(n):
                        w = threading.Thread(target=task_func) if mode == "Threading" else multiprocessing.Process(target=task_func)
                        workers.append(w)
                        w.start()
                    for w in workers:
                        w.join()
                return time.perf_counter() - start

            modes = ["Sequential", "Threading", "Processing"]
            print("Start testing, please wait...")
            cpu_results = [run_test(m, cpu_task) for m in modes]
            io_results = [run_test(m, io_task) for m in modes]

            self.after(0, lambda: self.draw_charts(modes, cpu_results, io_results))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn_run.config(state="normal"))
            self.after(0, lambda: self.status_label.config(text="Test finished!Displaying the diagram..."))

    def draw_charts(self, modes, cpu_res, io_res):
        for widget in self.canvas_container.winfo_children():
            widget.destroy()

        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        colors = ['#95a5a6', '#3498db', '#e74c3c']

        ax1.bar(modes, cpu_res, color=colors)
        ax1.set_title("CPU bound task (calculation)")

        ax2.bar(modes, io_res, color=colors)
        ax2.set_title("I/O bound task (Sleep)")

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update_idletasks()
