import tkinter as tk
from pathlib import Path
from views.base_view import BaseView
from tkinter import filedialog, messagebox
from core.cv_processor import GlueTrackDetector, ImageUtils, DebugViewer

class OpenCVPage(BaseView):

    def __init__(self, parent, controller):
        super().__init__(parent, controller, "OpenCV processing")

        # Initialize attributes
        self.filename = None
        self.current_image = None
        self.detector = None

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="1. Read and Open image",
                  command=self.process_open_image).pack(padx=5)

        tk.Button(btn_frame, text="2. Detect glue track",
                  command=self.detect_glue_track).pack(padx=5)

        tk.Button(btn_frame, text="3.Debug edge",
                  command=self.custom_omnidirectional_edge).pack(padx=5)

        tk.Button(btn_frame, text="4.Debug Gaps",
                  command=self.debug_gaps).pack(side="bottom", padx=5)

        tk.Button(btn_frame, text="5.Debug Overflow",
                  command=self.debug_overflow).pack(side="bottom", padx=5)

        # Image container
        self.img_container = tk.Frame(self, bg="#333")
        self.img_container.pack(fill="both", expand=True, padx=20, pady=20)

        self.panel = tk.Label(self.img_container,
                              text="Image is not loaded yet",
                              bg="#333", fg="white")
        self.panel.pack(fill="both", expand=True)

        # Debug viewer
        self.debug_viewer = DebugViewer(self)


    def process_open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not path:
            return

        # Update filename and detector
        self.filename = Path(path).stem

        image = ImageUtils.load_gray(path)
        if image is None:
            messagebox.showerror("Error", "Fail to load Image")
            return

        self.current_image = ImageUtils.resize_long_side(image, 2000)

        tk_img = ImageUtils.cv2_to_tk(self.current_image)
        self.panel.configure(image=tk_img, text="")
        self.panel.image = tk_img

        # Re-initialize detector with the new filename
        self.detector = GlueTrackDetector(self.filename)

    def detect_glue_track(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load image first")
            return

        result_img, result_text = self.detector.detect(
            self.current_image,
            debug_callback=self.debug_viewer.show_step
        )

        tk_img = ImageUtils.cv2_to_tk(result_img)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img

        messagebox.showinfo("檢測結果", result_text)

        print(self.filename, result_text)

    def custom_omnidirectional_edge(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load image first")
            return

        result_img = self.detector.purify_frame_to_clean_rectangle(
            self.current_image,
            self.debug_viewer.show_step,
            1
        )

        tk_img = ImageUtils.cv2_to_tk(result_img)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img

    def debug_gaps(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load image first")
            return

        def _debug(callback, img, title):

            if callback is not None:
                callback(img, title)

        from core.line_gap_detector import LineGapDetector
        self.line_gap_detector = LineGapDetector(
            min_area=20,
            filename=self.filename,
            debug=_debug
        )

        result_img, _ = self.line_gap_detector.detect(
            self.current_image,
            255-self.current_image,
            self.debug_viewer.show_step
        )

        tk_img = ImageUtils.cv2_to_tk(result_img)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img

    def debug_overflow(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load image first")
            return

        result_img, _ = self.detector.detect_glue_overflow(
            self.current_image,
            255-self.current_image,
            self.debug_viewer.show_step,
        )

        tk_img = ImageUtils.cv2_to_tk(result_img)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img
