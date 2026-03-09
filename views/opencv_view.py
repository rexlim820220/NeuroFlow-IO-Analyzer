import cv2
import numpy as np
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
        tk.Button(btn_frame, text="2. Detect glue track", command=self.detect_glue_track).pack(side='bottom', padx=5)

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

        self.current_cv_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.current_cv_image is None:
            messagebox.showerror("Error", "Fail to load Image")
            return

        src = self.current_cv_image
        self.scale = 2000 / max(src.shape)
        resized = cv2.resize(src, None, fx=self.scale, fy=self.scale)
        self.current_cv_image = resized

        pil_img = Image.fromarray(resized)
        pil_img.thumbnail((500, 500))

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.panel.configure(image=self.tk_img, text="")
        self.panel.image = self.tk_img

    def detect_glue_track(self, expand_distance = 30):
        if self.current_cv_image is None:
            messagebox.showwarning("Warning", "Please read and process grayscale image")
            return

        def show_step(img, title):
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((600, 400))
            tk_img = ImageTk.PhotoImage(pil_img)

            self.panel.configure(image=tk_img, text=title)
            self.panel.image = tk_img

            cv2.imwrite(f"debug_{title}.png", img)
            self.update_idletasks()
            self.update()

        gray = self.current_cv_image
        show_step(gray, "1. Input Image (Grayscale)")

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        show_step(blurred, "2. GaussianBlur")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        show_step(gradient, "3. Enhance contract")

        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)  # iterations 低，保留細節

        show_step(clean_binary, "4. clean_binary")

        contours, _ = cv2.findContours(clean_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            messagebox.showerror("Error", "找不到輪廓！")
            return

        areas = [cv2.contourArea(c) for c in contours]
        sorted_indices = np.argsort(areas)[::-1]

        if len(sorted_indices) < 2:
            messagebox.showerror("Error", "輪廓數量不足，無法找到內圍！")
            return

        outer_idx = sorted_indices[0]
        outer_contour = contours[outer_idx]
        outer_rect = cv2.boundingRect(outer_contour)

        potential_inner_contour = []
        for i in sorted_indices[1:]:
            c = contours[i]
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if(outer_rect[0] < cx < outer_rect[0] + outer_rect[2] and
               outer_rect[1] < cy < outer_rect[1] + outer_rect[3]):
                potential_inner_contour.append(c)

        if potential_inner_contour:
            inner_contour = potential_inner_contour[0]
        else:
            inner_contour = contours[sorted_indices[1]]

        A = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(A, [inner_contour], -1, 255, -1)
        show_step(A, "6. Filled Inner Area (A) - 內圍填充")
        
        a_contours, _ = cv2.findContours(A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(a_contours) == 0:
            messagebox.showerror("Error", "無法從 A 找到輪廓！")
            return
        else:
            print(f"{len(a_contours)}")

        max_a_contour = max(a_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_a_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        min_rect_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillPoly(min_rect_mask, [box], 255)
        A_prime = cv2.erode(min_rect_mask, None, iterations=1)

        show_step(A_prime, "7. eroded Rect Filled Mask")

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (expand_distance * 2 + 1, expand_distance * 2)
        )
        B = cv2.dilate(min_rect_mask, dilate_kernel)
        show_step(B, "8. Dilated Area (B)")

        ring_mask = cv2.subtract(B, A_prime)
        ring_region = cv2.bitwise_and(gradient, gradient, mask=ring_mask)
        show_step(ring_region, "9. Ring Region (AND with Gray)")

        _, ring_binary = cv2.threshold(ring_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        show_step(ring_binary, "10. Prep: Binary Ring for Detection")

        edges = cv2.Canny(ring_binary, 50, 150)
        show_step(edges, "11. Canny Edges for Hough")

        lines = cv2.HoughLinesP(
            edges,
            rho=1,                  # 距離解析度
            theta=np.pi/180,        # 角度解析度
            threshold=30,           # 最小投票數（線段強度）
            minLineLength=30,       # 最短線段長度（濾短雜訊）
            maxLineGap=15           # 兩線段最大間隙（小於此視為同一條線）
        )

        if lines is None:
            result = "PASS (未偵測到明顯線段)"
            hough_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            lines = lines.squeeze()

            hough_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(hough_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gap_threshold = 20
        num_gaps = 0

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p1 = lines[i][:2]
                p2 = lines[i][2:]
                q1 = lines[j][:2]
                q2 = lines[j][2:]

                dist1 = np.linalg.norm(p1 - q1)
                dist2 = np.linalg.norm(p1 - q2)
                dist3 = np.linalg.norm(p2 - q1)
                dist4 = np.linalg.norm(p2 - q2)
                min_dist = min(dist1, dist2, dist3, dist4)

                if min_dist > gap_threshold:
                    num_gaps += 1

        if num_gaps > 0:
            result = f"NG (偵測到 {num_gaps} 個潛在斷點)"
        else:
            result = "PASS (膠軌連續)"

        show_step(hough_display, "12. hough_display")

        final_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        num_gaps = 0
        if lines is None:
            result = "PASS (未偵測到明顯線段)"
        else:
            lines = lines.squeeze()

            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(final_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            gap_threshold = 20
            endpoints = []
            for line in lines:
                endpoints.append((line[0], line[1]))
                endpoints.append((line[2], line[3]))

            from scipy.spatial.distance import cdist

            if len(endpoints) > 0:
                dist_matrix = cdist(endpoints, endpoints)
                np.fill_diagonal(dist_matrix, np.inf)

                min_dists = np.min(dist_matrix, axis=1)
                gap_indices = np.where(min_dists > gap_threshold)[0]

                num_gaps = len(gap_indices) // 2

                box_size = 20
                for idx in gap_indices:
                    x, y = endpoints[idx]
                    cv2.rectangle(final_display, (x - box_size // 2, y - box_size // 2),
                                (x + box_size // 2, y + box_size // 2), (0, 0, 255), 2)

        show_step(final_display, "13. final_display")

        tk_img = self.cv2_to_tkinter(final_display)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img
        messagebox.showinfo("檢測結果", f"結果: {result}\n")
