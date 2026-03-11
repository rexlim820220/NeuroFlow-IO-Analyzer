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

    def detect_glue_track(self, expand_distance=30):
        if self.current_cv_image is None:
            messagebox.showwarning("Warning", "Please read and process grayscale image")
            return

        gray = self.current_cv_image
        show_step = self._show_step

        # 1. 前處理
        clean_binary = self._preprocess_image(gray, show_step)

        # 2. 找內圍輪廓
        gray, inner_contour = self._find_inner_contour(clean_binary, gray, show_step)
        if inner_contour is None:
            return

        # 3. 建立環狀遮罩
        ring_binary = self._build_ring_mask(gray, inner_contour, expand_distance, show_step)

        # 4. 線段檢測與斷點分析
        _, final_display, result = self._detect_lines_and_gaps(gray, ring_binary, show_step)

        # 5. 顯示結果
        tk_img = self.cv2_to_tkinter(final_display)
        self.panel.configure(image=tk_img)
        self.panel.image = tk_img
        messagebox.showinfo("檢測結果", f"結果: {result}\n")

    # ---------------- 子函式 ----------------

    def _preprocess_image(self, gray, show_step):
        blurred = cv2.GaussianBlur(gray, (25, 25), 0)
        show_step(blurred, "1. GaussianBlur")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        show_step(gradient, "2. Enhance contrast")

        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        show_step(clean_binary, "3. Clean Binary")

        return clean_binary

    def _find_inner_contour(self, clean_binary, gray, show_step):
        contours, _ = cv2.findContours(clean_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            messagebox.showerror("Error", "找不到輪廓！")
            return None

        areas = [cv2.contourArea(c) for c in contours]
        sorted_indices = np.argsort(areas)[::-1]
        if len(sorted_indices) < 2:
            messagebox.showerror("Error", "輪廓數量不足，無法找到內圍！")
            return None

        outer_idx = sorted_indices[0]
        outer_contour = contours[outer_idx]
        outer_rect = cv2.boundingRect(outer_contour)

        potential_inner = []
        for i in sorted_indices[1:]:
            c = contours[i]
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            if outer_rect[0] < cx < outer_rect[0] + outer_rect[2] and \
            outer_rect[1] < cy < outer_rect[1] + outer_rect[3]:
                potential_inner.append(c)

        inner_contour = potential_inner[0] if potential_inner else contours[sorted_indices[1]]

        A = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(A, [inner_contour], -1, 255, -1)
        final_mask = cv2.bitwise_not(A)
        show_step(final_mask, "4. Filled Inner Area")
        gray = cv2.bitwise_and(gray, gray, mask=final_mask)
        show_step(gray, "5. Clean GRAY")
        return gray, inner_contour

    def _build_ring_mask(self, gray, inner_contour, expand_distance, show_step):
        """建立最小外接矩形"""
        a_contours, _ = cv2.findContours(cv2.drawContours(np.zeros(gray.shape[:2], dtype=np.uint8),
                                                        [inner_contour], -1, 255, -1),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_a_contour = max(a_contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_a_contour)
        hull_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        A_prime = cv2.dilate(hull_mask, dilate_kernel)
        show_step(A_prime, "6. minimal convex hull")

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expand_distance*2, expand_distance*3))
        B = cv2.dilate(hull_mask, dilate_kernel)
        show_step(B, "7. Dilated Area")

        ring_mask = cv2.subtract(B, A_prime)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ring_mask = cv2.erode(ring_mask, erode_kernel, iterations=1)
        # 遮罩I
        show_step(ring_mask, "8. Ring Mask")
        # 反轉原圖
        invert = 255 - gray

        ring_region = np.full_like(invert, 0)
        ring_region[ring_mask > 0] = invert[ring_mask > 0]
        _, ring_region = cv2.threshold(ring_region, 150, 255, cv2.THRESH_BINARY)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ring_region = cv2.morphologyEx(ring_region, cv2.MORPH_OPEN, open_kernel, iterations=1)

        img_a = cv2.cvtColor(ring_region, cv2.COLOR_GRAY2BGR)
        img_a[A_prime > 0] = [255, 255, 255]
        img_a = 255 - img_a
        keep_mask = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) # 遮罩II
        show_step(keep_mask, "8. Keep Mask (with unwanted area in black)")

        new_gray = cv2.bitwise_and(gray, gray, mask=ring_mask)
        ring_mask_clean = cv2.bitwise_and(new_gray, new_gray, mask=keep_mask)

        show_step(ring_mask_clean, "9. Final Clean Ring Region (ROI)")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ring_mask_clean, connectivity=8)
        if num_labels <= 1:
            raise ValueError("圖中沒有找到任何白色區域")

        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        for i in range(2, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = i

        mask = (labels == largest_label).astype(np.uint8) * 255

        white_bg = np.full_like(ring_mask_clean, 255)

        result = white_bg.copy()
        result[mask > 0] = ring_mask_clean[mask > 0]

        ring_binary = cv2.adaptiveThreshold(
            result,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        show_step(ring_binary, "9. Ring Binary")
        return ring_binary

    def _detect_lines_and_gaps(self, gray, ring_binary, show_step):
        edges = cv2.Canny(ring_binary, 50, 150)
        show_step(edges, "10. Canny Edges")

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=15)
        hough_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        final_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = "PASS (未偵測到明顯線段)"
        if lines is not None:
            lines = lines.squeeze()
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(hough_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            endpoints = [(line[0], line[1]) for line in lines] + [(line[2], line[3]) for line in lines]
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(endpoints, endpoints)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dists = np.min(dist_matrix, axis=1)
            gap_indices = np.where(min_dists > 20)[0]
            num_gaps = len(gap_indices) // 2

            for line in lines:
                cv2.line(final_display, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            for idx in range(0 ,num_gaps):
                x, y = endpoints[idx]
                cv2.rectangle(final_display, (x-10, y-10), (x+10, y+10), (0, 0, 255), 2)

            result = f"NG (偵測到 {num_gaps} 個潛在斷點)" if num_gaps > 0 else "PASS (膠軌連續)"

        show_step(hough_display, "11. Hough Display")
        show_step(final_display, "12. Final Display")
        return hough_display, final_display, result

    def _show_step(self, img, title):
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
