import cv2
import numpy as np
from PIL import Image, ImageTk
from core.line_gap_detector import LineGapDetector

class GlueTrackDetector:

    def __init__(self, filename):
        self.line_gap_detector = LineGapDetector(
            min_area=20,
            filename=filename,
            debug=self._debug
        )

    def detect(self, gray, original, debug_callback=None):

        print(f"{gray.size} {original.size}")

        clean_binary = self._preprocess(gray, debug_callback)

        gray, inner_contour = self._find_inner_contour(
            clean_binary, gray, debug_callback
        )

        if inner_contour is None:
            return gray, "NG (找不到內圍)"

        ring_binary = self._build_ring_mask(
            gray, inner_contour, expand_distance=30, show=debug_callback
        )

        final_display_overflow, result_overflow = self.detect_glue_overflow(
            gray, ring_binary, debug_callback
        )

        final_display_gap, result_gap = self.line_gap_detector.detect(
            gray, ring_binary, debug_callback
        )

        result_texts = []

        if result_gap > 0:
            result_texts.append(f"偵測到 {result_gap} 個斷點")
        if result_overflow > 0:
            result_texts.append(f"偵測到 {result_overflow} 個潛在溢膠點")

        if not result_texts:
            result_texts.append("PASS (膠軌連續)")

        result_text = " | ".join(result_texts)

        final_display = cv2.addWeighted(final_display_gap, 0.5, final_display_overflow, 0.5, 0)

        h, w = final_display.shape[:2]

        (size1, _) = cv2.getTextSize(f"gap count: {result_gap}", cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
        (size2, _) = cv2.getTextSize(f"overflow count: {result_overflow}", cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)

        x1 = (w - size1[0]) // 2
        x2 = (w - size2[0]) // 2

        y1 = h // 2
        y2 = y1 + size1[1] + 20

        cv2.putText(final_display, f"gap count: {result_gap}", (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(final_display, f"overflow count: {result_overflow}", (x2, y2),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 100, 255), 2, cv2.LINE_AA)

        self._debug(debug_callback, final_display, "17 Final Result")

        return final_display, result_text

    # -----------------------------
    # 1 preprocess
    # -----------------------------

    def _preprocess(self, gray, show):

        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        self._debug(show, blurred, "1 GaussianBlur")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        self._debug(show, gradient, "2 Gradient")

        _, binary = cv2.threshold(
            gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

        self._debug(show, clean, "3 Clean Binary")

        return clean

    # -----------------------------
    # 2 find contour
    # -----------------------------

    def _find_inner_contour(self, binary, gray, show):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return gray, None

        TARGET_MIN_AREA = 1400000
        TARGET_MAX_AREA = 1500000

        area_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if TARGET_MIN_AREA <= area <= TARGET_MAX_AREA:
                area_candidates.append((area, c))

        area_candidates.sort(key=lambda x: x[0], reverse=True)

        final_inner = None

        if area_candidates:
            final_inner = area_candidates[0][1]
            print(f"Target found by area range. Area: {area_candidates[0][0]}")
        else:
            print("Warning: No contour found in target area range. Using fallback logic.")
            areas = [cv2.contourArea(c) for c in contours]
            sorted_idx = np.argsort(areas)[::-1]
            outer = contours[sorted_idx[0]]
            outer_rect = cv2.boundingRect(outer)

            potential = []
            for i in sorted_idx[1:]:
                c = contours[i]
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                if (outer_rect[0] < cx < outer_rect[0] + outer_rect[2] and
                    outer_rect[1] < cy < outer_rect[1] + outer_rect[3]):
                    potential.append(c)

            final_inner = potential[0] if potential else contours[sorted_idx[1]]
            print(f"Fallback Inner Area: {cv2.contourArea(final_inner)}")

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [final_inner], -1, 255, -1)
        final_mask = cv2.bitwise_not(mask)

        self._debug(show, final_mask, "4 Filled Inner")
        gray = cv2.bitwise_and(gray, gray, mask=final_mask)
        self._debug(show, gray, "5 Clean Gray")

        return gray, final_inner

    # -----------------------------
    # 3 build ring
    # -----------------------------

    def _build_ring_mask(self, gray, inner_contour, expand_distance, show):

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [inner_contour], -1, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = max(contours, key=cv2.contourArea)

        x_min = max_contour[:, :, 0].min()
        y_max = max_contour[:, :, 1].max()
        left_bottom_corner = np.array([[[x_min, y_max]]])
        new_cnt = np.vstack([max_contour, left_bottom_corner])

        hull = cv2.convexHull(new_cnt)
        hull_mask = np.zeros(gray.shape, np.uint8)

        cv2.drawContours(hull_mask, [hull], -1, 255, -1)

        def shift_image_y(kernel, offset_y, shift=False):
            img = cv2.dilate(hull_mask, kernel)
            h, w = img.shape[:2]

            offset_x = int(0.075 * offset_y)

            shifted = np.roll(img, -0.5*offset_y, axis=0)
            shifted = np.roll(shifted, -offset_x, axis=1)

            center_x, center_range = w // 2, 40
            left, right = max(0, center_x - center_range), min(center_x + center_range, w)

            if shift:
                shifted_center = np.roll(img[:h//2, left:right], -2*offset_y, axis=0)
            else:
                shifted_center = img[:h//2, left:right]

            result = shifted.copy()
            result[:h//2, left:right] = np.maximum(result[:h//2, left:right], shifted_center)

            return result

        inner_x, inner_y = int(expand_distance * 0.25), int(expand_distance * 0.35)

        kernel_inner = cv2.getStructuringElement(cv2.MORPH_RECT, (inner_x, inner_y))

        A_prime = shift_image_y(kernel_inner, 7)

        self._debug(show, A_prime, "6 Convex Hull")

        outer_x, outer_y = int(expand_distance * 1.6), int(expand_distance * 1.7)

        kernel_outer = cv2.getStructuringElement(cv2.MORPH_RECT, (outer_x, outer_y))

        B = shift_image_y(kernel_outer, 9, True)

        self._debug(show, B, "7 Dilated")

        ring_mask = cv2.subtract(B, A_prime)

        erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ring_mask = cv2.erode(ring_mask, erode)

        self._debug(show, ring_mask, "8 Ring Mask")

        ring_binary = np.full(gray.shape, 255, np.uint8)
        ring_binary[ring_mask == 255] = gray[ring_mask == 255]

        self._debug(show, ring_binary, "9 Ring Binary")

        #ring_binary = cv2.resize(ring_binary, None, fx=4, fy=4)

        return 255-ring_binary

    # -----------------------------
    # 4 glue overflow detect
    # -----------------------------

    def detect_glue_overflow(self, original_gray, ring_binary, show=True):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))

        tophat_mask = cv2.morphologyEx(ring_binary, cv2.MORPH_TOPHAT, kernel)

        _, glue_candidate  = cv2.threshold(tophat_mask, 250, 255, cv2.THRESH_BINARY)

        self._debug(show, glue_candidate, "14 Glue Candidate")

        glue_candidate = cv2.dilate(glue_candidate, kernel)

        self._debug(show, glue_candidate, "15 Dilated Candidate")

        final_overflow_count = 0
        display = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(glue_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        MIN_AREA = 0
        MIN_SOLIDITY = 0.9
        MIN_RATIO = 10

        print("================ Glue Overflow:================")

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if solidity >= MIN_SOLIDITY: continue
            print(f"{i+1}-th contour solidity : {solidity:.2f} and aspect_ratio: {aspect_ratio:.2f}")
            if aspect_ratio > MIN_RATIO: continue
            final_overflow_count += 1
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 100, 255), 2)
            cv2.putText(display, f"{i+1}", (x+w//2, y+h//-15), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 100, 255), 3, cv2.LINE_AA)

        self._debug(show, display, "16 Glue Result")

        return display, final_overflow_count

    # -----------------------------
    # debug helper
    # -----------------------------

    def _debug(self, callback, img, title, resize_back=False):

        if callback is not None:
            callback(img, title, resize_back)

class ImageUtils:

    @staticmethod
    def load_gray(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def resize_long_side(img, target=2000):
        scale = target / max(img.shape)
        return cv2.resize(img, None, fx=scale, fy=scale)

    @staticmethod
    def cv2_to_tk(img):

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(img)
        pil.thumbnail((600, 400))

        return ImageTk.PhotoImage(pil)

    @staticmethod
    def rect_kernel(x, y):
        return cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))

class DebugViewer:

    def __init__(self, view):
        self.view = view

    def show_step(self, img, title, resize_back=False):
        if resize_back:
            img = cv2.resize(img, None, fx=4, fy=4)

        tk_img = ImageUtils.cv2_to_tk(img)

        self.view.panel.configure(image=tk_img, text=title)
        self.view.panel.image = tk_img

        cv2.imwrite(f"debug_{title}.png", img)

        self.view.update_idletasks()
        self.view.update()
