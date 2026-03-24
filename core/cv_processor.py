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

    def detect(self, gray, expand_distance=30, debug_callback=None):

        clean_binary = self._preprocess(gray, debug_callback)

        gray, inner_contour = self._find_inner_contour(
            clean_binary, gray, debug_callback
        )

        if inner_contour is None:
            return gray, "NG (找不到內圍)"

        ring_binary = self._build_ring_mask(
            gray, inner_contour, expand_distance, debug_callback
        )

        final_display_gap, result_gap = self.line_gap_detector.detect(
            gray, ring_binary, debug_callback
        )

        final_display_overflow, result_overflow = self.detect_glue_overflow(
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

        return final_display, result_text

    # -----------------------------
    # 1 preprocess
    # -----------------------------

    def _preprocess(self, gray, show):

        blurred = cv2.GaussianBlur(gray, (35, 35), 0)
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

        areas = [cv2.contourArea(c) for c in contours]
        sorted_idx = np.argsort(areas)[::-1]

        outer = contours[sorted_idx[0]]
        outer_rect = cv2.boundingRect(outer)

        potential = []

        for i in sorted_idx[1:]:

            c = contours[i]

            M = cv2.moments(c)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if (
                outer_rect[0] < cx < outer_rect[0] + outer_rect[2]
                and outer_rect[1] < cy < outer_rect[1] + outer_rect[3]
            ):
                potential.append(c)

        inner = potential[0] if potential else contours[sorted_idx[1]]

        mask = np.zeros(gray.shape, np.uint8)

        cv2.drawContours(mask, [inner], -1, 255, -1)

        final_mask = cv2.bitwise_not(mask)

        self._debug(show, final_mask, "4 Filled Inner")

        gray = cv2.bitwise_and(gray, gray, mask=final_mask)

        self._debug(show, gray, "5 Clean Gray")

        return gray, inner

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

        def shift_image_y(kernel, offset_y):
            """Shift the whole mask upward by offset_y,
            with the central band shifted by 2*offset_y."""
            img = cv2.dilate(hull_mask, kernel)
            h, w = img.shape[:2]

            shifted = np.roll(img, -0.8*offset_y, axis=0)

            center_x, center_range = w // 2, 35
            left, right = max(0, center_x - center_range), min(center_x + center_range, w)

            shifted_center = np.roll(img[:h//2, left:right], -2*offset_y, axis=0)

            result = shifted.copy()
            result[:h//2, left:right] = shifted_center

            return result

        inner_x, inner_y = int(expand_distance * 0.5), int(expand_distance * 0.7)

        kernel_inner = cv2.getStructuringElement(cv2.MORPH_RECT, (inner_x, inner_y))

        A_prime = shift_image_y(kernel_inner, 6)

        self._debug(show, A_prime, "6 Convex Hull")

        outer_x, outer_y = int(expand_distance * 1.6), int(expand_distance * 1.7)

        kernel_outer = cv2.getStructuringElement(cv2.MORPH_RECT, (outer_x, outer_y))

        B = shift_image_y(kernel_outer, 8)

        self._debug(show, B, "7 Dilated")

        ring_mask = cv2.subtract(B, A_prime)

        erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ring_mask = cv2.erode(ring_mask, erode)

        self._debug(show, ring_mask, "8 Ring Mask")

        ring_binary = np.full(gray.shape, 255, np.uint8)
        ring_binary[ring_mask == 255] = gray[ring_mask == 255]

        self._debug(show, ring_binary, "9 Ring Binary")

        return ring_binary

    # -----------------------------
    # 4 Boundary shift
    # -----------------------------
    def calc_edge_offsets(self, ring_binary, show, d=2):

        SHIFT_D = d
        _, thresh_mask = cv2.threshold(ring_binary, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self._debug(show, thresh_mask, "Step 1: Original Threshold")

        mask = thresh_mask.astype(np.float32)
        shift_u = np.roll(mask, -SHIFT_D, axis=0)
        shift_d = np.roll(mask,  SHIFT_D, axis=0)
        shift_l = np.roll(mask, -SHIFT_D, axis=1)
        shift_r = np.roll(mask,  SHIFT_D, axis=1)

        shifted_ud = np.maximum.reduce([shift_u, shift_d])
        shifted_lr = np.maximum.reduce([shift_l, shift_r])

        edge_v = shifted_ud - mask
        edge_h = shifted_lr - mask

        edge_visual = cv2.cvtColor(thresh_mask, cv2.COLOR_GRAY2BGR)
        edge_visual[edge_v == 255] = [0, 0, 255]
        edge_visual[edge_h == 255] = [255, 0, 0]
        self._debug(show, edge_visual, "Step 2: Outer(Red) and Inner(Blue) Edges")

        image = cv2.cvtColor(ring_binary, cv2.COLOR_GRAY2BGR)
        red_mask = (edge_v == 255).astype(np.uint8) * 255
        blue_mask = (edge_h == 255).astype(np.uint8) * 255

        gap_points = cv2.bitwise_and(red_mask, blue_mask)
        image[gap_points > 0] = [0, 255, 0]
        self._debug(show, image, "Step 3: Label Gaps")
        return image

    # -----------------------------
    # 5 glue overflow detect
    # -----------------------------

    def detect_glue_overflow(self, original_gray, ring_binary, show=True):

        _, ring_binary = cv2.threshold(
            255-ring_binary, 100, 255, cv2.THRESH_BINARY
        )
        erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        eroded = cv2.erode(ring_binary, erode)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))
        merged_mask = cv2.morphologyEx(
            eroded,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        self._debug(show, merged_mask, "11 Refined Mask")
        overflow_mask_uint8 = merged_mask.astype(np.uint8) * 255

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            overflow_mask_uint8.astype(np.uint8), connectivity=8
        )

        overflow_count = 0
        display = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
        h_img, w_img = display.shape[:2]

        for idx in range(1, num_labels):
            segment_mask = (labels == idx).astype(np.uint8) * 255
            ov_cnts, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for o_cnt in ov_cnts:
                area = cv2.contourArea(o_cnt)
                print(f"{area:.2f}")
                if area < 200 or area > 2000:
                    continue
                x, y, w, h = cv2.boundingRect(o_cnt)
                if x <= 0 or y <= 0 or (x + w) >= w_img or (y + h) >= h_img:
                    continue
                overflow_count += 1
                cv2.rectangle(display, (x-15, y-15), (x+w+15, y+h+15), (0, 100, 255), 3)

        self._debug(show, display, "12 Overflow Final")

        return display, overflow_count

    # -----------------------------
    # debug helper
    # -----------------------------

    def _debug(self, callback, img, title):

        if callback is not None:
            callback(img, title)

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

    def show_step(self, img, title):

        tk_img = ImageUtils.cv2_to_tk(img)

        self.view.panel.configure(image=tk_img, text=title)
        self.view.panel.image = tk_img

        cv2.imwrite(f"debug_{title}.png", img)

        self.view.update_idletasks()
        self.view.update()
