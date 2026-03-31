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

        final_display_overflow, result_overflow = self.detect_glue_overflow(
            gray, ring_binary, debug_callback
        )

        #ring_binary = 255-self.purify_frame_to_clean_rectangle(
        #    255-ring_binary,
        #    debug_callback,
        #    1
        #)

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

        inner_x, inner_y = int(expand_distance * 0.4), int(expand_distance * 0.45)

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

        return 255-ring_binary

    # -----------------------------
    # 4 Boundary shift
    # -----------------------------
    def purify_frame_to_clean_rectangle(self, edge_image, show, d=3):
        if len(edge_image.shape) == 3:
            binary = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
        else:
            binary = edge_image.copy()

        mask = binary.astype(np.float32)

        shift_u = np.roll(mask, -d, axis=0)
        shift_d = np.roll(mask,  d, axis=0)
        shift_l = np.roll(mask, -d, axis=1)
        shift_r = np.roll(mask,  d, axis=1)

        expanded = np.maximum.reduce([shift_u, shift_d, shift_l, shift_r])
        stable_edge = np.minimum(expanded, mask)

        stable_edge = (stable_edge > 0).astype(np.uint8) * 255

        edge_up_down = stable_edge.copy()
        edge_left_right = stable_edge.copy()

        h, w = binary.shape
        clean_mask = np.zeros((h, w), dtype=np.uint8)

        def keep_longest_lines(edge_map, is_horizontal=True, top_n=3):
            edge_uint8 = (edge_map > 0).astype(np.uint8) * 255

            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                edge_uint8, connectivity=8
            )

            print(f"Debug { 'horizontal' if is_horizontal else 'vertical' }: 找到 {num} 個連通體")

            candidates = []
            for i in range(1, num):
                _, _, ww, hh = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                if is_horizontal:
                    length = ww
                else:
                    length = hh
                candidates.append((length, i))

            candidates.sort(reverse=True)
            res = np.zeros_like(edge_uint8, dtype=np.uint8)

            for length, label in candidates[:top_n]:
                print(f"Debug: 保留線段 label={label}, length={length}")
                res[labels == label] = 255

            return res

        print('horizontal: ')
        clean_mask = cv2.bitwise_or(clean_mask,
                                        keep_longest_lines(edge_up_down, True, top_n=10))

        print('vertical: ')
        clean_mask = cv2.bitwise_or(clean_mask,
                                        keep_longest_lines(edge_left_right, False, top_n=10))

        self._debug(show, clean_mask, "Purify")
        return clean_mask

    # -----------------------------
    # 5 glue overflow detect
    # -----------------------------

    def detect_glue_overflow(self, original_gray, ring_binary, show=True):

        _, ring_binary = cv2.threshold(
            ring_binary, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU
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
        self._debug(show, merged_mask, "13 Refined Mask")
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
                if area < 200 or area > 2000:
                    continue
                x, y, w, h = cv2.boundingRect(o_cnt)
                if x <= 0 or y <= 0 or (x + w) >= w_img or (y + h) >= h_img:
                    continue
                overflow_count += 1
                cv2.rectangle(display, (x-15, y-15), (x+w+15, y+h+15), (0, 100, 255), 3)

        self._debug(show, display, "14 Overflow Final")

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
