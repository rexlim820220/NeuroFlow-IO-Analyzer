import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.spatial.distance import cdist

class GlueTrackDetector:

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

        _, final_display, result = self._detect_lines_and_gaps(
            gray, ring_binary, debug_callback
        )

        return final_display, result

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

        hull = cv2.convexHull(max_contour)

        hull_mask = np.zeros(gray.shape, np.uint8)

        cv2.drawContours(hull_mask, [hull], -1, 255, -1)

        inner = int(expand_distance * 0.8)

        kernel_inner = cv2.getStructuringElement(cv2.MORPH_RECT, (inner, inner))

        A_prime = cv2.dilate(hull_mask, kernel_inner)

        self._debug(show, A_prime, "6 Convex Hull")

        outer = int(expand_distance * 2.2)

        kernel_outer = cv2.getStructuringElement(cv2.MORPH_RECT, (outer, outer))

        B = cv2.dilate(hull_mask, kernel_outer)

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
    # 4 line detect
    # -----------------------------

    def _detect_lines_and_gaps(self, gray, ring_binary, show):

        edges = cv2.Canny(ring_binary, 50, 150)

        self._debug(show, edges, "10 Edges")

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            30,
            minLineLength=30,
            maxLineGap=15,
        )

        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = "PASS"

        if lines is not None:

            lines = lines.squeeze()

            endpoints = []

            for x1, y1, x2, y2 in lines:

                cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                endpoints.append((x1, y1))
                endpoints.append((x2, y2))

            dist = cdist(endpoints, endpoints)

            np.fill_diagonal(dist, np.inf)

            min_dist = np.min(dist, axis=1)

            gap_indices = np.where(min_dist > 20)[0]

            for idx in gap_indices:

                x, y = endpoints[idx]

                cv2.rectangle(
                    display,
                    (x - 10, y - 10),
                    (x + 10, y + 10),
                    (0, 0, 255),
                    2,
                )

            if len(gap_indices) > 0:
                result = f"NG (偵測到 {len(gap_indices)//2} 個斷點)"

        self._debug(show, display, "11 Final")

        return display, display, result

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
