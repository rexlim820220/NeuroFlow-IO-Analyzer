import cv2
import random
import numpy as np
from scipy.spatial import KDTree

class LineGapDetector:

    def __init__(self,
                 min_area,
                 filename,
                 debug=False
                 ):
        self.min_area = min_area
        self.filename = filename
        self.debug = debug

    # =========================
    # 主流程
    # =========================
    def detect(self, gray, ring_binary, show=False):

        edges = self._detect_edges(ring_binary, show)

        self.debug(show, edges, "11 Edges")

        contours = self._extract_major_contours(edges)

        gaps, display = self._detect_gaps(contours, edges, gray, show)

        return display, gaps

    # =========================
    # Step 1: 邊緣偵測
    # =========================
    def _detect_edges(self, src, show):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))

        tophat_mask = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)

        _, refined  = cv2.threshold(tophat_mask, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.debug(show, refined, "10 tophat mask")

        return refined

    # =========================
    # Step 2: 取主要輪廓
    # =========================
    def _extract_major_contours(self, edges):
        POLY_EPSILON_RATIO = 1e-100
        DYNAMIC_RATIO = 0.1

        clean_solid_mask = np.zeros_like(edges)
        num_labels, labels = cv2.connectedComponents(edges, connectivity=8)

        for idx in range(1, num_labels):
            comp_mask = (labels == idx).astype(np.uint8) * 255
            cn, _ = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if cn:
                _, _, w, h = cv2.boundingRect(cn[0])
                single_obj_mask = np.zeros_like(clean_solid_mask)
                single_obj_mask[labels == idx] = 255
                cv2.drawContours(clean_solid_mask, cn, -1, 255, thickness=cv2.FILLED)
                kernel = np.ones((1, 1), np.uint8)
                if w > h * 1.5:
                    kernel = np.ones((5, 1), np.uint8)
                elif h > w * 1.5:
                    kernel = np.ones((1, 5), np.uint8)
                thickened = cv2.dilate(single_obj_mask, kernel)
                clean_solid_mask = cv2.bitwise_or(clean_solid_mask, thickened)

        final_contours = []
        solid_cn, _ = cv2.findContours(clean_solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        temp_list = []
        for c in solid_cn:
            epsilon = POLY_EPSILON_RATIO * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            length = cv2.arcLength(approx, True)
            if length > 0:
                temp_list.append((approx, length))

        if not temp_list:
            return []

        avg_len = np.mean([c[1] for c in temp_list])
        final_contours = [c[0] for c in temp_list if c[1] > avg_len * DYNAMIC_RATIO]

        return final_contours

    # =========================
    # Step 3: 計算 gap
    # =========================
    def _detect_gaps(self, contours, edge, gray, show):
        MIN_AREA = 0
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        debug = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        img_h, img_w = gray.shape
        img_center = (img_w // 2, img_h // 2)

        scored_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist_to_center = np.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
            scored_contours.append({
                'cnt': cnt,
                'area': area,
                'center': (cx, cy),
                'radius': dist_to_center,
                'angle': np.arctan2(cy - img_center[1], cx - img_center[0])
            })
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        self.debug(show, image, "11 Green contours")

        if scored_contours:
            max_area = max(item['area'] for item in scored_contours)
            scored_contours = [c for c in scored_contours if c['area'] > max_area * 0.01]

        scored_contours.sort(key=lambda x: x['angle'])

        real_gaps = 0

        n = len(scored_contours)
        if n == 1:
            cnt = scored_contours[0]['cnt']
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull_indices)

            if defects is None:
                return 0, image
            idx = np.argmax(defects[:,0,3])
            start_idx, end_idx, _, _ = defects[idx, 0]

            p1 = cnt[start_idx][0]
            p2 = cnt[end_idx][0]

            self._draw_rect(image, p1, p2)
            self.debug(show, image, "12 Red Gaps")
            return 1 if self._check_line_empty(p1, p2, edge) else 0, image

        print("================ Gap Overflow:================")
        for k in range(n):
            s1 = scored_contours[k]
            s2 = scored_contours[(k + 1) % n]

            pts1 = s1['cnt'].reshape(-1, 2)
            pts2 = s2['cnt'].reshape(-1, 2)

            tree = KDTree(pts2)
            d, idx = tree.query(pts1, k=1)
            min_idx = d.argmin()
            p1 = pts1[min_idx]
            p2 = pts2[idx[min_idx]]

            if self._check_line_break(p1, p2, debug, s1, s2, k+1, show):
                self._draw_rect(image, p1, p2, 5)
                real_gaps += 1

            color = (random.randint(50, 150), random.randint(70, 170), random.randint(55, 155))
            cv2.drawContours(debug, [s1['cnt']], -1, color, 2)
            cv2.putText(debug, f"{k+1}", s1['center'], cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)
            cv2.putText(debug, f"{k+1}", s1['center'], cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1, cv2.LINE_AA)

        self.debug(show, debug, "12 Red Gaps Normal Debug")
        self.debug(show, image, "13 Red Gaps")
        return real_gaps, image

    def _draw_rect(self, image, p1, p2, padding = 25):
        tl_x = int(min(p1[0], p2[0]))
        tl_y = int(min(p1[1], p2[1]))

        br_x = int(max(p1[0], p2[0]))
        br_y = int(max(p1[1], p2[1]))

        tl = (tl_x - padding, tl_y - padding)
        br = (br_x + padding, br_y + padding)

        cv2.rectangle(image, tl, br, (0, 0, 255), 3)

    def _draw_line_and_points(self, image, p1, p2):
        """
        Draw line segments p1 and p2 and mark the two points with circles.
        """
        r = random.randint(100, 255) # 紅色偏高
        g = random.randint(0, 45)    # 綠色偏低
        b = random.randint(45, 160)  # 藍色偏低
        color = (b, g, r)

        # 畫線
        cv2.line(image, tuple(p1), tuple(p2), color, thickness=4)

        # 畫圓圈
        cv2.circle(image, tuple(p1), radius=3, color=color, thickness=-1)
        cv2.circle(image, tuple(p2), radius=3, color=color, thickness=-1)

        #distance = np.linalg.norm(p1 - p2)
        #mid_point = ((p1[0]+p2[0])//2+10, (p1[1]+p2[1])//2-20)
        #cv2.putText(image, f"{distance:.2f}", mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 127, 9), 3, cv2.LINE_AA)

    def _check_line_break(self, p1, p2, debug_img, s1=None, s2=None, id=0, show=False):
        """
        Determine if it is a true break point
        """
        p1 = np.array(p1, dtype=int)
        p2 = np.array(p2, dtype=int)

        MIN_DIST = 4
        MIN_RATIO = 0.1
        dist = np.linalg.norm(p2 - p1)
        if dist < MIN_DIST:
            print(f"*({id}, {id+1}) ：線段太短({dist:.2f})直接視為非斷點")
            return False

        mid = (p1 + p2) // 2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        nx, ny = -dy / length, dx / length if length > 0 else (0, 1)

        sample_length = int(length * 0.5)
        half = sample_length // 2

        safety_margin = int(length * 0.05)
        white_count = 0
        total = 0

        for t in np.linspace(-half, half, half + 1):
            x = int(mid[0] + t * nx)
            y = int(mid[1] + t * ny)
            if  (0 <= x < debug_img.shape[1] and
                0 <= y < debug_img.shape[0] and
                abs(t) <= half - safety_margin):
                total += 1
                if np.any(debug_img[y, x] > 0):
                    white_count += 1

        intersection_ratio = white_count / total if total > 0 else 1.0

        direction_ok = True
        if s1 is not None and s2 is not None:
            direction_ok = self._is_line_aligned_with_contours(p1, p2, s1, s2, angle_threshold_deg=40)

        is_true_gap = (intersection_ratio < MIN_RATIO) and direction_ok

        if show:
            x1, y1 = p1
            x2, y2 = p2
            text_margin = 10
            x_text = (x1 + x2) // 2 - text_margin
            y_text = (y1 + y2) // 2 - text_margin
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.6
            thickness = 1
            line_type = cv2.LINE_AA
            line_spacing = int(25 * font_scale)

            cv2.line(debug_img, tuple(p1), tuple(p2), (0, 255, 0), 1)
            cv2.line(debug_img,
                    (mid[0] - int(half * nx), mid[1] - int(half * ny)),
                    (mid[0] + int(half * nx), mid[1] + int(half * ny)),
                    (0, 0, 255), 3)
            texts = [
                f"{intersection_ratio:.1f}",
                f"{'OK' if direction_ok else 'NG'}"
            ]

            for i, text in enumerate(texts):
                y_line = y_text - i * line_spacing
                cv2.putText(debug_img,
                            text,
                            (x_text, y_line),
                            font,
                            font_scale,
                            (255, 100 if i == 0 else 0, 0 if i == 0 else 110),
                            thickness,
                            line_type)

        print(f"Gap {id} | Ratio: {intersection_ratio:.3f} | Direction OK: {direction_ok} → {'真斷點' if is_true_gap else '偽斷點'}")

        return is_true_gap

    def _is_line_aligned_with_contours(self, p1, p2, s1, s2, angle_threshold_deg=35):
        """
        判斷 p1-p2 連線是否與 s1、s2 兩個連通體的主要走向一致
        """
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) < 1 and abs(dy) < 1:
            return False
        line_angle = np.arctan2(dy, dx)

        def get_rect_angle(s):
            _, _, w, h = cv2.boundingRect(s['cnt'])
            if w > h * 1.5:
                return 0.0
            elif h > w * 1.5:
                return np.pi / 2
            else:
                pts = s['cnt'].reshape(-1, 2)
                center = np.mean(pts, axis=0)
                farthest_idx = np.argmax(np.sum((pts - center)**2, axis=1))
                far_pt = pts[farthest_idx]
                dx2 = far_pt[0] - center[0]
                dy2 = far_pt[1] - center[1]
                return np.arctan2(dy2, dx2)

        angle1 = get_rect_angle(s1)
        angle2 = get_rect_angle(s2)

        def angle_diff(a, b):
            d = abs(a - b)
            return min(d, np.pi - d)

        diff1 = angle_diff(line_angle, angle1)
        diff2 = angle_diff(line_angle, angle2)

        return min(diff1, diff2) < np.deg2rad(angle_threshold_deg)
