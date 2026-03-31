import cv2
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

        d = 3
        # -------- shift function --------
        def shift(img, offset, axis):
            res = np.zeros_like(img)

            if axis == 0:  # vertical shift
                if offset > 0:
                    res[offset:] = img[:-offset]
                else:
                    res[:offset] = img[-offset:]

            else:  # horizontal shift
                if offset > 0:
                    res[:, offset:] = img[:, :-offset]
                else:
                    res[:, :offset] = img[:, -offset:]

            return res

        # ============================
        # detect vertical lines
        # ============================

        up = shift(ring_binary, -d, 0)
        down = shift(ring_binary, d, 0)

        vertical_support = np.maximum(up, down)
        vertical_lines = np.minimum(ring_binary, vertical_support)

        # ============================
        # detect horizontal lines
        # ============================

        left = shift(ring_binary, -d, 1)
        right = shift(ring_binary, d, 1)

        horizontal_support = np.maximum(left, right)
        horizontal_lines = np.minimum(ring_binary, horizontal_support)

        # ============================
        # expand perpendicular
        # ============================

        # vertical line → horizontal expand
        v_expand = np.maximum.reduce([
            shift(vertical_lines, -1, 1),
            vertical_lines,
            shift(vertical_lines, 1, 1)
        ])

        # horizontal line → vertical expand
        h_expand = np.maximum.reduce([
            shift(horizontal_lines, -1, 0),
            horizontal_lines,
            shift(horizontal_lines, 1, 0)
        ])

        ring_binary = np.maximum(v_expand, h_expand).astype(np.uint8) * 255

        edges = self._detect_edges(ring_binary)

        self.debug(show, edges, f"10 Edges")

        contours = self._extract_major_contours(edges)

        gaps, display = self._detect_gaps(contours, edges, gray, show)

        return display, gaps

    # =========================
    # Step 1: 邊緣偵測
    # =========================
    def _detect_edges(self, src):
        _, binary = cv2.threshold(
            src, 5, 255, cv2.THRESH_BINARY
        )
        return binary

    # =========================
    # Step 2: 取主要輪廓
    # =========================
    def _extract_major_contours(self, edges):
        POLY_EPSILON_RATIO = 0.001
        DYNAMIC_RATIO = 0.5

        clean_solid_mask = np.zeros_like(edges)
        num_labels, labels = cv2.connectedComponents(edges, connectivity=8)

        for idx in range(1, num_labels):
            comp_mask = (labels == idx).astype(np.uint8) * 255
            cn, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cn:
                cv2.drawContours(clean_solid_mask, cn, -1, 255, thickness=cv2.FILLED)
                cv2.drawContours(clean_solid_mask, cn, -1, 255, thickness=1)

        final_contours = []
        solid_cn, _ = cv2.findContours(clean_solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        MIN_AREA = 10
        TANGENT_TAIL = 25
        MIN_COS_THETA = 0.3
        NUM_SAMPLE = 20
        MIN_HITS = 11
        MAX_DIST = 300
        MIN_DIST = 0

        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if len(contours) < 2:
            return 0, image

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
                'center': (cx, cy),
                'radius': dist_to_center,
                'angle': np.arctan2(cy - img_center[1], cx - img_center[0])
            })
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        if len(scored_contours) < 2:
            return 0, image

        self.debug(show, image, "11 Green contours")

        scored_contours.sort(key=lambda x: x['angle'])

        real_gaps = 0
        n = len(scored_contours)

        for k in range(n):
            s1 = scored_contours[k]
            s2 = scored_contours[(k + 1) % n]

            pts1 = s1['cnt'].reshape(-1, 2)
            pts2 = s2['cnt'].reshape(-1, 2)

            tree = KDTree(pts2)
            d, idx = tree.query(pts1, k=1)
            min_idx = d.argmin()
            min_dist = d[min_idx]

            if not (MIN_DIST <= min_dist <= MAX_DIST):
                continue
            else:
                import random
                color = (0, random.randint(170, 255), random.randint(155, 253))
                cv2.putText(image, f"{k+1}", s1['center'], cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 3, cv2.LINE_AA)
                cv2.drawContours(image, [s1['cnt']], -1, color, 1)

            p1 = pts1[min_idx]
            p2 = pts2[idx[min_idx]]

            '''line_samples = np.linspace(p1, p2, NUM_SAMPLE).astype(int)
            hits = sum(
                edge[int(pt[1]), int(pt[0])] > 0
                for pt in line_samples
                if 0 <= pt[1] < img_h and 0 <= pt[0] < img_w
            )
            if hits >= MIN_HITS:
                continue

            def get_tangent_vector(cnt, anchor_pt):
                dists = np.linalg.norm(cnt.reshape(-1, 2) - anchor_pt, axis=1)
                tail_pts = cnt.reshape(-1, 2)[dists.argsort()[:TANGENT_TAIL]]
                if len(tail_pts) < 2:
                    return None
                [vx, vy, _, _] = cv2.fitLine(tail_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                return np.array([vx[0], vy[0]])

            v1 = get_tangent_vector(s1['cnt'], p1)
            if v1 is None:
                continue

            v_gap = (p2 - p1).astype(float)
            v_gap_norm = np.linalg.norm(v_gap)
            if v_gap_norm > 0:
                v_gap /= v_gap_norm

            cos_theta = abs(np.dot(v1.flatten(), v_gap.flatten()))
            if cos_theta < MIN_COS_THETA:
                continue'''

            self._draw_rect(image, p1, p2)
            self._draw_line_and_points(image, p1, p2)
            real_gaps += 1

        self.debug(show, image, "12 Red Gaps")
        return real_gaps, image

    def _draw_rect(self, image, p1, p2):
        tl_x = int(min(p1[0], p2[0]))
        tl_y = int(min(p1[1], p2[1]))

        br_x = int(max(p1[0], p2[0]))
        br_y = int(max(p1[1], p2[1]))

        padding = 25
        tl = (tl_x - padding, tl_y - padding)
        br = (br_x + padding, br_y + padding)

        cv2.rectangle(image, tl, br, (0, 0, 255), 3)

    def _draw_line_and_points(self, image, p1, p2):
        """
        在 image 上畫出 p1 與 p2 的線段，並用圓圈標記兩點。
        顏色隨機指定。
        """
        import random
        r = random.randint(100, 255)# 紅色偏高
        g = random.randint(0, 45)   # 綠色偏低
        b = random.randint(45, 160) # 藍色偏低
        color = (b, g, r)

        # 畫線
        cv2.line(image, tuple(p1), tuple(p2), color, thickness=4)

        # 畫圓圈
        cv2.circle(image, tuple(p1), radius=3, color=color, thickness=-1)
        cv2.circle(image, tuple(p2), radius=3, color=color, thickness=-1)

        distance = np.linalg.norm(p1 - p2)
        mid_point = ((p1[0]+p2[0])//2+10, (p1[1]+p2[1])//2-20)
        #cv2.putText(image, f"{distance:.2f}", mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 127, 9), 3, cv2.LINE_AA)
