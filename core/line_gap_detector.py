import cv2
import math
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
        gaps = 0
        edges = self._detect_edges(ring_binary)
        self.debug(show, edges, "10 Edges")
        contours = self._extract_major_contours(edges)
        self._draw_contours(gray, contours, show)
        gaps, display = self._detect_gaps(contours, edges, show)
        return display, gaps

    # =========================
    # Step 1: 邊緣偵測
    # =========================
    def _detect_edges(self, src):
        edges = cv2.medianBlur(src, 1)
        _, binary = cv2.threshold(
            255-edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    # =========================
    # Step 2: 取主要輪廓
    # =========================
    def _extract_major_contours(self, edges):
        POLY_EPSILON_RATIO = 0.001
        DYNAMIC_RATIO = 0.5
        num_labels, labels = cv2.connectedComponents(edges, connectivity=8)
        contours = []

        for idx in range(1, num_labels):
            mask = (labels == idx).astype(np.uint8) * 255
            cn, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            epsilon = POLY_EPSILON_RATIO * cv2.arcLength(cn[0], True)
            approx = cv2.approxPolyDP(cn[0], epsilon, False)
            length = cv2.arcLength(approx, True)

            if length:
                contours.append((approx, length))

        avg_len = np.mean([c[1] for c in contours]) if contours else 0
        contours = [c[0] for c in contours if c[1] > avg_len * DYNAMIC_RATIO]
        return contours

    # =========================
    # Step 3: 畫輪廓
    # =========================
    def _draw_contours(self, gray, contours, show):
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
        self.debug(show, image, "11 Green contours")

    # =========================
    # Step 4: 計算 gap
    # =========================
    def _detect_gaps(self, contours, gray, show):
        if len(contours) < 2:
            return 0, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img_h, img_w = gray.shape
        img_center = (img_w // 2, img_h // 2)

        scored_contours = []
        for cnt in contours:
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

        all_radii = [s['radius'] for s in scored_contours]
        median_radius = np.median(all_radii)

        valid_scored = [s for s in scored_contours 
                        if median_radius * 0.7 < s['radius'] < median_radius * 1.3]

        if len(valid_scored) < 2: return 0, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        valid_scored.sort(key=lambda x: x['angle'])

        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        real_gaps = 0
        n = len(valid_scored)
        
        for k in range(n):
            s1 = valid_scored[k]
            s2 = valid_scored[(k + 1) % n]
            
            pts1 = s1['cnt'].reshape(-1, 2)
            pts2 = s2['cnt'].reshape(-1, 2)
            
            tree = KDTree(pts2)
            d, idx = tree.query(pts1, k=1)
            min_idx = d.argmin()
            min_dist = d[min_idx]

            if min_dist:
                p1 = pts1[min_idx]
                p2 = pts2[idx[min_idx]]

                line_samples = np.linspace(p1, p2, 30).astype(int)
                hits = sum(
                    gray[int(pt[1]), int(pt[0])] > 0
                    for pt in line_samples
                )
                if hits > 9:
                    continue

                def get_tangent_vector(cnt, anchor_pt):
                    dists = np.linalg.norm(cnt.reshape(-1, 2) - anchor_pt, axis=1)
                    tail_pts = cnt.reshape(-1, 2)[dists.argsort()[:25]]
                    if len(tail_pts) < 2: return None
                    [vx, vy, _, _] = cv2.fitLine(tail_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    return np.array([vx[0], vy[0]])
                
                v1 = get_tangent_vector(s1['cnt'], p1)
                
                v_gap = (p2 - p1).astype(float)
                v_gap_norm = np.linalg.norm(v_gap)
                if v_gap_norm < 9:
                    continue
                if v_gap_norm > 228:
                    continue
                elif v_gap_norm > 0:
                    v_gap /= v_gap_norm
                
                if v1 is not None:
                    cos_theta = abs(np.dot(v1.flatten(), v_gap.flatten()))
                    if cos_theta < 0.3:
                        continue

                self._draw_rect(image, p1, p2)
                self._draw_line_and_points(image, p1, p2)
                real_gaps += 1

        self.debug(show, image, "11 Red Gaps")
        result_text = f"{self.filename} NG (偵測到{real_gaps}個潛在溢膠點)" if real_gaps > 0 else "PASS"
        print(f"{result_text}")
        return real_gaps, image

    def _draw_rect(self, image, p1, p2):
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)

        diag_dist = math.dist(p1, p2)

        box_size = int(diag_dist * 0.2)
        box_size = max(10, min(box_size, 100)) 

        tl = (mid_x - box_size, mid_y - box_size)
        br = (mid_x + box_size, mid_y + box_size)

        cv2.rectangle(image, tl, br, (0, 0, 255), 3)

    def _draw_line_and_points(self, image, p1, p2):
        """
        在 image 上畫出 p1 與 p2 的線段，並用圓圈標記兩點。
        顏色隨機指定。
        """
        import random
        r = random.randint(0, 45)    # 紅色偏低
        g = random.randint(100, 255) # 綠色偏高
        b = random.randint(45, 160)  # 藍色偏低
        color = (b, g, r)

        # 畫線
        cv2.line(image, tuple(p1), tuple(p2), color, thickness=4)

        # 畫圓圈
        cv2.circle(image, tuple(p1), radius=3, color=color, thickness=-1)
        cv2.circle(image, tuple(p2), radius=3, color=color, thickness=-1)

        distance = np.linalg.norm(p1 - p2)
        mid_point = ((p1[0]+p2[0])//2+10, (p1[1]+p2[1])//2-20)
        cv2.putText(image, f"{distance:.2f}", mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 127, 9), 3, cv2.LINE_AA)
