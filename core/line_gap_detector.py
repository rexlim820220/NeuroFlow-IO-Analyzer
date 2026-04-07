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
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), -1)

        self.debug(show, image, "11 Green contours")

        if scored_contours:
            max_area = max(item['area'] for item in scored_contours)
            scored_contours = [c for c in scored_contours if c['area'] > max_area * 0.009]

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
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        #image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        for k in range(n):
            s1 = scored_contours[k]
            s2 = scored_contours[(k + 1) % n]

            pts1 = s1['cnt'].reshape(-1, 2)
            pts2 = s2['cnt'].reshape(-1, 2)

            tree = KDTree(pts2)
            d, idx = tree.query(pts1, k=1)
            min_idx = d.argmin()

            import random
            color = (random.randint(150, 255), random.randint(170, 255), random.randint(155, 253))
            cv2.putText(image, f"{k+1}", s1['center'], cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 3, cv2.LINE_AA)
            # cv2.drawContours(image, [s1['cnt']], -1, color, 1)

            p1 = pts1[min_idx]
            p2 = pts2[idx[min_idx]]

            if self._check_line_break(p1, p2, 255-gray, k+1, show):
                self._draw_rect(image, p1, p2)
                #self._draw_line_and_points(image, p1, p2)
                real_gaps += 1

        self.debug(show, image, "12 Red Gaps")
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
        import random
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

    def _check_line_break(self, p1, p2, edge_mask, id=0, show=False):
        tl_x = int(min(p1[0], p2[0]))
        tl_y = int(min(p1[1], p2[1]))

        br_x = int(max(p1[0], p2[0]))
        br_y = int(max(p1[1], p2[1]))

        tl = (tl_x, tl_y)
        br = (br_x, br_y)

        width = br[0] - tl[0]
        height = br[1] - tl[1]

        MIN_WIDTH = 3
        MIN_HEIGHT = 3

        if width < MIN_WIDTH and height < MIN_HEIGHT:
            return False

        original_image = edge_mask.copy()

        new_image = cv2.line(np.zeros(edge_mask.shape, np.uint8), tuple(p1), tuple(p2), 255, thickness=5)

        mask = cv2.bitwise_and(new_image, original_image)

        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = mask[y:y+h, x:x+w]

            _, binary_diff = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            self.debug(show, binary_diff, f"12.{id} Gap")
            h, w = binary_diff.shape[:2]

            if w > h:
                line_densities = np.sum(binary_diff, axis=1) / (w * 255)
                max_density = np.max(line_densities)
            else:
                line_densities = np.sum(binary_diff, axis=0) / (h * 255)
                max_density = np.max(line_densities)

            IS_CONNECTED_THRESHOLD = 0.3

            if max_density > IS_CONNECTED_THRESHOLD:
                print(f"Gap ({id}, {id+1}): 發現貫穿線 (Density: {max_density:.2f}) -> 判定為假斷點")
                return False

            print(f"Gap ({id}, {id+1}): 無貫穿線 (Density: {max_density:.2f}) -> 判定為真斷點")
            return True
