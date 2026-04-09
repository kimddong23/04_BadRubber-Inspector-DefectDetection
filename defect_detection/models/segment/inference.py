from typing import List, Sequence, Tuple, Dict

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO

from defect_detection.outputs import Segmentation
from .classes import classes


class Segmenter:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int = 32,
        conf_threshold: float = 0.5,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes = classes
        self._warmup()

    def _warmup(self, batch_size: int = 1) -> None:
        for _ in tqdm.tqdm(range(5), desc="Warm up YOLO segmenter model"):
            dummy = [np.zeros((self.imgsz, self.imgsz, 3), np.uint8)]
            _ = self.model(dummy, imgsz=self.imgsz, verbose=False)

    def infer_patches(
        self,
        patches: Sequence[np.ndarray],
        offsets: Sequence[Tuple[int, int, int, int]],  # x1, y1, W, H
        full_w: int,
        full_h: int,
    ) -> List[Segmentation]:
        if len(patches) == 0:
            return []

        results = self.model(
            patches,
            imgsz=self.imgsz,
            verbose=False,
        )

        class_polys: Dict[int, List[Tuple[np.ndarray, float, float]]] = {}

        for r, (x1, y1, W, H) in zip(results, offsets):

            if r.masks is None:
                continue

            for mask, cls_id, conf in zip(
                r.masks.xy,
                r.boxes.cls,
                r.boxes.conf,
            ):
                conf = float(conf)
                if conf < self.conf_threshold:
                    continue

                cls_id = int(cls_id)

                polygon_patch = np.array(mask)

                polygon_global = polygon_patch.copy()
                polygon_global[:, 0] += x1
                polygon_global[:, 1] += y1
                polygon_global = polygon_global.astype(np.float32)

                area = cv2.contourArea(polygon_global)

                class_polys.setdefault(cls_id, []).append(
                    (polygon_global, conf, area)
                )

        return self._merge_polygons_by_class(
            class_polys,
            full_w,
            full_h,
        )

    def _merge_polygons_by_class(
        self,
        class_polys: Dict[int, List[Tuple[np.ndarray, float, float]]],
        W: int,
        H: int,
    ) -> List[Segmentation]:

        region_segments: List[Segmentation] = []

        for cls_id, polys in class_polys.items():
            if self.classes[cls_id]["pass"]:
                continue

            mask = np.zeros((H, W), dtype=np.uint8)

            # 1. mask 생성 (global 그대로)
            for poly, _, _ in polys:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

            # 2. morphology (optional)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 3. contour 추출
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                contour = contour.squeeze(1).astype(np.float32)

                if contour.shape[0] < 3:
                    continue

                # 4. 포함 polygon 찾기
                merged_confs = []
                merged_areas = []

                for poly, conf, area in polys:
                    cx, cy = poly.mean(axis=0)

                    if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                        merged_confs.append(conf)
                        merged_areas.append(area)

                if len(merged_confs) == 0:
                    continue

                merged_confs = np.array(merged_confs)
                merged_areas = np.array(merged_areas)

                # 5. confidence (area-weighted)
                confidence = float(
                    np.sum(merged_confs * merged_areas)
                    / np.sum(merged_areas)
                )

                # 6. bbox
                xmin, ymin = contour.min(axis=0)
                xmax, ymax = contour.max(axis=0)

                bbox_xyxy = (
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax),
                )

                # 7. normalize (전체 기준)
                polygon_n = contour.copy()
                polygon_n[:, 0] /= W
                polygon_n[:, 1] /= H

                xmin_n, ymin_n = polygon_n.min(axis=0)
                xmax_n, ymax_n = polygon_n.max(axis=0)

                bboxes_xyxy_n = (
                    max(0.0, min(1.0, float(xmin_n))),
                    max(0.0, min(1.0, float(ymin_n))),
                    max(0.0, min(1.0, float(xmax_n))),
                    max(0.0, min(1.0, float(ymax_n))),
                )

                # 8. area
                area = cv2.contourArea(contour)
                area_n = area / float(W * H)

                region_segments.append(
                    Segmentation(
                        polygon=contour,
                        polygon_n=polygon_n,
                        bboxes_xyxy=bbox_xyxy,
                        bboxes_xyxy_n=bboxes_xyxy_n,
                        confidence=confidence,
                        area=area,
                        area_n=area_n,
                        class_id=cls_id,
                        class_name=self.classes[cls_id]["name"],
                        color=(0, 0, 255),
                    )
                )

        return region_segments