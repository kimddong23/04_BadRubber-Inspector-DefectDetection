import time
from typing import List, Tuple

import cv2

from defect_detection.models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, Segmenter, RegionSegmenterAdapter, ObjectDetector, Cluster
from defect_detection.outputs import RegionClassificationOutput, ClassificationBatchItem, merge_anomlay_outputs, filter_by_cluster
from defect_detection.utils import load_config, random_color
from .result import DetectorOutput
from .visualize import draw_normalized_polygons

class Detector:
    def __init__(self):
        config = load_config()

        self.anomaly_extractor = AnomalyCLIPInference(
            checkpoint_path=config["anomalyclip"]["checkpoint"],
            imgsz=config["anomalyclip"]["imgsz"],
            score_threshold=config["anomalyclip"]["threshold"],
            area_threshold=config["anomalyclip"]["min_area"],
        )

        self.bgremover = BackgroundRemover(
            checkpoint_path=config["bgremover"]["checkpoint"],
            imgsz=config["bgremover"]["imgsz"],
        )

        if config['cluster'] is not None:
            self.region_cluster = RegionClassifierAdapter(
                Cluster(
                checkpoints_path=config["cluster"]["checkpoints_path"],
                )
            )
        else:
            self.region_cluster = None

        if config['dot_detector1'] is not None:
            self.dot_detector1 = ObjectDetector(
                checkpoint_path=config["dot_detector1"]["checkpoint"],
                imgsz=config["dot_detector1"]["imgsz"],
                threshold=config["dot_detector1"]["threshold"],
                name="dot_detector1",
            )
        else:
            self.dot_detector1 = None

        if config['dot_detector2'] is not None:
            self.dot_detector2 = ObjectDetector(
                checkpoint_path=config["dot_detector2"]["checkpoint"],
                imgsz=config["dot_detector2"]["imgsz"],
                threshold=config["dot_detector2"]["threshold"],
                name="dot_detector2",
            )
        else:
            self.dot_detector2 = None

        if config['classifier'] is not None:
            self.region_classifier = RegionClassifierAdapter(
                Classifier(
                checkpoint_path=config["classifier"]["checkpoint"],
                imgsz=config["classifier"]["imgsz"],
                conf_threshold=config["classifier"]["threshold"],
                )
            )
        else:
            self.region_classifier = None

        self.region_segmenter = RegionSegmenterAdapter(
            Segmenter(
            checkpoint_path=config["segmenter"]["checkpoint"],
            imgsz=config["segmenter"]["imgsz"],
            conf_threshold=config["segmenter"]["threshold"],
            )
        )

    # ---------------------------------
    # Main API
    # ---------------------------------

    def detect(self, imgs_path: List[str]) -> DetectorOutput:
        t0 = time.time()

        # read images
        images = [cv2.imread(p) for p in imgs_path]
        t1 = time.time()

        foreground = self.bgremover.infer(images)
        t2 = time.time()

        # detect anomaly regions
        anomaly = self.anomaly_extractor.infer(images, foreground.masks)
        t3 = time.time()

        clusters = self.region_cluster.infer(images, anomaly) if self.region_cluster is not None else None
        anomaly = filter_by_cluster(anomaly, clusters) if clusters is not None else anomaly
        t4 = time.time()

        dot1 = self.dot_detector1.infer(images) if self.dot_detector1 is not None else None
        t5 = time.time()

        dot2 = self.dot_detector2.infer(images) if self.dot_detector2 is not None else None
        t6 = time.time()
        
        merged_anomaly = merge_anomlay_outputs([anomaly, dot1, dot2])
        t7 = time.time()

        # classify anomaly regions
        merged_anomaly_cls = self.region_classifier.infer(images, merged_anomaly) if self.region_classifier is not None else clusters
        t8 = time.time()

        segmentation = self.region_segmenter.infer(images, merged_anomaly, merged_anomaly_cls)
        t9 = time.time()

        segmentation_cls = [ClassificationBatchItem(regions=[]) for _ in range(len(images))]
        t10 = time.time()

        print(f"load images: {(t1-t0)*1000}ms")
        print(f"foreground: {(t2-t1)*1000}ms")
        print(f"anomaly: {(t3-t2)*1000}ms")
        print(f"cluster: {(t4-t3)*1000}ms")
        print(f"dot1: {(t5-t4)*1000}ms")
        print(f"dot2: {(t6-t5)*1000}ms")
        print(f"merged_anomaly: {(t7-t6)*1000}ms")
        print(f"merged_anomaly_cls: {(t8-t7)*1000}ms")
        print(f"segmentation: {(t9-t8)*1000}ms")
        print(f"segmentation_cls: {(t10-t9)*1000}ms")
        print(f"image count: {len(images)}")
        print(f"total: {(t10-t0)*1000}ms")

        return DetectorOutput(
            images=images,
            images_path=imgs_path,
            foreground=foreground,
            anomaly=merged_anomaly,
            anomaly_cls=merged_anomaly_cls,
            segmentation=segmentation,
            segmentation_cls=segmentation_cls,
        )