import os
import json
import argparse
import sys
import signal

import cv2
import tqdm
from datetime import datetime

from defect_detection import Detector
from defect_detection.detect import crop_regions
from defect_detection.utils import Report, save_polygons_to_yolo_format, next_path

current_processing_path = None

def update_progress(processing_path, total, processed, last_file, start_time, status="processing"):
    global current_processing_path
    progress = {
        "status" : status,
        "total_files": total,
        "processed_files": processed,
        "progress_percent": round((processed / total) * 100, 2),
        "last_processed_file": last_file,
        "start_time": start_time,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(processing_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=4, ensure_ascii=False)


def handle_exit(signum=None, frame=None):
    global current_processing_path

    if current_processing_path is not None:

        try:
            with open(current_processing_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["status"] = "stop"
            data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(current_processing_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception:
            pass

    sys.exit(0)

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def run(src_root, dst_root, line, grade, dates, batch_size=9):   

    global current_processing_path

    report = Report()
    detector = Detector()

    # dates
    for date in dates:

        processing_path = os.path.join(dst_root, line, grade, f"{date}.processing.json")

        current_processing_path = processing_path

        src_dir = os.path.join(src_root, line, date)
        dst_dir = os.path.join(dst_root, line, grade, date)
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dst_result_image_dir = os.path.join(dst_dir, "results", "images")
        dst_result_meta_dir = os.path.join(dst_dir, "results", "metadatas")
        dst_crop_images_dir = os.path.join(dst_dir, "crops", "images")
        dst_crop_labels_dir = os.path.join(dst_dir, "crops", "labels")
        os.makedirs(dst_result_image_dir, exist_ok=True)
        os.makedirs(dst_result_meta_dir, exist_ok=True)
        os.makedirs(dst_crop_images_dir, exist_ok=True)
        os.makedirs(dst_crop_labels_dir, exist_ok=True)
        
        cams = [
            d for d in tqdm.tqdm(os.listdir(src_dir), desc=f"Loading {date} {line} {grade} cams")
            if os.path.isdir(os.path.join(src_dir, d))
            and d.startswith("CAM")
        ] or [""]

        all_files = []

        for cam in cams:
            cam_dir = os.path.join(src_dir, cam)
            files = [f for f in os.listdir(cam_dir) if f.endswith(".jpg")]
            all_files.extend([(cam, f) for f in files])

        total_files = len(all_files)
        processed_count = 0

        if total_files == 0:
            continue

        # cams
        for cam in cams:
            cam_dir = os.path.join(src_dir, cam)

            files = [f for f in os.listdir(cam_dir)if f.endswith(".jpg")]

            # batch inference
            for file_batch in tqdm.tqdm(list(batch(files, batch_size=9)),desc=f"{date} {cam} batch inference"):
                print("\nbatch size: ", len(file_batch))
                img_paths = [os.path.join(cam_dir, f) for f in file_batch]
                results = detector.detect(img_paths)

                for result in results:
                    # reclassification
                    for b_idx in range(len(result.anomaly.regions)):
                        anomaly_regions = result.anomaly.regions[b_idx]
                        anomaly_classes = result.anomaly_cls.regions[b_idx]
                        segmentations = result.segmentation.regions[b_idx]
                        for polygon_source in segmentations:
                            polygon_source.class_id = anomaly_classes.class_id
                            polygon_source.class_name = anomaly_classes.class_name

                    # result image
                    imagename = os.path.splitext(os.path.basename(result.image_path))[0]
                    # cv2.imwrite(os.path.join(dst_result_image_dir, f"{imagename}.jpg"), result.visualize())

                    crops, metadata = crop_regions(
                        image=result.image,
                        imagename=imagename,
                        crop_sources=result.anomaly.regions,
                        crop_cls_sources=result.anomaly_cls.regions,
                        polygon_sources=result.segmentation.regions,
                    )
                    with open(os.path.join(dst_result_meta_dir, f"{imagename}.json"), "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4)
                    for filename, (crop, segmentations) in crops.items():
                        cls_name = filename.split("_")[0]
                        dst_crop_image_next_dir = next_path(os.path.join(dst_crop_images_dir, cls_name), limit=5000)
                        cv2.imwrite(os.path.join(dst_crop_image_next_dir, f"{filename}.jpg"), crop)
                        save_polygons_to_yolo_format(os.path.join(dst_crop_labels_dir, f"{filename}.txt"), [seg["polygon"] for seg in segmentations], [seg["class_id"] for seg in segmentations])

                    processed_count += 1
                    update_progress(
                        processing_path,
                        total_files,
                        processed_count,
                        imagename,
                        start_time
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, default="./NAS/LG_Chemistry_S1K2")
    parser.add_argument("--dst-root", type=str, default="./NAS/LG_Chemistry_Site")
    parser.add_argument("--line", type=str)
    parser.add_argument("--grade", type=str)
    parser.add_argument("--dates", type=str, nargs="+")
    parser.add_argument("--batch-size", type=int, default=9, help="Batch size")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    run(src_root=args.src_root, dst_root=args.dst_root, line=args.line, grade=args.grade, dates=args.dates, batch_size=args.batch_size)