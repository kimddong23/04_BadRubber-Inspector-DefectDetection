import os

import cv2

from defect_detection import Detector

if __name__ == "__main__":
    # 1. input datas
    img_dir = "./tests/dot"
    imgs_path = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir) if img_path.endswith(".jpg")]
    imgs = [cv2.imread(img_path) for img_path in imgs_path]
    dst_dir = "./tests/dot_results"
    os.makedirs(dst_dir, exist_ok=True)

    # 2. inspect
    detector = Detector()
    results = detector.detect(imgs)

    # 3. visualize
    for img_path, result in zip(imgs_path, results):
        image_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(dst_dir, image_name), result.visualize())