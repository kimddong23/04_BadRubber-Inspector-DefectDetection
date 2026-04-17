import cv2
import numpy as np

class HeatmapColorRanges:
    
    def __init__(self, amap: np.ndarray):
        heatmap = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        self.heatmap_hsv = self._convert_to_hsv(heatmap)

    def _convert_to_hsv(self, heatmap: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

    def red(self) -> np.ndarray:
        return cv2.inRange(
            self.heatmap_hsv,
            np.array([0, 70, 70], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )

    def red_high(self) -> np.ndarray:
        return cv2.inRange(
            self.heatmap_hsv,
            np.array([160, 70, 70], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        
    def yellow(self) -> np.ndarray:
        return cv2.inRange(
            self.heatmap_hsv,
            np.array([25, 70, 70], dtype=np.uint8),
            np.array([50, 255, 255], dtype=np.uint8),
        )