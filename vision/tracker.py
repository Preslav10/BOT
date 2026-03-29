
import numpy as np

class Tracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def _iou(self, a, b):
        x1=max(a[0],b[0])
        y1=max(a[1],b[1])
        x2=min(a[2],b[2])
        y2=min(a[3],b[3])

        inter=max(0,x2-x1)*max(0,y2-y1)
        areaA=(a[2]-a[0])*(a[3]-a[1])
        areaB=(b[2]-b[0])*(b[3]-b[1])
        union=areaA+areaB-inter

        return inter/union if union>0 else 0

    def update(self, detections):
        updated = {}

        for det in detections:
            box = det["box"]
            best_id = None
            best_iou = 0

            for tid, tbox in self.tracks.items():
                iou = self._iou(box, tbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_iou > 0.3:
                updated[best_id] = box
                det["track_id"] = best_id
            else:
                tid = self.next_id
                self.next_id += 1
                updated[tid] = box
                det["track_id"] = tid

        self.tracks = updated
        return detections
