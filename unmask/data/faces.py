"""Face detection (MTCNN), face choice across frames, and square cropping.

Face choice (PLAN 6.2): most confident face in the first frame that has one, then in each later frame the
detection whose centre is nearest to the previous chosen box. A detection further than one box side away
is treated as a miss, so the crop cannot jump to another person. Misses are filled from the nearest frame.
"""
import cv2
import numpy as np


class FaceDetector:
    """MTCNN on GPU. detect() returns, per frame, (boxes Nx4 in original pixels, probs N) or None."""

    def __init__(self, device="cuda", max_side=960):
        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.max_side = max_side

    def detect(self, frames_bgr):
        h, w = frames_bgr[0].shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        imgs = []
        for f in frames_bgr:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            if s < 1:
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(imgs)
        out = []
        for b, p in zip(boxes, probs):
            if b is None or len(b) == 0:
                out.append(None)
            else:
                out.append((np.asarray(b, dtype=np.float64) / s, np.asarray(p, dtype=np.float64)))
        return out

    def __call__(self, frames_bgr):
        return choose_track(self.detect(frames_bgr))


def _centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def choose_track(dets):
    """Per-frame detections -> one box per frame, or None if no frame has a face."""
    chosen = [None] * len(dets)
    prev = None
    for i, d in enumerate(dets):
        if d is None:
            continue
        boxes, probs = d
        if prev is None:
            b = boxes[int(np.argmax(probs))]
        else:
            dist = np.linalg.norm(np.stack([_centre(x) for x in boxes]) - _centre(prev), axis=1)
            j = int(np.argmin(dist))
            side = max(prev[2] - prev[0], prev[3] - prev[1])
            if dist[j] > side:
                continue
            b = boxes[j]
        chosen[i] = [float(v) for v in b]
        prev = chosen[i]
    return fill_missing(chosen)


def fill_missing(boxes):
    have = [i for i, b in enumerate(boxes) if b is not None]
    if not have:
        return None
    return [b if b is not None else boxes[min(have, key=lambda j: abs(j - i))] for i, b in enumerate(boxes)]


def square_crop(frame, box, margin=0.30, size=224):
    """Square crop around box with `margin` added per side, clipped to the frame, resized to size x size."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = max(1, int(round(min(side, W, H))))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return cv2.resize(frame[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def crops_for(frames, boxes, ref_shape, margin=0.30, size=224):
    """Crop each frame with its box. Boxes are rescaled if a frame's resolution differs from ref_shape."""
    out = []
    for f, b in zip(frames, boxes):
        if f.shape[:2] != tuple(ref_shape):
            sy, sx = f.shape[0] / ref_shape[0], f.shape[1] / ref_shape[1]
            b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        out.append(square_crop(f, b, margin, size))
    return out
