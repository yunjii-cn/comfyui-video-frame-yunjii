import os
import math
import json
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

from .debug_log import info, warn, error

_TAG = "SDPoseBackend"

YOLO_INPUT_SIZE = 640
VITPOSE_INPUT_H = 256
VITPOSE_INPUT_W = 192
VITPOSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VITPOSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_BODY_KP = 23
NUM_FACE_KP = 68
NUM_HAND_KP = 21
NUM_TOTAL_KP = NUM_BODY_KP + NUM_FACE_KP + 2 * NUM_HAND_KP

BODY_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7), (7, 8),
    (1, 11), (11, 12), (12, 13), (13, 14),
    (14, 15), (15, 16), (16, 17),
    (11, 18), (18, 19), (19, 20),
    (0, 20), (20, 21), (21, 22),
]

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

BODY_COLORS = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102),
    (230, 230, 0), (255, 153, 51), (255, 128, 0),
    (255, 153, 51), (255, 178, 102), (230, 230, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170),
    (0, 255, 255), (85, 255, 255), (170, 255, 255),
    (255, 255, 255), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (85, 255, 255),
]

HAND_COLORS = [
    (100, 100, 255), (100, 100, 255), (100, 100, 255),
    (100, 100, 255), (100, 200, 255), (100, 200, 255),
    (100, 200, 255), (100, 200, 255), (100, 255, 200),
    (100, 255, 200), (100, 255, 200), (100, 255, 200),
    (200, 255, 100), (200, 255, 100), (200, 255, 100),
    (200, 255, 100), (255, 200, 100), (255, 200, 100),
    (255, 200, 100), (255, 200, 100), (255, 100, 100),
]

FACE_COLOR = (255, 255, 255)


def _get_3rd_point(a, b):
    d = a - b
    return np.array([b[0] - d[1], b[1] + d[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, inv=False):
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = _get_rot_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def _get_rot_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0.0, 0.0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return np.array(src_result, dtype=np.float32)


def transform_preds(pts, center, scale, output_size):
    trans = get_affine_transform(center, scale, 0, output_size, inv=True)
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = pts_h @ trans.T
    return out


def _get_max_preds(heatmaps):
    N, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(N, K, -1)
    idx = np.argmax(heatmaps_reshaped, axis=2)
    maxvals = np.amax(heatmaps_reshaped, axis=2)
    idx = idx.reshape(N, K, 1)
    maxvals = maxvals.reshape(N, K, 1)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    return preds, maxvals


def keypoints_from_heatmaps(heatmaps, center, scale):
    preds, maxvals = _get_max_preds(heatmaps)
    N, K, H, W = heatmaps.shape
    for n in range(N):
        for k in range(K):
            px = int(round(preds[n, k, 0]))
            py = int(round(preds[n, k, 1]))
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmaps[n, k, py, px + 1] - heatmaps[n, k, py, px - 1],
                    heatmaps[n, k, py + 1, px] - heatmaps[n, k, py - 1, px],
                ])
                preds[n, k] += np.sign(diff) * 0.25
    for n in range(N):
        pts = preds[n, :, :2]
        mapped = transform_preds(pts, center, scale, [W, H])
        preds[n, :, :2] = mapped
    keypoints = np.concatenate((preds, maxvals), axis=2)
    return keypoints


def bbox_from_detector(bbox, input_resolution, rescale=1.0):
    if isinstance(bbox, list):
        bbox = np.array(bbox, dtype=np.float32)
    if bbox.ndim == 1:
        bbox = bbox.reshape(1, -1)
    x1, y1, x2, y2 = bbox[0, :4]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    aspect_ratio = input_resolution[0] / input_resolution[1]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * rescale, h * rescale], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    return center, scale


def crop(img, center, scale, res):
    trans = get_affine_transform(center, scale, 0, res, inv=False)
    cropped = cv2.warpAffine(
        img, trans, (int(res[0]), int(res[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cropped


def _yolo_preprocess(img_bgr, input_size=YOLO_INPUT_SIZE):
    h, w = img_bgr.shape[:2]
    r = min(input_size / h, input_size / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    dh = (input_size - new_h) // 2
    dw = (input_size - new_w) // 2
    padded[dh:dh + new_h, dw:dw + new_w] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    blob = np.ascontiguousarray(blob)
    return blob, r, dw, dh


def _yolo_postprocess(output, conf_thresh, iou_thresh, r, dw, dh, orig_h, orig_w):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.ndim == 3:
        output = output[0]

    if output.shape[-1] == 6:
        output_t = output
        if output_t.ndim == 2 and output_t.shape[0] < output_t.shape[1]:
            output_t = output_t.transpose(1, 0)
        scores = output_t[:, 4]
        class_ids = output_t[:, 5].astype(np.int32)
        valid_mask = (scores >= conf_thresh) & (class_ids != 3)
        person_mask = valid_mask & (class_ids == 0)
        if not person_mask.any():
            return np.zeros((0, 6), dtype=np.float32)
        filtered = output_t[person_mask]
        filtered_scores = scores[person_mask]
        filtered_class_ids = class_ids[person_mask]
        cx, cy, bw, bh = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        x1 = (cx - bw / 2 - dw) / r
        y1 = (cy - bh / 2 - dh) / r
        x2 = (cx + bw / 2 - dw) / r
        y2 = (cy + bh / 2 - dh) / r
    else:
        if output.shape[1] != 6:
            output = output.transpose(1, 0)
        class_scores = output[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        max_class_scores = class_scores.max(axis=1)
        obj_scores = max_class_scores
        person_mask = (class_ids == 0) & (obj_scores >= conf_thresh)
        if not person_mask.any():
            return np.zeros((0, 6), dtype=np.float32)
        filtered = output[person_mask]
        filtered_scores = obj_scores[person_mask]
        filtered_class_ids = class_ids[person_mask]
        cx, cy, bw, bh = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        x1 = (cx - bw / 2 - dw) / r
        y1 = (cy - bh / 2 - dh) / r
        x2 = (cx + bw / 2 - dw) / r
        y2 = (cy + bh / 2 - dh) / r

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = _nms(boxes, filtered_scores, iou_thresh)
    result = np.zeros((len(keep), 6), dtype=np.float32)
    result[:, :4] = boxes[keep]
    result[:, 4] = filtered_scores[keep]
    result[:, 5] = filtered_class_ids[keep].astype(np.float32)
    return result


def _nms(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def _vitpose_preprocess(img_bgr, center, scale):
    cropped = crop(img_bgr, center, scale, [VITPOSE_INPUT_W, VITPOSE_INPUT_H])
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_float = (img_float - VITPOSE_MEAN) / VITPOSE_STD
    img_float = img_float.transpose(2, 0, 1)[np.newaxis]
    img_float = np.ascontiguousarray(img_float)
    return img_float


def _draw_body(canvas, keypoints, detect_body=True):
    if not detect_body:
        return
    body_kps = keypoints[:NUM_BODY_KP]
    for i, (a, b) in enumerate(BODY_SKELETON):
        if a >= NUM_BODY_KP or b >= NUM_BODY_KP:
            continue
        if body_kps[a, 2] > 0.3 and body_kps[b, 2] > 0.3:
            color = BODY_COLORS[i % len(BODY_COLORS)]
            pt_a = (int(body_kps[a, 0]), int(body_kps[a, 1]))
            pt_b = (int(body_kps[b, 0]), int(body_kps[b, 1]))
            cv2.line(canvas, pt_a, pt_b, color, 2, cv2.LINE_AA)
    for i in range(NUM_BODY_KP):
        if body_kps[i, 2] > 0.3:
            color = BODY_COLORS[i % len(BODY_COLORS)]
            pt = (int(body_kps[i, 0]), int(body_kps[i, 1]))
            cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)


def _draw_face(canvas, keypoints, detect_face=True):
    if not detect_face:
        return
    face_kps = keypoints[NUM_BODY_KP:NUM_BODY_KP + NUM_FACE_KP]
    for i in range(NUM_FACE_KP):
        if face_kps[i, 2] > 0.3:
            pt = (int(face_kps[i, 0]), int(face_kps[i, 1]))
            cv2.circle(canvas, pt, 2, FACE_COLOR, -1, cv2.LINE_AA)


def _draw_hand(canvas, keypoints, offset, detect_hands=True):
    if not detect_hands:
        return
    hand_kps = keypoints[offset:offset + NUM_HAND_KP]
    for i, (a, b) in enumerate(HAND_SKELETON):
        if a >= NUM_HAND_KP or b >= NUM_HAND_KP:
            continue
        if hand_kps[a, 2] > 0.3 and hand_kps[b, 2] > 0.3:
            color = HAND_COLORS[i % len(HAND_COLORS)]
            pt_a = (int(hand_kps[a, 0]), int(hand_kps[a, 1]))
            pt_b = (int(hand_kps[b, 0]), int(hand_kps[b, 1]))
            cv2.line(canvas, pt_a, pt_b, color, 2, cv2.LINE_AA)
    for i in range(NUM_HAND_KP):
        if hand_kps[i, 2] > 0.3:
            color = HAND_COLORS[i % len(HAND_COLORS)]
            pt = (int(hand_kps[i, 0]), int(hand_kps[i, 1]))
            cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)


def draw_aapose_by_meta_new(canvas, keypoints, detect_body=True, detect_hands=True, detect_face=True):
    _draw_body(canvas, keypoints, detect_body)
    _draw_face(canvas, keypoints, detect_face)
    _draw_hand(canvas, keypoints, NUM_BODY_KP + NUM_FACE_KP, detect_hands)
    _draw_hand(canvas, keypoints, NUM_BODY_KP + NUM_FACE_KP + NUM_HAND_KP, detect_hands)
    return canvas


def _smooth_keypoints_sequence(keypoints_seq, window=5):
    if len(keypoints_seq) < 3 or window < 3:
        return keypoints_seq
    half_w = window // 2
    smoothed = []
    for i in range(len(keypoints_seq)):
        start = max(0, i - half_w)
        end = min(len(keypoints_seq), i + half_w + 1)
        chunk = [k for k in keypoints_seq[start:end] if k is not None]
        if not chunk:
            smoothed.append(keypoints_seq[i])
            continue
        avg = np.zeros_like(chunk[0])
        valid_count = 0
        for c in chunk:
            mask = c[:, 2] > 0.3
            avg[mask, 0] += c[mask, 0]
            avg[mask, 1] += c[mask, 1]
            avg[mask, 2] += c[mask, 2]
            valid_count += mask.sum()
        if valid_count > 0:
            mask = avg[:, 2] > 0
            avg[mask, 0] /= avg[mask, 2]
            avg[mask, 1] /= avg[mask, 2]
            avg[:, 2] = avg[:, 2] / len(chunk)
            avg[:, 2] = np.clip(avg[:, 2], 0, 1)
        smoothed.append(avg)
    return smoothed


class SDPoseBackend:
    def __init__(self, vitpose_model_path, yolo_model_path, device="CUDAExecutionProvider"):
        if ort is None:
            raise ImportError("onnxruntime is required for SDPoseBackend")
        if cv2 is None:
            raise ImportError("opencv-python is required for SDPoseBackend")

        self._vitpose_path = vitpose_model_path
        self._yolo_path = yolo_model_path
        self._device = device

        self._vitpose_session = None
        self._yolo_session = None

        self._init_yolo()
        self._init_vitpose()

    def _get_providers(self):
        available = ort.get_available_providers()
        if self._device in available:
            return [self._device, "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _init_yolo(self):
        if not os.path.isfile(self._yolo_path):
            error(_TAG, "YOLO ONNX model not found: %s", self._yolo_path)
            raise FileNotFoundError(f"YOLO ONNX model not found: {self._yolo_path}")
        providers = self._get_providers()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._yolo_session = ort.InferenceSession(
            self._yolo_path, sess_opts=sess_opts, providers=providers
        )
        info(_TAG, "YOLO model loaded: %s", self._yolo_path)

    def _init_vitpose(self):
        if not os.path.isfile(self._vitpose_path):
            error(_TAG, "ViTPose ONNX model not found: %s", self._vitpose_path)
            raise FileNotFoundError(f"ViTPose ONNX model not found: {self._vitpose_path}")
        providers = self._get_providers()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._vitpose_session = ort.InferenceSession(
            self._vitpose_path, sess_opts=sess_opts, providers=providers
        )
        info(_TAG, "ViTPose model loaded: %s", self._vitpose_path)

    def _detect_person(self, img_bgr, conf_thresh=0.5, iou_thresh=0.45):
        h, w = img_bgr.shape[:2]
        blob, r, dw, dh = _yolo_preprocess(img_bgr)
        input_name = self._yolo_session.get_inputs()[0].name
        output = self._yolo_session.run(None, {input_name: blob})
        dets = _yolo_postprocess(output[0], conf_thresh, iou_thresh, r, dw, dh, h, w)
        if dets.shape[0] == 0:
            return None
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        best_idx = np.argmax(areas)
        return dets[best_idx, :4]

    def _extract_keypoints(self, img_bgr, bbox):
        if bbox is None:
            h, w = img_bgr.shape[:2]
            bbox = np.array([0, 0, w, h], dtype=np.float32)
        center, scale = bbox_from_detector(
            bbox, [VITPOSE_INPUT_W, VITPOSE_INPUT_H], rescale=1.25
        )
        input_tensor = _vitpose_preprocess(img_bgr, center, scale)
        input_name = self._vitpose_session.get_inputs()[0].name
        output = self._vitpose_session.run(None, {input_name: input_tensor})
        heatmaps = output[0]
        if heatmaps.ndim == 3:
            heatmaps = heatmaps[np.newaxis]
        keypoints = keypoints_from_heatmaps(heatmaps, center, scale)
        return keypoints[0]

    def _render_pose_image(self, keypoints, width, height, detect_body=True, detect_hands=True, detect_face=True):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        draw_aapose_by_meta_new(canvas, keypoints, detect_body, detect_hands, detect_face)
        return canvas

    def extract_poses_from_video(
        self,
        video_path,
        target_frames,
        output_width,
        output_height,
        detect_body=True,
        detect_hands=True,
        detect_face=True,
        smooth=True,
        smooth_window=5,
    ):
        if not os.path.isfile(video_path):
            error(_TAG, "Video file not found: %s", video_path)
            return None, "", 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error(_TAG, "Failed to open video: %s", video_path)
            return None, "", 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            error(_TAG, "Video has no frames: %s", video_path)
            return None, "", 0

        info(_TAG, "Processing video: %s, total_frames=%d, target_frames=%d", video_path, total_frames, target_frames)

        if total_frames > target_frames:
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            indices = np.arange(total_frames)

        all_keypoints = []
        frame_count = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                warn(_TAG, "Failed to read frame %d", idx)
                all_keypoints.append(None)
                continue

            resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

            bbox = self._detect_person(resized)
            keypoints = self._extract_keypoints(resized, bbox)

            if keypoints is not None and keypoints.shape[0] == NUM_TOTAL_KP:
                all_keypoints.append(keypoints)
            else:
                all_keypoints.append(None)

            frame_count += 1

        cap.release()

        if smooth and frame_count > 2:
            all_keypoints = _smooth_keypoints_sequence(all_keypoints, smooth_window)

        pose_images = []
        pose_data_list = []

        for i, kps in enumerate(all_keypoints):
            if kps is None:
                kps = np.zeros((NUM_TOTAL_KP, 3), dtype=np.float32)
            canvas = self._render_pose_image(
                kps, output_width, output_height, detect_body, detect_hands, detect_face
            )
            pose_images.append(canvas)
            pose_data_list.append(kps.tolist())

        if torch is not None:
            images_np = np.stack(pose_images, axis=0).astype(np.float32) / 255.0
            pose_tensor = torch.from_numpy(images_np)
        else:
            pose_tensor = np.stack(pose_images, axis=0).astype(np.float32) / 255.0

        pose_data_str = json.dumps(pose_data_list, ensure_ascii=False)

        info(_TAG, "Pose extraction complete: %d frames, resolution=%dx%d", frame_count, output_width, output_height)
        return pose_tensor, pose_data_str, frame_count

    def cleanup(self):
        if self._vitpose_session is not None:
            del self._vitpose_session
            self._vitpose_session = None
        if self._yolo_session is not None:
            del self._yolo_session
            self._yolo_session = None
        info(_TAG, "SDPoseBackend cleaned up")
