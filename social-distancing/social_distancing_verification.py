import numpy as np
import cv2
import math
import tensorflow as tf
import itertools as iter
import sys


def resize_box(box, h, w):
    return tuple(np.array([box[0]*h, box[1]*w, box[2]*h, box[3]*w]).astype(int))

def detect(model, _image, CONF_THR = 0.50):
    image = _image[np.newaxis, ...].copy()
    h,w,_ = _image.shape
    results = model(image)
    result = {key:value.numpy() for key,value in results.items()}
    classes = result["detection_classes"].astype(int)
    scores = result["detection_scores"]
    mask = np.logical_and(scores >= CONF_THR, classes == 1)
    scores = scores[mask]
    boxes = result["detection_boxes"][mask]
    boxes = [resize_box(box, h, w) for box in boxes]
    classes = classes[mask]
    return boxes, scores, classes

def decode_boxes(boxes):
    return [[b[1], b[0], b[3], b[2]] for b in boxes]
    
def predict(model, images, conf_thr = 0.50):
    boxes = []
    scores = []
    classes = []
    for image in images:
        b, s, c = detect(model, image, conf_thr)
        boxes.append(decode_boxes(b))
        scores.append(s)
        classes.append(c)
    return boxes, scores, classes

def get_bbox_centroid(box):
    y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
    return ((x1+x2)/2, y2)
  
def map_centroid(centroid, matrix):
    return cv2.perspectiveTransform(centroid, matrix)

def compute_bbox_centroids(detection_boxes):
    return [get_bbox_centroid(box) for box in detection_boxes]
  

def compute_bird_eye_centroids(detection_boxes, matrix):
    detection_centroids = compute_bbox_centroids(detection_boxes)
    mapped_centroids = [map_centroid(np.array([np.array([centroid])]), matrix) for centroid in detection_centroids]
    mapped_centroids = [(int(centroid[0][0][0]), int(centroid[0][0][1])) for centroid in mapped_centroids]
    return mapped_centroids

def compute_safe_distance_in_px(real_width, target_width, SAFE_DISTANCE_THR_METER = 1):
    safe_distance = (target_width * SAFE_DISTANCE_THR_METER)/real_width
    return safe_distance

def distance_between(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def check_centroids_distancing(frame_centroids, real_width, target_width = 800):
    checked_centroids = []
    mask = []
    couples = [couple for couple in iter.combinations(frame_centroids, 2)]
    for couple in couples:
        distance = distance_between(couple[0], couple[1])
        violation = distance <= compute_safe_distance_in_px(real_width, target_width)
        checked_centroids.append((couple, violation))
        mask.append(violation)

    return checked_centroids, mask

def check_boxes_distancing(frame_boxes, mask):
    checked_boxes = []
    red = []
    couples = [couple for couple in iter.combinations(frame_boxes, 2)]
    for i, couple in enumerate(couples):
        if mask[i]:
            red.append([couple[0][0], couple[0][1], couple[0][2], couple[0][3]])
            red.append([couple[1][0], couple[1][1], couple[1][2], couple[1][3]])
    for b in frame_boxes:
        checked_boxes.append((b, (0, 0, 255) if b in red else (0, 255, 0)))
    return checked_boxes

def draw_checked_boxes_on_frame(frame, checked_boxes):
    _img = frame.copy()
    for b in checked_boxes:
        start_point = (b[0][0], b[0][1])
        end_point = (b[0][2], b[0][3])
        cv2.rectangle(_img, start_point, end_point, b[1], 3)
    return _img

def get_video_frames(video_path, rotate_right=False):
  vid = cv2.VideoCapture(video_path)
  frames = []
  while(vid.isOpened()):
    ret, fr = vid.read()
    if ret == True:
      if rotate_right:
        fr = cv2.rotate(fr, cv2.cv2.ROTATE_90_CLOCKWISE)
      frames.append(fr)
    else:
      break

  vid.release()
  return frames

def save_video(frames, video_path="video.avi", TARGET_W = 800, TARGET_H = 450):
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (TARGET_W, TARGET_H))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

model_name = "models/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8"
path_video = sys.argv[1]
matrix_path = "perspective_matrix.npy"

print("Extracting video frames...")
frames = get_video_frames(path_video)
print("Extracted franes: "+str(len(frames)))
h, w = frames[0].shape[:2]
print("Loading model...")
model = tf.saved_model.load(model_name+"/saved_model")
matrix = np.load(matrix_path)
print("Starting Human Detection...")
boxes, s, c = predict(model, frames)
print("Starting social distancing verification...")
bird_eye_centroids = [compute_bird_eye_centroids(frame_boxes, matrix) for frame_boxes in boxes]
masks = [check_centroids_distancing(centroids, 9, w)[1] for centroids in bird_eye_centroids]
checked_boxes = [check_boxes_distancing(frame_boxes, mask) for frame_boxes, mask in zip(boxes, masks)]
checked_frames = [draw_checked_boxes_on_frame(f, b) for f, b in zip(frames, checked_boxes)]
print("Saving output video...")
save_video(checked_frames,"checked_video.avi", TARGET_W=w, TARGET_H=h)
