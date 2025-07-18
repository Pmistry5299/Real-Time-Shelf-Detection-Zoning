import cv2
import numpy as np
from collections import deque


cap = cv2.VideoCapture("http://192.168.2.24:4747/video")
# Fallback to webcam if IP failsAC
if not cap.isOpened():
    print("[ERROR] Failed to open IP stream. Trying webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] No camera found. Exiting.")
        exit()

# Set camera resolution
image_width, image_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

shelf_history = deque(maxlen=15)

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (100, 255, 100), (255, 100, 100)
]

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_horizontal_lines(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=region.shape[1] // 3, maxLineGap=10)

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:  # Almost horizontal
                horizontal_lines.append((x1, y1, x2, y2))
    return horizontal_lines

def is_shelf(contour, frame, min_width=150, max_angle=15):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) != 4:
        return False

    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90

    aspect_ratio = w / h if h != 0 else 0
    if not (w > min_width and 1.0 < aspect_ratio < 15.0 and abs(angle) < max_angle):
        return False

    # Extract region of interest (ROI) to check for shelf levels
    x_int, y_int, w_int, h_int = cv2.boundingRect(contour)
    roi = frame[y_int:y_int + h_int, x_int:x_int + w_int]

    # Count horizontal lines
    lines = detect_horizontal_lines(roi)
    return len(lines) >= 1  # Shelf should have 2 or more levels

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def draw_zones(frame, shelves):
    for idx, (x, y, w, h) in enumerate(shelves):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        num_divs = 3 if idx % 2 == 0 else 4
        portion_width = w // num_divs

        for i in range(num_divs):
            px1 = x + i * portion_width
            px2 = x + (i + 1) * portion_width if i < num_divs - 1 else x + w
            color = colors[i % len(colors)]
            overlay = frame.copy()
            cv2.rectangle(overlay, (px1, y), (px2, y + h), color, -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.rectangle(frame, (px1, y), (px2, y + h), (0, 0, 0), 2)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not received.")
        break

    frame = cv2.resize(frame, (image_width, image_height))
    processed = preprocess_frame(frame)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        if is_shelf(contour, frame):
            x, y, w, h = cv2.boundingRect(contour)
            detected.append((x, y, w, h))

    shelf_history.append(detected)
    all_boxes = [box for frame_boxes in shelf_history for box in frame_boxes]

    grouped_boxes = []
    while all_boxes:
        base = all_boxes.pop(0)
        group, rest = [base], []
        for b in all_boxes:
            if iou(base, b) > 0.4:
                group.append(b)
            else:
                rest.append(b)
        all_boxes = rest
        if len(group) >= len(shelf_history) // 2:
            avg_box = tuple(map(lambda l: int(np.mean(l)), zip(*group)))
            grouped_boxes.append(avg_box)

    stable_result = sorted(grouped_boxes, key=lambda b: b[1])
    output = draw_zones(frame.copy(), stable_result)

    cv2.putText(output, f"Detected shelves: {len(stable_result)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Shelf Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
