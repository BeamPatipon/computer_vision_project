import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO('best.pt')  
class_list = model.names  

cap = cv2.VideoCapture('test2.mp4')

lane_lines = [
    ((280, 600), (480, 600)),  
    ((485, 600), (660, 600)),  
    ((665, 600), (860, 600)),  
    ((870, 600), (1050, 600)),  
    ((1060, 600), (1250, 600))  
]

lane_counts = [defaultdict(int) for _ in range(len(lane_lines))]
crossed_ids = [set() for _ in range(len(lane_lines))]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080)) 

    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])

    for (x1, y1), (x2, y2) in lane_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  
            class_name = class_list[class_idx]

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for i, ((lx1, ly), (lx2, _)) in enumerate(lane_lines):
                if lx1 < cx < lx2 and cy > ly and track_id not in crossed_ids[i]:  
                    crossed_ids[i].add(track_id)
                    lane_counts[i][class_name] += 1

    y_offset = 30
    for i, lane_count in enumerate(lane_counts):
        cv2.putText(frame, f"Lane {i+1}:", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y_offset += 30
        for class_name, count in lane_count.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

    cv2.imshow("YOLO Object Tracking & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
