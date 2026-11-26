import cv2
import torch
import numpy as np

# โหลดโมเดล YOLOv5
model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path='runs/train/exp2/weights/best.pt')  # ระบุ path ของโมเดล

# เริ่มต้นการเปิดกล้อง
cap = cv2.VideoCapture(0)  # ใช้กล้องตัวที่ 0 (สามารถเปลี่ยนเป็นกล้องอื่นได้ เช่น 1, 2...)

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break

    # การตรวจจับวัตถุโดยใช้ YOLOv5
    results = model(frame)

    # ผลลัพธ์การตรวจจับ
    people_count = 0
    boxes = results.xywh[0]  # Bounding boxes (x, y, width, height)
    confidences = results.conf[0]  # ความมั่นใจ (confidence)
    class_ids = results.names  # ค่าชื่อ class (ex. "person")

    # ตรวจจับทุกคน (class id 0 คือนิยามของคนใน COCO dataset)
    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
        if confidence > 0.5:  # กำหนดค่าความมั่นใจต่ำสุดที่ 0.5
            if int(results.pred[0][i][5]) == 0:  # ถ้าเป็น class "person" (class id 0)
                people_count += 1

                # แสดงกรอบ (bounding box) และ label (label: คน, confidence)
                x, y, w, h = box.tolist()  # ตำแหน่งและขนาดของ bounding box
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # แสดงจำนวนคนที่มุมซ้ายบน
    cv2.putText(frame, f"People Count: {people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงผลลัพธ์
    cv2.imshow("YOLOv5 Detection", frame)

    # กด 'q' เพื่อปิดโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
