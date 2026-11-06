import cv2
import sys
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import threading
# 🌟 จำกัด CPU cores สำหรับ AI เพื่อไม่แย่งกล้อง
import torch
torch.set_num_threads(2)
# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SERVO_PIN = 18
# 🎯 หาก Servo กลับมาตรงกลางแล้วชี้เอียงซ้าย/ขวา ให้ลองปรับค่า ANGLE_REST ดูครับ
# เช่น ลองเปลี่ยนจาก 90 เป็น 80, 85 หรือ 95, 100 จนกว่าจะตรงเป๊ะ
ANGLE_REST = 85  # <--- ลองเปลี่ยนค่าตรงนี้เพื่อจูนให้ตรงกลางพอดี
ANGLE_UHT  = 170
ANGLE_CAN  = 0
CONF_THRESHOLD = 0.75
DETECTION_STREAK_REQUIRED = 3
COOLDOWN_TIME = 3
SHOW_WINDOW = True
# 🔧 ปรับขนาดภาพให้เล็กลงตอนส่งให้ AI (ช่วยเพิ่ม FPS มาก!)
AI_INPUT_SIZE = 320          # ลดจาก default 640 → 320
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# ==========================================
# 🔌 GPIO SETUP
# ==========================================
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(7.5)
time.sleep(1)
pwm.ChangeDutyCycle(0)
def move_servo(angle):
    """สั่ง Servo หมุนไปที่มุมที่กำหนด"""
    # เปลี่ยนจาก 2 เป็น 2.5 เพื่อให้ 90 องศาตกที่ Duty Cycle 7.5 เป๊ะตามค่าเริ่มต้น
    duty = 2.5 + (angle / 18)

    # ปรับใช้เฉพาะคำสั่ง PWM ป้องกันการแทรกแซงจาก GPIO.output ปกติ
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)
# ==========================================
# 📸 CAMERA THREAD (ปรับปรุงให้ Thread-safe)
# ==========================================
class CameraBuffer:
    """
    กล้องทำงานใน Thread แยก ดึงภาพตลอดเวลา
    ใช้ Lock ป้องกัน race condition ระหว่าง Thread
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # ใช้ V4L2 backend ตรงๆ เร็วกว่า
        # ตั้งค่ากล้อง
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        # ลด buffer ของกล้อง → ได้ภาพล่าสุดเสมอ ไม่ใช่เฟรมเก่าค้าง
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    def _update(self):
        """ดึงภาพจากกล้องตลอดเวลาใน background"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.ret = ret
                    self.frame = frame
            # ไม่ต้อง sleep เลย → ดึงภาพเร็วที่สุดเท่าที่กล้องจะให้ได้
    def read(self):
        """อ่านเฟรมล่าสุด (Thread-safe)"""
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)
    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()
# ==========================================
# 🧠 AI ENGINE (ปรับปรุงให้เร็วขึ้น)
# ==========================================
print("🧠 Loading AI model...")
model = YOLO("best.pt")
# 🔥 Warm-up: ให้โมเดลรันรอบแรกก่อน จะได้ไม่ค้างตอนเริ่ม
dummy = torch.zeros(1, 3, AI_INPUT_SIZE, AI_INPUT_SIZE)
model.predict(dummy, verbose=False)
print("✅ AI Model warmed up!")
cam = CameraBuffer()
if not cam.ret:
    print("❌ Camera not detected")
    cam.stop()
    GPIO.cleanup()
    sys.exit(1)
print("✅ System Ready (Press Q to quit)")
move_servo(ANGLE_REST)
# ==========================================
# 🔒 SHARED STATE (ใช้ Lock ป้องกัน race condition)
# ==========================================
state_lock = threading.Lock()
ai_results_boxes = []
detected_class = None
ai_updated = False
is_moving = False
system_running = True
count_uht = 0
count_can = 0
status_text = "IDLE"
# ==========================================
# 🧵 AI WORKER THREAD
# ==========================================
def ai_worker():
    """
    Thread สำหรับ AI วิเคราะห์ภาพ
    - ใช้ imgsz เล็กลง (320) เพื่อลดเวลาประมวลผล
    - ข้ามเฟรมถ้า servo กำลังเคลื่อนที่
    """
    global ai_results_boxes, detected_class, ai_updated
    while system_running:
        # ข้ามถ้า servo กำลังเคลื่อนที่
        with state_lock:
            if is_moving:
                time.sleep(0.1)
                continue
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        # 🚀 ใช้ imgsz=320 แทน default 640 → เร็วขึ้น ~4 เท่า!
        results = model(frame, conf=CONF_THRESHOLD, imgsz=AI_INPUT_SIZE, verbose=False)
        best_conf = 0
        temp_class = None
        temp_boxes = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            temp_boxes.append((x1, y1, x2, y2, name, conf))
            if conf > best_conf:
                best_conf = conf
                temp_class = name
        # อัปเดตผลลัพธ์แบบ Thread-safe
        with state_lock:
            ai_results_boxes = temp_boxes
            detected_class = temp_class
            ai_updated = True
        # หน่วงนิดหน่อยให้ CPU พัก
        time.sleep(0.03)
ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()
# ==========================================
# ⚙️ SERVO ACTION
# ==========================================
def sort_trash(angle, trash_type):
    """สั่ง Servo คัดแยกขยะ"""
    global is_moving, count_uht, count_can, status_text
    with state_lock:
        is_moving = True
        status_text = f"SORTING → {trash_type}"
    move_servo(angle)
    with state_lock:
        if trash_type == "UHT":
            count_uht += 1
        elif trash_type == "CAN":
            count_can += 1
    move_servo(ANGLE_REST)
    with state_lock:
        status_text = "IDLE"
        is_moving = False
# ==========================================
# 🚀 MAIN LOOP (แสดงผลภาพ + ตรรกะการคัดแยก)
# ==========================================
last_action_time = 0
detection_streak = 0
last_detected_class = None
prev_time = time.time()
fps_smooth = 0  # FPS แบบ smoothed ไม่กระตุก
try:
    while system_running:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        current_time = time.time()
        # === ตรวจสอบผลจาก AI ===
        with state_lock:
            _ai_updated = ai_updated
            _detected_class = detected_class
            _is_moving = is_moving
            _boxes = list(ai_results_boxes)
            _status = status_text
            _cnt_uht = count_uht
            _cnt_can = count_can
            if _ai_updated:
                ai_updated = False
        if _ai_updated and not _is_moving:
            if current_time - last_action_time <= COOLDOWN_TIME:
                detection_streak = 0  # รีเซ็ตหรือหยุดนับ streak หากยังอยู่ในช่วง Cooldown
            elif _detected_class == last_detected_class and _detected_class is not None:
                detection_streak += 1
            else:
                detection_streak = 0
            last_detected_class = _detected_class
            # สั่งคัดแยก
            if detection_streak >= DETECTION_STREAK_REQUIRED:
                if _detected_class == "UHT":
                    threading.Thread(target=sort_trash, args=(ANGLE_UHT, "UHT"), daemon=True).start()
                elif _detected_class == "CAN":
                    threading.Thread(target=sort_trash, args=(ANGLE_CAN, "CAN"), daemon=True).start()
                last_action_time = current_time
                detection_streak = 0
            elif _detected_class is not None:
                with state_lock:
                    status_text = "DETECTING"
                _status = "DETECTING"
            else:
                with state_lock:
                    status_text = "IDLE"
                _status = "IDLE"
        # === วาดภาพ ===
        if SHOW_WINDOW:
            annotated = frame
            # วาด Bounding Box
            if not _is_moving:
                for (x1, y1, x2, y2, name, conf) in _boxes:
                    # สีตามประเภท
                    color = (0, 200, 255) if name == "UHT" else (255, 100, 0) if name == "CAN" else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {conf:.0%}"
                    # พื้นหลังสำหรับข้อความ
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw, max(0, y1 - 2)), color, -1)
                    cv2.putText(annotated, label, (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # คำนวณ FPS แบบ Smoothed (ไม่กระตุก)
            dt = time.time() - prev_time + 1e-6
            prev_time = time.time()
            instant_fps = 1.0 / dt
            fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1  # Exponential Moving Average
            # === HUD (Head-Up Display) ===
            # พื้นหลังกึ่งโปร่งใส
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 5), (250, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, f"Status: {_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {int(fps_smooth)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated, f"UHT: {_cnt_uht}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(annotated, f"CAN: {_cnt_can}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(annotated, f"Streak: {detection_streak}/{DETECTION_STREAK_REQUIRED}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Smart AI Trash Sorting", annotated)
            # รองรับการกดปิดหน้าต่างด้วยปุ่ม X ของ OpenCV (หรือกด Q)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Smart AI Trash Sorting", cv2.WND_PROP_VISIBLE) < 1:
                break
except KeyboardInterrupt:
    print("\n🛑 Stopping system...")
finally:
    system_running = False
    cam.stop()
    pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("✅ System Shutdown Complete")
    print(f"📊 Total sorted → UHT: {count_uht} | CAN: {count_can}")