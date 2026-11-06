# ♻️ Smart AI Trash Sorting (ระบบคัดแยกขยะรีไซเคิลอัตโนมัติด้วย AI)

ระบบตรวจจับและคัดแยกขยะรีไซเคิล (กล่องเครื่องดื่ม UHT และกระป๋องอลูมิเนียม CAN) แบบอัตโนมัติแบบเรียลไทม์ โดยใช้คอมพิวเตอร์วิทัศน์ร่วมกับโมเดลปัญญาประดิษฐ์ **YOLOv8** ประมวลผลบนบอร์ด **Raspberry Pi** และควบคุมกลไกคัดแยกด้วย **Servo Motor**

---

## 📑 สารบัญ
1. [ภาพรวมของระบบ (Overview)](#-ภาพรวมของระบบ-overview)
2. [โครงสร้างไฟล์ (Repository Structure)](#-โครงสร้างไฟล์-repository-structure)
3. [อุปกรณ์และการต่อวงจร (Hardware & Wiring)](#-อุปกรณ์และการต่อวงจร-hardware--wiring)
4. [รายละเอียดการทำงานของโค้ด (Code Inspection)](#-รายละเอียดการทำงานของโค้ด-code-inspection)
   - [1. สคริปต์เทรนโมเดล (Training Script)](#1-สคริปต์เทรนโมเดล-from-ultralytics-import-yolopy)
   - [2. สคริปต์ควบคุมบน Raspberry Pi (Main Inference Script)](#2-สคริปต์ควบคุมบน-raspberry-pi-code-in-ripy)
5. [การติดตั้งและเตรียมสภาพแวดล้อม (Installation)](#-การติดตั้งและเตรียมสภาพแวดล้อม-installation)
6. [วิธีเริ่มใช้งาน (Usage)](#-วิธีเริ่มใช้งาน-usage)
7. [การปรับแต่งค่าพารามิเตอร์ (Configuration)](#-การปรับแต่งค่าพารามิเตอร์-configuration)

---

## 🔍 ภาพรวมของระบบ (Overview)

ระบบถูกออกแบบมาสำหรับการคัดแยกขยะ 2 ประเภท:
- 🧃 **UHT**: กล่องเครื่องดื่ม/กล่องนม UHT
- 🥫 **CAN**: กระป๋องอลูมิเนียม

### จุดเด่นด้านประสิทธิภาพ (Performance & Optimizations):
- **สถาปัตยกรรม Multi-Threading**: แยกการทำงานเป็น 3 ส่วน (Camera Stream Buffer, AI Worker Thread, และ Main GUI/Servo Loop) ทำให้ภาพจากกล้องลื่นไหล ไม่กระตุกจากรอบการประมวลผลของ AI
- **ปรับแต่งความเร็วสำหรับ Raspberry Pi**:
  - ย่อขนาดภาพอินพุตสำหรับ AI เหลือ `320x320` พิกเซล (ประมวลผลเร็วกว่าขนาดปกติ 640x640 ถึง ~4 เท่า)
  - จำกัด CPU Cores สำหรับ PyTorch ให้ใช้เพียง 2 Threads (`torch.set_num_threads(2)`) ป้องกันไม่ให้ CPU ทำงานเต็ม 100% จนส่งผลกระทบต่อการดึงภาพ
  - ดึงภาพผ่าน V4L2 Backend และตั้งค่า Buffer Size = 1 ป้องกันภาพดีเลย์หรือการค้างของเฟรมเก่า
- **ระบบป้องกันการคัดแยกผิดพลาด (Streak & Debounce Logic)**:
  - กำหนด Confidence ขั้นต่ำไว้ที่ 75% (`CONF_THRESHOLD = 0.75`)
  - ต้องตรวจพบวัตถุชนิดเดียวกันต่อเนื่องอย่างน้อย 3 เฟรมติด (`DETECTION_STREAK_REQUIRED = 3`) เพื่อป้องกัน False Positive จากแสงหรือมุมมองชั่วขณะ
  - มีระบบ Cooldown 3 วินาทีหลังจากปัดขยะ เพื่อป้องกันการสั่งเซอร์โวหมุนซ้ำซ้อน
- **หน้าจอแสดงผล HUD (Head-Up Display)**: แสดงผล FPS แบบ Exponential Moving Average, สถานะระบบ (IDLE, DETECTING, SORTING) และสถิติจำนวนขยะสะสมแบบสด

---

## 📁 โครงสร้างไฟล์ (Repository Structure)

```text
recycle/
├── README.md                          # เอกสารคู่มืออธิบายการทำงานของระบบ
├── from ultralytics import YOLO.py    # สคริปต์สำหรับฝึกสอน (Train) โมเดล YOLOv8 บนคอมพิวเตอร์
└── code in RI.py                      # สคริปต์หลักสำหรับรันบน Raspberry Pi (AI + กล้อง + Servo)
```

> 💡 **คำแนะนำสำหรับการพัฒนา**: เพื่อให้เป็นไปตามมาตรฐาน PEP 8 ในอนาคตสามารถเปลี่ยนชื่อไฟล์ให้กระชับขึ้นได้ เช่น:
> - `from ultralytics import YOLO.py` → `train.py`
> - `code in RI.py` → `main_rpi.py`

---

## 🔌 อุปกรณ์และการต่อวงจร (Hardware & Wiring)

### รายการอุปกรณ์ที่ใช้:
1. **Raspberry Pi 4 หรือ Raspberry Pi 5** (แนะนำแรม 4GB ขึ้นไป)
2. **USB Webcam** หรือ **Raspberry Pi Camera Module**
3. **Servo Motor** (เช่น MG995, MG996R หรือ SG90)
4. **แหล่งจ่ายไฟแยก 5V 2A–3A** สำหรับ Servo Motor (สำคัญมาก: เพื่อป้องกันไม่ให้ไฟ Raspberry Pi ตกขณะมอเตอร์หมุน)

### แผนผังการต่อวงจร Servo Motor:

| ขาของ Servo Motor | เชื่อมต่อไปยัง | หมายเหตุ |
| :--- | :--- | :--- |
| **สายสัญญาณ (สีส้ม/เหลือง)** | **GPIO 18** (Physical Pin 12) | ควบคุมผ่าน PWM Hardware Pin |
| **ไฟเลี้ยง VCC (สีแดง)** | **ขั้วบวก (+) ของแหล่งจ่ายไฟภายนอก 5V** | **ห้าม** ต่อตรงกับ 5V ของบอร์ด Pi หากใช้เซอร์โวขนาดใหญ่ |
| **สายกราวด์ GND (สีน้ำตาล/ดำ)** | **ขั้วลบ (-) ของแหล่งจ่ายไฟภายนอก** | **ต้องต่อ GND ร่วม** กับ GND ของ Raspberry Pi (เช่น Physical Pin 6) |

---

## 🔬 รายละเอียดการทำงานของโค้ด (Code Inspection)

### 1. สคริปต์เทรนโมเดล (`from ultralytics import YOLO.py`)

ใช้สำหรับฝึกสอนโมเดลบนคอมพิวเตอร์ที่มีการ์ดจอ (PC/GPU) ก่อนนำไฟล์ weights ไปรันบน Raspberry Pi:

```python
from ultralytics import YOLO

# 1. โหลดโมเดล Nano (ขนาดกะทัดรัด ประมวลผลได้เร็วที่สุด)
model = YOLO('yolov8n.pt')

# 2. เริ่มการฝึกสอน
results = model.train(
    data=r"C:\AI_recycle\CAN AND MILK 3.0.v1i.yolov8\data.yaml", 
    epochs=50,      # จำนวนรอบฝึกสอน
    imgsz=320,      # ขนาดภาพ 320x320 เหมาะกับสมรรถนะของบอร์ด Pi
    batch=16,       # ปรับตามขนาด VRAM ของการ์ดจอ
    plots=True      # สร้างกราฟสรุปผล (Confusion Matrix, F1-Curve, etc.)
)
```
- ผลลัพธ์ที่ได้จากการเทรนจะอยู่ในโฟลเดอร์ `runs/detect/train/weights/best.pt` ให้นำไฟล์ `best.pt` นี้ไปใช้ใน Raspberry Pi

---

### 2. สคริปต์ควบคุมบน Raspberry Pi (`code in RI.py`)

เป็นสคริปต์หลักที่ประกอบด้วย 4 โมดูลการทำงานสำคัญ:

1. **Servo Control**:
   - ควบคุมการทำงานของมอเตอร์ด้วยสัญญาณ PWM ความถี่ 50Hz
   - มีฟังก์ชัน `move_servo(angle)` แปลงองศา (0 - 180°) เป็นค่า Duty Cycle (`2.5 + angle / 18`)
   - มีการตัดสัญญาณ (`ChangeDutyCycle(0)`) หลังหมุนเสร็จ เพื่อลดความร้อนและป้องกันอาการมอเตอร์สั่นค้าง (Jittering)
2. **CameraBuffer Class (Thread-safe)**:
   - อ่านเฟรมภาพจากกล้องตลอดเวลาใน Background Thread เพื่อทิ้งเฟรมเก่าและเก็บเฉพาะเฟรมล่าสุดไว้ในหน่วยความจำ
3. **AI Worker Thread**:
   - ดึงภาพขนาด 320x320 ไปประมวลผลผ่าน `model(frame, conf=0.75, imgsz=320)`
   - วนลูปเฉพาะตอนที่ Servo ไม่ได้กำลังหมุนปัดขยะ เพื่อประหยัดพลังงาน
4. **Main Loop & Detection Logic**:
   - คำนวณ Streak ตรวจสอบความถูกต้องต่อเนื่อง 3 เฟรม
   - สั่งหมุน Servo ไปยังมุมที่กำหนด:
     - **กล่องนม (UHT)**: หมุนไปที่มุม `170°` แล้วกลับมาที่มุมพัก `85°`
     - **กระป๋อง (CAN)**: หมุนไปที่มุม `0°` แล้วกลับมาที่มุมพัก `85°`
   - บันทึกยอดสะสม (`count_uht`, `count_can`)
   - วาด Bounding Box และ HUD บนหน้าจอ OpenCV

---

## 📦 การติดตั้งและเตรียมสภาพแวดล้อม (Installation)

### สำหรับเครื่อง Raspberry Pi:

1. เปิด Terminal บน Raspberry Pi แล้วอัปเดตระบบ:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3-pip python3-opencv libgl1-mesa-glx
   ```

2. ติดตั้งไลบรารี Python ที่จำเป็น:
   ```bash
   pip3 install ultralytics torch torchvision
   ```
   *(หมายเหตุ: สำหรับ Raspberry Pi 5 หากใช้งาน `RPi.GPIO` แล้วพบข้อผิดพลาด ให้ติดตั้ง `rpi-lgpio` แทน)*

3. Clone โค้ดลงบนบอร์ด:
   ```bash
   git clone https://github.com/EVOhongZAW/recycle.git
   cd recycle
   ```

4. วางไฟล์โมเดล `best.pt` ที่เทรนเสร็จแล้วลงในโฟลเดอร์โปรเจกต์เดียวกันนี้

---

## 🚀 วิธีเริ่มใช้งาน (Usage)

รันสคริปต์บน Raspberry Pi:
```bash
python3 "code in RI.py"
```

### การควบคุมระบบ:
- **กดปุ่ม `Q`** บนคีย์บอร์ด หรือคลิกปิดหน้าต่างแสดงผล เพื่อหยุดการทำงานอย่างปลอดภัย
- ระบบจะสั่งปิดกล้อง, คืนค่าพิน GPIO (`GPIO.cleanup()`), และแสดงรายงานสรุปจำนวนขยะทั้งหมดที่คัดแยกได้ในรอบนั้นบน Terminal

---

## ⚙️ การปรับแต่งค่าพารามิเตอร์ (Configuration)

คุณสามารถปรับแต่งตัวแปรที่ส่วนต้นของไฟล์ `code in RI.py` ได้ตามลักษณะโครงสร้างจริงของถังขยะ:

| ตัวแปร | ค่าเริ่มต้น | รายละเอียด |
| :--- | :---: | :--- |
| `SERVO_PIN` | `18` | ขา GPIO ที่ใช้ต่อสายสัญญาณ Servo |
| `ANGLE_REST` | `85` | องศาตรงกลาง (ตำแหน่งพักของแขนปัด) |
| `ANGLE_UHT` | `170` | องศาเมื่อตรวจพบกล่องนม UHT |
| `ANGLE_CAN` | `0` | องศาเมื่อตรวจพบกระป๋อง CAN |
| `CONF_THRESHOLD` | `0.75` | ค่าความเชื่อมั่นขั้นต่ำ (0.75 = 75%) |
| `DETECTION_STREAK_REQUIRED` | `3` | จำนวนเฟรมที่ต้องตรวจเจอต่อเนื่องก่อนสั่งปัด |
| `COOLDOWN_TIME` | `3` | เวลาหน่วงหลังปัดขยะ (วินาที) |
| `AI_INPUT_SIZE` | `320` | ขนาดภาพที่ส่งให้โมเดล AI วิเคราะห์ |