import cv2
import numpy as np
import serial
import time

drawing = False
start_point = None
end_point = None
tracker = None
tracking_active = False

SERIAL_PORT = "/dev/tty.usbmodemXXXX"  # CHANGE THIS
BAUD_RATE = 115200
ser = None

def select_roi_callback(event, x, y, flags, param):
    """Mouse callback for drawing selection box"""
    global drawing, start_point, end_point, tracker, tracking_active
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        tracking_active = False
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        
        x1, y1 = start_point
        x2, y2 = end_point

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if width > 10 and height > 10:  
            bbox = (x_min, y_min, width, height)
            frame = param['frame']
            init_tracker(frame, bbox)
            print(f"✓ Tracking object at: ({x_min}, {y_min}, {width}, {height})")

def init_tracker(frame, bbox):
    """Initialize the object tracker"""
    global tracker, tracking_active
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    tracking_active = True
    print("🎯 Tracker initialized!")

def draw_crosshair(frame, center, size=20, color=(0, 255, 0), thickness=2):
    """Draw a crosshair at the tracking point"""
    x, y = center
    
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    
    cv2.circle(frame, center, 5, color, -1)
    cv2.circle(frame, center, 5, (255, 255, 255), 1)

def main():
    global drawing, start_point, end_point, tracker, tracking_active, ser
    
    # ---------- SERIAL INIT ----------
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)
        print("🔥 Serial connected")
    except:
        print("⚠️ Serial not connected (check port)")
        ser = None
    # -------------------------------
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    window_name = "Object Tracker - Draw box around object"
    cv2.namedWindow(window_name)
    
    callback_params = {'frame': None}
    cv2.setMouseCallback(window_name, select_roi_callback, callback_params)
    
    print("=" * 60)
    print("INTERACTIVE OBJECT TRACKER")
    print("=" * 60)
    print("Instructions:")
    print("  1. Click and drag to draw a box around any object")
    print("  2. Release to start tracking")
    print("  3. Press 'r' to reset and select new object")
    print("  4. Press 'q' to quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        callback_params['frame'] = frame.copy()
        
        if tracking_active and tracker is not None:
            success, bbox = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2
                center_y = y + h // 2
                
                frame_h, frame_w, _ = frame.shape
                
                error_x = (center_x - frame_w / 2) / (frame_w / 2)
                error_y = (center_y - frame_h / 2) / (frame_h / 2)
                
                MAX_DEFLECTION = 30
                
                fin_left   = int(-error_x * MAX_DEFLECTION)
                fin_right  = int(error_x * MAX_DEFLECTION)
                fin_top    = int(-error_y * MAX_DEFLECTION)
                fin_bottom = int(error_y * MAX_DEFLECTION)
                
                print(f"🎯 Simulated fin angles: Left={fin_left}, Right={fin_right}, Top={fin_top}, Bottom={fin_bottom}")
                
                if ser:
                    command = f"<{fin_left},{fin_right},{fin_top},{fin_bottom}>\n"
                    ser.write(command.encode())
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                draw_crosshair(frame, (center_x, center_y), size=30, color=(0, 255, 0), thickness=2)
                
                cv2.putText(frame, "TRACKING", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "TRACKING LOST - Draw new box", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tracking_active = False
        
        if drawing and start_point and end_point:
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.putText(frame, "Release to track", (start_point[0], start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if not tracking_active and not drawing:
            cv2.putText(frame, "Click and drag to select object", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracking_active = False
            tracker = None
            drawing = False
            start_point = None
            end_point = None
    
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    print("✓ Done!")

if __name__ == "__main__":
    main()