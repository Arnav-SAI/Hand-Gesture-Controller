
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

try:
    import mediapipe.python.solutions as mp_solutions
    mp.solutions = mp_solutions
except ImportError:
    pass

FRAME_MARGIN = 100
SMOOTHING_FACTOR = 5
CLICK_THRESHOLD = 40
DOUBLE_CLICK_COOLDOWN = 0.5

COLOR_MOUSE_ACTIVE = (0, 255, 0)
COLOR_CLICK = (0, 0, 255)
COLOR_SCROLL = (255, 0, 255)
COLOR_NEUTRAL = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)

pyautogui.FAILSAFE = True

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                    )
        return img

    def find_position(self, img, hand_no=0):
        lm_list = []
        x_list = []
        y_list = []
        bbox = []
        
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape
            
            for id, lm in enumerate(my_hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                x_list.append(cx)
                y_list.append(cy)
            
            if x_list:
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = xmin, ymin, xmax, ymax
                
        return lm_list, bbox

    def fingers_up(self, lm_list):
        fingers = []
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]: 
             fingers.append(True)
        else:
             fingers.append(False)

        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)
        return fingers

class GestureProcessor:
    def __init__(self):
        pass

    def get_gesture(self, lm_list, fingers):
        if not lm_list:
            return "IDLE", None, None

        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        x_thumb, y_thumb = lm_list[4][1:]

        dist_idx_thumb = math.hypot(x1 - x_thumb, y1 - y_thumb)
        dist_mid_thumb = math.hypot(x2 - x_thumb, y2 - y_thumb)

        if middle_up and dist_mid_thumb < CLICK_THRESHOLD:
            gesture = "RCLICK"
            info = {'x': x2, 'y': y2} 

        elif index_up and dist_idx_thumb < CLICK_THRESHOLD:
            gesture = "CLICK"
            info = {'x': x1, 'y': y1}

        elif index_up and middle_up:
            gesture = "SCROLL"
            info = {'x': x1, 'y': y1} 

        elif index_up and not middle_up:
            gesture = "MOVE"
            info = {'x': x1, 'y': y1}
        
        else:
            gesture = "IDLE"
            info = {'x': x1, 'y': y1}

        return gesture, info

class SystemController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.smoothing = SMOOTHING_FACTOR
        self.last_click_time = 0
    
    def apply_smoothing(self, target_x, target_y):
        self.curr_x = self.prev_x + (target_x - self.prev_x) / self.smoothing
        self.curr_y = self.prev_y + (target_y - self.prev_y) / self.smoothing
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        return self.curr_x, self.curr_y

    def map_coords(self, x, y):
        x_mapped = np.interp(x, [FRAME_MARGIN, CAM_W - FRAME_MARGIN], [0, SCREEN_W])
        y_mapped = np.interp(y, [FRAME_MARGIN, CAM_H - FRAME_MARGIN], [0, SCREEN_H])
        return x_mapped, y_mapped

    def execute(self, gesture, info):
        if gesture == "IDLE" or not info:
            return

        raw_x, raw_y = info['x'], info['y']
        screen_x, screen_y = self.map_coords(raw_x, raw_y)
        smooth_x, smooth_y = self.apply_smoothing(screen_x, screen_y)

        if gesture == "MOVE":
            pyautogui.moveTo(smooth_x, smooth_y)

        elif gesture == "CLICK":
            if time.time() - self.last_click_time > DOUBLE_CLICK_COOLDOWN:
                pyautogui.click()
                self.last_click_time = time.time()

        elif gesture == "RCLICK":
            if time.time() - self.last_click_time > DOUBLE_CLICK_COOLDOWN:
                pyautogui.rightClick()
                self.last_click_time = time.time()

        elif gesture == "SCROLL":
            active_h = CAM_H - 2 * FRAME_MARGIN
            center_y = FRAME_MARGIN + active_h // 2
            
            dy = raw_y - center_y
            if abs(dy) > 20:
                # Scroll amount
                pyautogui.scroll(int(-dy * 0.1)) 

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Please check your permissions (Terminal needs Camera access).")
        return

    cap.set(3, CAM_W)
    cap.set(4, CAM_H)

    detector = HandDetector(max_hands=1)
    processor = GestureProcessor()
    controller = SystemController()

    p_time = 0

    print("AI Hand Controller Started. Press 'q' to exit.")
    
    success, img = cap.read()
    if not success:
         print("Error: Camera is accessible but returned no frame.")
         print("Check if another app is using the camera.")
         return

    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)

        if np.mean(img) < 5:
            cv2.putText(img, "CAMERA BLOCKED?", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(img, "Check Mac Permissions", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        img = detector.find_hands(img)
        lm_list, bbox = detector.find_position(img)

        current_gesture = "IDLE"
        if lm_list:
            fingers = detector.fingers_up(lm_list)
            current_gesture, info = processor.get_gesture(lm_list, fingers)
            
            try:
                controller.execute(current_gesture, info)
            except pyautogui.FailSafeException:
                print("FailSafe triggered from corner. Exiting...")
                break

            cv2.rectangle(img, (FRAME_MARGIN, FRAME_MARGIN), 
                          (CAM_W - FRAME_MARGIN, CAM_H - FRAME_MARGIN), 
                          (255, 0, 255), 2)
            
            if info:
                cx, cy = info['x'], info['y']
                color = COLOR_NEUTRAL
                if current_gesture == "MOVE": color = COLOR_MOUSE_ACTIVE
                elif current_gesture == "CLICK": color = COLOR_CLICK
                elif current_gesture == "RCLICK": color = COLOR_CLICK
                elif current_gesture == "SCROLL": color = COLOR_SCROLL
                
                cv2.circle(img, (cx, cy), 15, color, cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        cv2.putText(img, f'Mode: {current_gesture}', (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

        cv2.imshow("AI Hand Controller", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
