import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
tip_ids = [8, 12, 16, 20]

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    current_action = "STOP"
    speed = 0

    if results.multi_hand_landmarks:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            score = handedness.classification[0].score
            lm = hand_lms.landmark
            fingers = []

            if hand_label == "Left":
                thumb_open = lm[4].x > lm[2].x 
                if thumb_open: fingers.append(1)
                else: fingers.append(0)

                for tip in tip_ids:
                    if lm[tip].y < lm[tip - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)

                if total_fingers == 0:
                    current_action = "STOP"
            
                elif total_fingers == 1:
                    if fingers[0] == 1:
                        if lm[4].y < lm[0].y: current_action = "MOVE UP"
                        else: current_action = "MOVE DOWN"
                    else: current_action = "FORWARD"
                
                elif total_fingers == 2: current_action = "BACKWARD"
                elif total_fingers == 3: current_action = "LEFT"
                elif total_fingers == 4: current_action = "RIGHT"
            
                elif total_fingers == 5:
                    current_action = "SPEED CONTROL ACTIVE"
                    dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y) * w
                    speed = int(np.interp(dist, [30, 200], [0, 100]))
                    cv2.line(img, (int(lm[4].x*w), int(lm[4].y*h)), (int(lm[8].x*w), int(lm[8].y*h)), (0, 255, 0), 3)

                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            else:
                color = (0, 0, 255) 

    cv2.putText(img, f"ACTION: {current_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if speed > 0:
        cv2.putText(img, f"SPEED: {speed}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()