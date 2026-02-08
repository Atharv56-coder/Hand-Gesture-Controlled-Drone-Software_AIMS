import cv2
import mediapipe as mp
import math
import numpy as np
from tensorflow.keras.models import load_model # Added

model = load_model('hand_gesture_model_2.h5')
actions = ["BACKWARD", "FORWARD", "LEFT", "DOWN", "UP", "NONE", "RIGHT", "SPEED", "STOP"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    current_action = "NONE"
    speed = 0

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            temp_list = []
            for lm in hand_lms.landmark:
                temp_list.extend([lm.x, lm.y])

            base_x, base_y = temp_list[0], temp_list[1]
            normalized_features = []
            for i in range(0, len(temp_list), 2):
                normalized_features.append(temp_list[i] - base_x)
                normalized_features.append(temp_list[i+1] - base_y)

            max_val = max(map(abs, normalized_features))
            if max_val != 0:
                normalized_features = [n / max_val for n in normalized_features]

            prediction = model.predict(np.array([normalized_features]), verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.7:
                current_action = actions[class_id]

            # SPEED LOGIC
            if current_action == "SPEED":
                lm = hand_lms.landmark
                dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y) * w
                speed = int(np.interp(dist, [30, 200], [0, 100]))
                cv2.line(img, (int(lm[4].x*w), int(lm[4].y*h)), (int(lm[8].x*w), int(lm[8].y*h)), (0, 255, 0), 3)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # print(landmarks_features[:4])

    cv2.putText(img, f"ACTION: {current_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    if speed > 0:
        cv2.putText(img, f"SPEED: {speed}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("CNN Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()