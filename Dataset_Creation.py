import cv2
import mediapipe as mp
import os
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

base_path = "augmented_dataset" 
all_data = []

for label in os.listdir(base_path):
    folder_path = os.path.join(base_path, label)
    if not os.path.isdir(folder_path): continue
    
    print(f"Extracting landmarks for: {label}...")
    
    for img_name in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_name))
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:

                wrist_x = hand_lms.landmark[0].x
                wrist_y = hand_lms.landmark[0].y
                
                row = [label]
                for lm in hand_lms.landmark:
                    row.append(lm.x - wrist_x)
                    row.append(lm.y - wrist_y) 
                
                all_data.append(row)

columns = ['label'] + [f'pt{i}_{axis}' for i in range(21) for axis in ['x', 'y']]
df = pd.DataFrame(all_data, columns=columns)
df.to_csv("gesture_dataset_500_3.csv", index=False)

print(f"Success! Created CSV with {len(df)} rows.")