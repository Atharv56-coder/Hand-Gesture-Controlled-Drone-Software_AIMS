import numpy as np
import pandas as pd

def normalize_landmarks(landmark_list):
    
    temp_list = np.array(landmark_list)
    
    base_x = temp_list[0]
    base_y = temp_list[1]
    
    for i in range(0, len(temp_list), 2):
        temp_list[i] = temp_list[i] - base_x
        temp_list[i+1] = temp_list[i+1] - base_y
        
    max_val = np.max(np.abs(temp_list))
    if max_val != 0:
        temp_list = temp_list / max_val
        
    return temp_list

data = pd.read_csv('new_gesture_dataset_500_2.csv') 

labels = data.iloc[:, 0].values
features = data.iloc[:, 1:].values

normalized_features = []

for row in features:
    norm_row = normalize_landmarks(row)
    normalized_features.append(norm_row)

df_norm = pd.DataFrame(normalized_features)
df_norm.insert(0, 'label', labels)
df_norm.to_csv('normalized_landmarks_2.csv', index=False)

print("Normalization complete! File saved as 'normalized_landmarks.csv'")