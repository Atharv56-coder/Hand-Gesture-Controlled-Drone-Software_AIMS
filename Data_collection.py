import cv2 
import os   

base_path = "my_custom_gestures"
os.makedirs(base_path, exist_ok=True) 

gesture_name = input("Enter the name of the gesture you are capturing: ")
save_path = os.path.join(base_path, gesture_name)
os.makedirs(save_path, exist_ok=True) 

cap = cv2.VideoCapture(0)
count = 0 

print(f"Capturing for: {gesture_name}. Press 's' to save, 'q' to quit.")

while True:
    success, img = cap.read() 
    if not success: break     

    img = cv2.flip(img, 1)

    cv2.putText(img, f"Captured: {count}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Data Collector", img) 

    key = cv2.waitKey(1) 
    
    if key == ord('s'): 
        img_filename = os.path.join(save_path, f"{gesture_name}_{count}.jpg")
        cv2.imwrite(img_filename, img)
        count += 1
        print(f"Saved {img_filename}")

    if key == ord('q') or count >= 500:
        break

cap.release()
cv2.destroyAllWindows() 