# Hand-Gesture Controlled Drone (AIMS) 🚁🖐️
An intelligent drone control system that translates real-time hand gestures into flight commands using Computer Vision and Deep Learning. This project uses MediaPipe for landmark extraction and a Custom CNN for gesture classification.

📌 Project Overview
The goal of this software is to provide a touchless, intuitive interface for drone navigation. By mapping 21 hand landmarks into a normalized coordinate system, the model can accurately identify flight commands regardless of hand size or position in the frame.

Key Features
Real-time Gesture Recognition: Sub-30ms latency for command processing.

Normalized Landmark Data: Invariant to hand distance (scaling) and position (translation).

Proportional Speed Control: Uses the distance between thumb and index finger to adjust drone velocity.

9 Distinct Commands: * FORWARD, BACKWARD, LEFT, RIGHT

UP, DOWN, STOP (Emergency Hover)

SPEED (Dynamic Adjustment), NONE (Idle)

🛠️ Tech Stack
Language: Python 3.8+

Computer Vision: OpenCV, MediaPipe

Deep Learning: TensorFlow / Keras

Data Analysis: Pandas, NumPy