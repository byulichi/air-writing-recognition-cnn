import cv2
import numpy as np
import mediapipe as mp

# 1. INITIALIZE MEDIAPIPE (The Hand Tracker)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# THIS IS THE LINE THAT WAS MISSING!
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# 2. INITIALIZE WEBCAM
cap = cv2.VideoCapture(0)

# 3. SET UP THE CANVAS
ret, frame = cap.read()
h, w, c = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Variables to remember where the finger was in the last frame
prev_x, prev_y = None, None
mode = "STOP"  # "DRAW", "PAUSE", or "STOP"

print("Webcam Active. Wave your index finger to draw.")
print("Press 'c' to clear the canvas.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Landmarks
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            # Convert normalized coordinates to pixels
            curr_x = int(index_tip.x * w)
            curr_y = int(index_tip.y * h)

            # Detect finger state
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            # Gesture mode
            if index_up and not middle_up:
                mode = "DRAW"
            elif index_up and middle_up:
                mode = "PAUSE"
            else:
                mode = "STOP"

            # Draw a circle on the video frame for visualization
            cv2.circle(frame, (curr_x, curr_y), 10, (0, 255, 255), cv2.FILLED)

            # Draw on canvas if in DRAW mode
            if mode == "DRAW":
                if prev_x is None:
                    prev_x, prev_y = curr_x, curr_y
                cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), (255, 255, 255), 15)
                prev_x, prev_y = curr_x, curr_y
            else:
                prev_x, prev_y = None, None

            # Draw hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None

    # Blend canvas and frame
    blended_frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Display mode on screen
    cv2.putText(blended_frame, mode, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the windows
    cv2.imshow("Air Writing - Video Feed", blended_frame)
    cv2.imshow("The Black Canvas", canvas)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()