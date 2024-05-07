from ultralytics import YOLO
import cv2
from datetime import datetime

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = './langvideo.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

location = "molde"

# Read frames
while ret:
    ret, frame = cap.read()
    desired_width = 1040  # Adjust as needed
    desired_height = 1140  # Adjust as needed
    frame = cv2.resize(frame, (desired_width, desired_height))

    if ret:
        # Detect objects and track them
        results = model.track(frame, conf=0.7, persist=True, classes=4)

        # Plot results
        frame_ = results[0].plot()

        # Iterate over detected objects
        for box in results[0].boxes:
            class_id = int(box.cls)  # Get class ID
            class_label = results[0].names[class_id]  # Get class label from class ID
            print(f'Detected class: {class_label}')  # Print class label


            #Save the frame with the date and time as the filename
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{current_time}_{class_label}_{location}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved image: {filename}")

        # Visualize
        cv2.imshow('frame', frame_)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
