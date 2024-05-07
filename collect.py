import cv2
import os
import mediapipe as mp

# Parameters
data_dir = 'data'
employee_names = ['Loai','Yunus','Fatema']  # Add more names as needed

# Create directories for data collection
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for name in employee_names:
    dir_path = os.path.join(data_dir, name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Collect data
for name in employee_names:
    print(f"Collecting data for {name}: Press 'c' to collect. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using Mediapipe
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw bounding box on the detected face
                cv2.rectangle(frame, (x-40, y-40), (x + w + 5, y + h + 5), (0, 0, 255), 2)
                # Display the current frame
                cv2.imshow('Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):  # Capture an image
                    # Crop the detected face
                    cropped_face = frame[y-40:y + h + 5, x-40:x + w + 5]

                    # Resize the cropped face to 224x224
                    resized_face = cv2.resize(cropped_face, (224, 224))

                    # Save the captured image
                    image_path = os.path.join(data_dir, name, f'{name}_{len(os.listdir(os.path.join(data_dir, name))) + 1}.jpg')
                    cv2.imwrite(image_path, resized_face)
                    print("Image Saved") 

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

cap.release()
