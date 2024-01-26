# Necessary libraries
import cv2
from ultralytics import YOLO
import numpy as np

# Use VideoCapture constructor from cv2
# 0 is used for the default camera, you can change it to a different number if you have multiple cameras
# 1 is used for alternative camera (1/26/2024)
cap = cv2.VideoCapture(1)  

# Call and store the yolov8 pretrained model
model = YOLO("yolov8m.pt")  # Load YOLOv8 model with the specified pretrained weights

# Opening the classes file in read mode
my_file = open("classes.txt", "r")  # Assuming classes.txt contains the names of object classes, one per line

# Reading the file
data = my_file.read()

# Creating a list of all object classes
classes_list = data.replace('\n', ',').split(",")  # Convert the string of classes to a list

# Closing class file
my_file.close()

# Looping through
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Storing model output
    results = model(frame, device="mps")  # Run the YOLOv8 model on the current frame using Metal GPU acceleration

    # Calling the first element in results
    result = results[0]

    # Creating bounding boxes and converting them to a numpy array
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # Extract bounding boxes from the model output
    classes = np.array(result.boxes.cls.cpu(), dtype="int")  # Extract predicted classes from the model output

    # Drawing a bounding box and writing textual classification on top of each box
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)  # Draw a rectangle around the detected object

        cv2.putText(frame, classes_list[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)  # Display the class label above the rectangle

    cv2.imshow("frame", frame)  # Display the frame with bounding boxes and labels

    key = cv2.waitKey(1)  # Allows camera feed to remain on

    # Quitting program with the q keyboard input
    if key & 0xFF == ord("q"):
        break

# Releasing video capture
cap.release()

cv2.destroyAllWindows()

# Update as of 1/26/2024