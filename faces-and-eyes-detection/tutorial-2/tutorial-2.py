# Facial Detection and Recognition

# Faces and Eyes Detection - Tutorial #2

# Description:
# - A simple tutorial for Faces and Eyes Detection,
#   using frames in shape of circles for each detected component,
#   during a video capture;

# Authors:
# - Ruben Andre Barreiro

# Import future versions' libraries
from __future__ import print_function

# Import OpenCV library
import cv2 as cv

# Function to detect and display the detections of Faces and Eyes in a current loaded frame
def detectAndDisplay(frame):

    # Convert the current loaded frame of the video capture to a gray scale to be use, if necessary
    print("Loading the current loaded frame of the video capture...")
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect Faces on the current loaded frame of the video capture in a gray scale
    print("Converting the current loaded frame of the video capture to a gray scale, to detect the Faces on it...")
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        center = ((x + (w // 2)), (y + (h // 2)))

        frame = cv.ellipse(frame, center, ((w // 2), (h // 2)), 0, 0, 360, (0, 0, 255), 4)

        face_roi = frame_gray[y:(y + h), x:(x + w)]

        # Detect Eyes on the current loaded frame of the video capture in a gray scale
        print("Converting the current loaded frame of the video capture to a gray scale, to detect the Eyes on it...")
        eyes = eyes_cascade.detectMultiScale(face_roi)

        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.25))

            frame = cv.circle(frame, eye_center, radius, (0, 255, 0), 4)

    # Generate the current final loaded frame of the video capture with all the detected Faces and Eyes
    print("Generating the final loaded frame of the video capture with all the detected Faces and Eyes...")
    cv.imshow('Faces and Eyes Detection - Tutorial #2', frame)


# Fetch the Cascade Classifiers for Faces and Eyes
print("Loading Haar Cascade Classifiers...")
face_cascade = cv.CascadeClassifier('classifiers/haar-cascade/haarcascade_frontalface_alt.xml')
eyes_cascade = cv.CascadeClassifier('classifiers/haar-cascade/haarcascade_eye_tree_eyeglasses.xml')

# Print a message in the case of an error occurred, during the loading of the Cascade Classifiers for the Faces
if not face_cascade.load('classifiers/haar-cascade/haarcascade_frontalface_alt.xml'):
    print('Error in loading of the Cascade Classifiers for the Faces...')
    exit(0)

# Print a message in the case of an error occurred, during the loading of the Cascade Classifiers for the Eyes
if not eyes_cascade.load('classifiers/haar-cascade/haarcascade_eye_tree_eyeglasses.xml'):
    print('Error in loading of the Cascade Classifiers for the Eyes...')
    exit(0)

camera_device = 0

# Read/Load the Video Stream/Capture
cap = cv.VideoCapture(camera_device)

# Print a message in the case of an error occurred, during the reading/loading of the Video Stream/Capture
if not cap.isOpened:
    print('Error in reading/loading of the Video Stream/Capture...')
    exit(0)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (1080, 720), interpolation=cv.INTER_AREA)

    if frame is None:
        print('No Frame from the Video Stream/Capture!!!')
        break

    detectAndDisplay(frame)

    # Wait until some key is pressed to close the window with the video capture with all the detected Faces and Eyes...
    print("Press some key to close the window with the video capture with all the detected Faces and Eyes...")
    if cv.waitKey(10) == 27:
        break