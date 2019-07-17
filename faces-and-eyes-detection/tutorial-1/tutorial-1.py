# Facial Detection and Recognition

# Faces and Eyes Detection - Tutorial #1

# Description:
# - A simple tutorial for Faces and Eyes Detection,
#   using frames in shape of rectangles for each detected component;

# Authors:
# - Ruben Andre Barreiro


# Import OpenCV library
import cv2 as cv

# Fetch the Cascade Classifiers for Faces and Eyes
print("Loading Haar Cascade Classifiers...")
face_cascade = cv.CascadeClassifier('classifiers/haar-cascade/haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('classifiers/haar-cascade/haarcascade_eye.xml')

# Print a message in the case of an error occurred, during the loading of the Cascade Classifiers for the Faces
if not face_cascade.load('classifiers/haar-cascade/haarcascade_frontalface_default.xml'):
    print('Error in loading of the Cascade Classifiers for the Faces...')
    exit(0)

# Print a message in the case of an error occurred, during the loading of the Cascade Classifiers for the Eyes
if not eyes_cascade.load('classifiers/haar-cascade/haarcascade_eye.xml'):
    print('Error in loading of the Cascade Classifiers for the Eyes...')
    exit(0)

# Load the pretended image to be analysed and resize it
print("Loading image and resizing it...")
img = cv.imread('images/protrait-photo-big-1.jpg')
img = cv.resize(img, (1080, 720), interpolation = cv.INTER_AREA)

# Convert the loaded image to a gray scale to be use, if necessary
print("Loading image and resizing it...")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect Faces on the loaded image in a gray scale
print("Converting image to a gray scale, to detect the Faces on it...")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Analyse all the possible points in the loaded image to detect a Face
print("Analysing all the possible points in the loaded to detect a Face...")
for (x, y, w, h) in faces:

    # Draw a rectangle for each detected Face, around it
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)

    roi_gray = gray[y:(y + h), x:(x + w)]
    roi_color = img[y:(y + h), x:(x + w)]

    # Detect Eyes on the loaded image in a gray scale
    print("Converting image to a gray scale, to detect the Eyes on it...")
    eyes = eyes_cascade.detectMultiScale(roi_gray)

    # Analyse all the possible points in the loaded image to detect an Eye
    print("Analysing all the possible points in the loaded image to detect an Eye...")
    for (ex, ey, ew, eh) in eyes:

        # Draw a rectangle for each detected Eye, around it
        cv.rectangle(roi_color, (ex, ey), ((ex + ew), (ey + eh)), (0, 255, 0), 2)

# Generate the final image with all the detected Faces and Eyes
print("Generating the final image with all the detected Faces and Eyes...")
cv.imshow('Faces and Eyes Detection - Tutorial #1', img)

# Wait until some key is pressed to close the window with the generated image with all the detected Faces and Eyes...
print("Press some key to close the window with the generated image with all the detected Faces and Eyes...")
c = cv.waitKey(0)
cv.destroyAllWindows()