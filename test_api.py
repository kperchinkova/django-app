# import the necessary packages
import requests
import cv2

# define the URL to our face detection API
url = "http://localhost:8000/face_detection/predict/"

# load our image and now use the face detection API to find faces in
# images by uploading an image directly
image = cv2.imread("test.png")
# hist = hog(image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
payload = {"image": open("test.png", "rb")}
r = requests.post(url, files=payload).json()
print "test.png: {}".format(r)

# show the output image
cv2.imshow("test.png", image)
cv2.waitKey(0) 