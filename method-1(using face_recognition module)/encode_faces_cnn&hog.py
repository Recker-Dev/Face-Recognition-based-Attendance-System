# import necessary packages
from imutils import paths
import face_recognition
import numpy as np
import argparse
import pickle
import cv2
import os

#constructing the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="path to input directory of images")
ap.add_argument("-e","--encodings",required=True, help="path to output serialized db of facial encodings")
ap.add_argument("-m", "--detector", type=str, default="cnn",help="face detection model to use: either 'hog' or 'cnn'")
args= vars(ap.parse_args())

#grabbing path to input directory of images in our dataset
print("[INFO] quantifying faces...")
imagesPaths= list(paths.list_images(args["dataset"]))

#initialize the list of known encodings and know names
knownEncodings=[]
knownNames=[]
# We also need to initialize two lists before our loop, knownEncodings and knownNames , respectively. 
# These two lists will contain the face encodings and corresponding names for each person in the dataset.

# loop over the images path(s)
for(i,imagePath) in enumerate (imagesPaths):

    #extract person name from the image path
    print("[INFO] processing image {}/{}".format(i+1,len(imagesPaths)))
    name=imagePath.split(os.path.sep)[-2]

    #load the input image and convert it from BGR (opencv order)
    #to dlib ordering(RGB)
    image=cv2.imread(imagePath)
    rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x,y)- coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,model=args["detector"])

    #compute the facial embeddings for the face
    encodings= face_recognition.face_encodings(rgb,boxes)
    # print(encodings)

    #loop over the encodings
    for encoding in encodings:
        # add each encoding and the name in set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# What would be the point of encoding the images unless we could use the encodings,
#  in another script which handles the recognition?
#  dump the facial encodings + names to disk

print("[INFO] serializing encodings...")
data= {"encodings":knownEncodings,"names":knownNames}
f=open(args["encodings"],"wb")
f.write(pickle.dumps(data))
f.close() 