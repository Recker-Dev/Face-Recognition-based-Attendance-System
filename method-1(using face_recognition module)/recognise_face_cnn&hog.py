import face_recognition
import argparse
import pickle
import cv2
import csv

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['PRESENT']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in data:
            if item != "Unknown":  # Skip adding "Unknown" entries
                writer.writerow({'PRESENT': item})

ap= argparse.ArgumentParser()
ap.add_argument("-e","--encodings", required=True, help="path to serailized database of facial encodings")
ap.add_argument("-i","--image", required=True, help="path to input image")
ap.add_argument("-d","--detection-method",type=str, default="hog", help="face detection model to use: either 'hog' or 'cnn' ")
ap.add_argument("-f","--filename",type=str, default="present.csv", help="name of the CSV file to store the recognized faces (default: present.csv)")
args=vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data=pickle.loads(open(args["encodings"],"rb").read())

# load the input image and convert it from BGR to RGB
image=cv2.imread(args["image"])
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face

print("[INFO] recognizing faces...")
boxes=face_recognition.face_locations(rgb,model=args["detection_method"])
encodings=face_recognition.face_encodings(rgb,boxes)

# initialize the list of names for each face detected
names=[]

#loop over the facial embeddings
for encoding in encodings:
    # attempt to match each face in the input image to our known
	# encodings
    matches=face_recognition.compare_faces(data["encodings"],encoding)
    name="Unknown"

    #check to see if we have found a match
    if True in matches:
       # find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
        matchedIndxs=[i for (i,b) in enumerate(matches) if b ]
        counts={}

        # loop over the matched indexes and maintain a count for
		# each recognized face face
        for i in matchedIndxs:
            name=data["names"][i]
            counts[name]=counts.get(name,0)+1

        # determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
        name=max(counts, key=counts.get)


    #update the list of names
    names.append(name)

# Writing recognized names to CSV
write_to_csv(names, args["filename"])

# loop over the recognized faces
for((top,right,bottom,left),name) in zip(boxes,names):
    # draw the predicted face name on the image
    cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
    y=top-15 if top -15 >15 else top+15
    cv2.putText(image,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

#show the output
cv2.imshow("Image",image)
cv2.waitKey(0)


