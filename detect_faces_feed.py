# in this script we will display video of the pi camera capture, lining the faces in frame

# use like this:
# sudo python3 detect_faces.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle


import argparse
import os
import pathlib
import pickle
from collections import Counter

import cv2
import face_recognition
import imutils


def main():
    # set arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
    arg_parser.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
    arg_parser.add_argument("-t", "--test", required=True, help="path to test dataset")

    args = vars(arg_parser.parse_args())

    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("[INFO] start to read test dataset...")

    test_path = os.path.join(pathlib.Path(__file__).parent.resolve(), args["test"])
    for root, dirs, files in os.walk(test_path):
        for file_name in files:
            img_path = os.path.join(str(root), str(file_name))
            frame = cv2.imread(img_path)
            frame = imutils.resize(frame, width=500)
            gray = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
            rgb = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            gray = imutils.resize(gray, width=500)
            rgb = imutils.resize(rgb, width=500)
            rects = face_cascade.detectMultiScale(gray,
                                                  scaleFactor=1.05,
                                                  minNeighbors=10, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

            # compute the facial embeddings for each face bounding box
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
                # check if there is a match
                if True in matches:
                    matched_indexes = [i for (i, b) in enumerate(matches) if b]
                    counts = Counter()
                    for i in matched_indexes:
                        name = data["names"][i]
                        counts[name] += 1
                    # the name with the highest counts is our predicted name
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            # display the image
            cv2.imshow("Frame", frame)
            root_path = pathlib.Path(__file__).parent.resolve()
            cv2.imwrite(os.path.join(root_path, 'results', str(file_name)), frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
