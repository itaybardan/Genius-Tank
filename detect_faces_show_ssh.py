#!/usr/bin/env python3

# in this script we will display video of the pi camera capture, lining the faces in frame
# and catagorize to 4 personalities (unknown, itay, sinwar, nassralah

# use like this:
# sudo python3 detect_faces.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle


import argparse
import pickle
import time
from collections import Counter

import cv2
import face_recognition
import imutils
from PIL import Image
from imutils.video import FPS
from imutils.video import VideoStream


def main():
    # set arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
    arg_parser.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
    args = vars(arg_parser.parse_args())

    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    detector = cv2.CascadeClassifier(args["cascade"])

    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30),
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
        '''
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
        '''
        # an alternative to cv2.imgshow - should be working trough ssh

        img2 = Image.fromarray(frame, 'RGB')
        img2.show()

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
