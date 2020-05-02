# this script encode the faces in the dataset images, with face_Recognition model

# run like this: 
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method cnn
# or
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

def main():
    # construct the argument parser and parse the arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--dataset", required=True,help="path to input directory of faces + images")
    argparser.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
    argparser.add_argument("-d", "--detection-method", type=str, default="cnn",help="face detection model to use: either `hog` or `cnn`")
    args = vars(argparser.parse_args())

    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(args["dataset"]))

    knownEncodings = []
    knownNames = []

    for (i, image_path) in enumerate(image_paths):
            print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

            # the name is the name of the folder
            name = image_path.split(os.path.sep)[-2]
            print(f"image path : {image_path}")
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:

                    knownEncodings.append(encoding)
                    knownNames.append(name)

    # dump the facial encodings names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()

if __name__ == '__main__':
    main()
