import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

# from keras.models import load_model
import numpy as np
import argparse
import pickle
import time
import cv2
import os
import sklearn
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/test_embeddings.pickle",
    help='Path to test embeddings')


args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
true_labels = le.fit_transform(data['names'])

# Load the classifier model
model = load_model(args.mymodel)

pred_labels = []
for embedding in embeddings:
    # Predict class
    preds = model.predict(embedding)
    preds = preds.flatten()
    # Get the highest accuracy embedded vector
    label = np.argmax(preds)
    pred_labels.append(label)

print(classification_report(y_true, y_pred)
