#
#
# from collections import deque
#
# import numpy as np
# from flask import Flask, Response, render_template
# import cv2
# import tensorflow as tf
#
# IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
# # Specify the number of frames of a video that will be fed to the model as one sequence.
# SEQUENCE_LENGTH = 16
# DATASET_DIR = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/"
# CLASSES_LIST = ["NonViolence", "Violence"]
#
# model = tf.keras.models.load_model("model/mobilenet-lstmV2.h5")
#
# def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):
#     # Read from the video file.
#     video_reader = cv2.VideoCapture(video_file_path)
#
#     # Get the width and height of the video.
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # VideoWriter to store the output video in the disk.
#     video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
#                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
#
#     # Declare a queue to store video frames.
#     frames_queue = deque(maxlen=SEQUENCE_LENGTH)
#
#     # Store the predicted class in the video.
#     predicted_class_name = ''
#
#     # Iterate until the video is accessed successfully.
#     while video_reader.isOpened():
#
#         ok, frame = video_reader.read()
#
#         if not ok:
#             break
#             # Resize the Frame to fixed Dimensions.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#
#         # Normalize the resized frame
#         normalized_frame = resized_frame / 255
#
#         # Appending the pre-processed frame into the frames list.
#         frames_queue.append(normalized_frame)
#
#         # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
#         # Check if the number of frames in the queue are equal to the fixed sequence length.
#         if len(frames_queue) == SEQUENCE_LENGTH:
#             # Pass the normalized frames to the model and get the predicted probabilities.
#             predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
#
#             # Get the index of class with highest probability.
#             predicted_label = np.argmax(predicted_labels_probabilities)
#
#             # Get the class name using the retrieved index.
#             predicted_class_name = CLASSES_LIST[predicted_label]
#
#         # Write predicted class name on top of the frame.
#         if predicted_class_name == "Violence":
#             cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
#         else:
#             cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)
#
#         # Write The frame into the disk using the VideoWriter
#         video_writer.write(frame)
#
#     video_reader.release()
#     video_writer.release()
#
#
#
# if __name__ == '__main__':
#     predict_frames('/home/dtech/Documents/git/falcon-eye/SCVD/videos/Non-Violence Videos/nv23.mp4', 'out/Output-Test-Video.mp4', SEQUENCE_LENGTH)
#
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

# Download Model
if not os.path.exists(os.path.join(os.getcwd(), MODEL_FILE)):
    print("Downloading model")
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break