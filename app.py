import os
import sys
import time, threading
import moviepy.editor as moviepy

import json
from collections import deque
from io import BytesIO

from flask import Flask, Response, render_template, jsonify
import imutils
from imutils.video import VideoStream, FileVideoStream
from shapely.geometry import Point, Polygon, LineString

import numpy as np
import cv2
import tensorflow as tf

from EasyROI import EasyROI

# Config
CONFIG = {"debug": False, "prediction": {"violence": True, "tamper": True, "roi": True}, "clip-time": 30}
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
RESTRICTED_AREA = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], np.int32)

# Global Variables
outputFrame = None
lock = threading.Lock()
# video = VideoStream(src=0).start()
# video = FileVideoStream("/home/dtech/Documents/git/falcon-eye/test-rail.mp4").start()
# video = FileVideoStream("/home/dtech/Documents/git/falcon-eye/V_19.mp4").start()
video = FileVideoStream("/home/dtech/Documents/git/falcon-eye/rail3.mp4").start()

fgbg = cv2.createBackgroundSubtractorMOG2()
frame = video.read()
fgmask = fgbg.apply(frame)
kernel = np.ones((5, 5), np.uint8)
detection = {"violence": 0, "roi": [], "tamper": 0}
video_clip = []
video_clip_time_start = time.time()
roi_data = {"track": [], "ra": []}
class_names = []
time.sleep(2.0)

# Constants

with open("model/yoloV4-tiny/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

DATASET_DIR = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/"
CLASSES_LIST = ["NonViolence", "Violence"]

model = tf.keras.models.load_model("model/mobilenet-lstmV4.h5")
yolo = cv2.dnn.readNet("model/yoloV4-tiny/yolov4-tiny.weights", "model/yoloV4-tiny/yolov4-tiny.cfg")
yolo = cv2.dnn_DetectionModel(yolo)
yolo.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    return render_template('index.html')


pred_queue = deque(maxlen=SEQUENCE_LENGTH)
predicted_class_name = ''


def predict_violence(frame):
    pred = model.predict(np.expand_dims(cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255, axis=0), verbose=False)[0]
    pred_queue.append(pred)
    results = np.array(pred_queue).mean(axis=0)
    # print(pred[0])
    return 1 if (pred > 0.56)[0] else 0
    # global predicted_class_name, frames_queue
    # resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #
    # # Normalize the resized frame
    # normalized_frame = resized_frame / 255
    #
    # # Appending the pre-processed frame into the frames list.
    # frames_queue.append(normalized_frame)
    #
    # # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
    # # Check if the number of frames in the queue are equal to the fixed sequence length.
    # predicted_label = 0
    # if len(frames_queue) == SEQUENCE_LENGTH:
    #     # Pass the normalized frames to the model and get the predicted probabilities.
    #     predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0), verbose=0)[0]
    #     # print(predicted_labels_probabilities)
    #     # Get the index of class with highest probability.
    #     predicted_label = np.argmax(predicted_labels_probabilities)
    #
    #     # Get the class name using the retrieved index.
    #     predicted_class_name = CLASSES_LIST[predicted_label]
    #
    # # Write predicted class name on top of the frame.
    # # if predicted_class_name == "Violence":
    # #     cv2.putText(frame, predicted_class_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # # else:
    # #     cv2.putText(frame, predicted_class_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    # return predicted_label, frame


def predict_tamper(frame):
    global fgmask
    a = 0
    bounding_rect = []
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        bounding_rect.append(cv2.boundingRect(contours[i]))
    for i in range(0, len(contours)):
        if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
            a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
        return a >= int(frame.shape[0]) * int(frame.shape[1]) / 3


def falcon_eye():
    global video, outputFrame, lock, video_clip_time_start
    video_clip_file = cv2.VideoWriter(f'static/clip/{time.ctime().replace(" ", "_")}.webm',
                                      cv2.VideoWriter_fourcc(*"vp80"),
                                      10, (500, 500))

    # video = getMotionInfuenceMap(video)
    while True:
        frame = video.read()
        # print((time.time() - video_clip_time_start))
        if time.time() - video_clip_time_start > CONFIG["clip-time"]:

            video_clip_time_start = time.time()
            if detection["violence"] == 1 or detection["roi"] != [] or detection["tamper"] == 1:
                video_clip_file.release()
                if (CONFIG["debug"]):
                    print(f'[FalconEye:{time.ctime()}] Clip saved: "{time.ctime().replace(" ", "_")}.mp4"')

            video_clip_time_start = time.time()
            video_clip_file = cv2.VideoWriter(f'static/clip/{time.ctime().replace(" ", "_")}.webm',
                                              cv2.VideoWriter_fourcc(*"vp80"),
                                              10, (500, 500))

        # if (not ok):
        #     continue
        # cv2.imshow(frame, "LIVE")
        # cv2.waitKey()
        # frame =  cv2.resize(frame, (200, 200))
        start = time.time()
        v = 0
        t = 0
        if (CONFIG["prediction"]["violence"]):
            v = predict_violence(frame)
            detection["violence"] = v
        if (CONFIG["prediction"]["tamper"]):
            t = predict_tamper(frame)
            detection["tamper"] = t
        classes, scores, boxes = yolo.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        detection["roi"] = []
        for (classid, score, box) in zip(classes, scores, boxes):
            if class_names[classid] == "train":
                continue
            color = COLORS[int(classid) % len(COLORS)]
            # print(box)
            p1 = Point(box[0], box[1] + box[3])
            p2 = Point(box[0] + box[2], box[1] + box[3])
            feet_line = LineString([(box[0], box[1] + box[3]), (box[0] + box[2], box[1] + box[3])])
            for i in range(len(poly_track)):
                if poly_track[i].intersects(feet_line):
                    color = (0, 0, 255)
                    # send("alert-track")
                    detection["roi"].append({"type": "track", "ob": class_names[classid], "area": ("track" + str(i))})
            for i in range(len(poly_ra)):
                if poly_ra[i].intersects(feet_line):
                    color = (0, 0, 255)
                    # send("alert-track")
                    detection["roi"].append({"type": "ra", "ob": class_names[classid], "area": ("ra" + str(i))})

            label = "%s : %f" % (class_names[classid], score)
            cv2.rectangle(frame, box, color, 1)
            cv2.line(frame, (box[0], box[1] + box[3]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)
            cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Restricted Area ROI
        for i in range(len(roi_data["ra"])):
            cv2.polylines(frame, [np.array(roi_data["ra"][i], np.int32)],
                          True, (0, 255, 255),
                          2)
            cv2.putText(frame, "RA" + str(i + 1), (roi_data["ra"][i][0][0], roi_data["ra"][i][0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Draw Track ROI
        for i in range(len(roi_data["track"])):
            cv2.polylines(frame, [np.array(roi_data["track"][i], np.int32)],
                          True, (0, 0, 255),
                          2)
            cv2.putText(frame, "T" + str(i + 1), (roi_data["track"][i][0][0], roi_data["track"][i][0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw FPS
        fps = "FPS: %.2f " % (1 / (end - start))
        cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        video_clip_file.write(cv2.resize(frame, (500, 500)))
        if (CONFIG["debug"]):
            print(f'[FalconEye:{time.ctime()}] {detection}')

        with lock:
            outputFrame = frame.copy()


@app.route('/video_feed')
def video_feed():
    # global video
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection')
def REST_detection():
    # clip_list = os.listdir("static/clip")
    # for clip in clip_list:
    #     if clip.split(".")[1] == "avi":
    #         vid = moviepy.VideoFileClip("static/clip/" + clip)
    #         vid.write_videofile(("static/clip/" + clip).replace(".avi", ".mp4"))
    return jsonify(detection)


@app.route('/clips')
def update_clips():
    return jsonify(os.listdir("static/clip"))


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:

        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


if __name__ == '__main__':
    args = sys.argv
    with open('roi_data.json', 'r') as openfile:
        # Reading from json file
        roi_data = json.load(openfile)
        openfile.close()
    poly_ra = []
    poly_track = []
    for i in range(len(roi_data["ra"])):
        poly_ra.append(Polygon(np.array(roi_data["ra"][i])))
    for i in range(len(roi_data["track"])):
        poly_track.append(Polygon(np.array(roi_data["track"][i])))
    print(roi_data)
    if (len(args) > 1):
        if (args[1] == "add"):
            roi_helper = EasyROI(verbose=True)
            ok = False
            while (not ok):
                frame = video.read()
                # if (not ok):
                #     continue
                assert len(args) == 3, "add <ra/track>"
                match (args[2]):
                    case "ra":
                        polygon_roi = roi_helper.draw_polygon(frame,
                                                              1)  # quantity=3 specifies number of polygons to draw
                        frame_temp = roi_helper.visualize_roi(frame, polygon_roi)
                        roi_data["ra"].append(np.array(polygon_roi['roi'][0]['vertices']).tolist())
                        with open("roi_data.json", "w") as outfile:
                            json.dump(roi_data, outfile)
                            outfile.close()
                        print(polygon_roi['roi'][0]['vertices'])
                    case "track":
                        polygon_roi = roi_helper.draw_polygon(frame,
                                                              1)  # quantity=3 specifies number of polygons to draw
                        frame_temp = roi_helper.visualize_roi(frame, polygon_roi)
                        roi_data["track"].append(np.array(polygon_roi['roi'][0]['vertices']).tolist())
                        with open("roi_data.json", "w") as outfile:
                            json.dump(roi_data, outfile)
                            outfile.close()
                        print(polygon_roi['roi'][0]['vertices'])

            exit()
    t = threading.Thread(target=falcon_eye)
    t.daemon = True
    t.start()
    print("Falcon Eye Online!")
    # socketio.run(app, port=8080)
    app.run(host='0.0.0.0', port=8080, threaded=True, use_reloader=False)
