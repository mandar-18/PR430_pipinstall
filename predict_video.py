# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 128
# python predict_video.py --model model/road_activity.model --label-bin model/rd.pickle --input example_clips/fire_footage.mp4 --ou
# tput output/fire_footage2.avi --size 128

# import the necessary packages
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
from mail import sendmail
import pickle
import imutils
import cv2
import datetime
import time
from flask import Flask, render_template, request


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('Main_page.html')

@app.route('/prediction.html')
def predict():
    return render_template('prediction.html')

@app.route('/About_us.html')
def about_us():
    return render_template('About_us.html')

@app.route('/Result1.html', methods=['POST'])
def Result1():
    global annotation
    if request.method == 'POST':
        MODEL_PATH = 'model/final.model'
        PICKLE_PATH = 'model/final.pickle'
        #MODEL_PATH = 'model/real_time.model'
        #PICKLE_PATH = 'model/real_time.pickle'
        INPUT_VIDEO = request.form['inp_video']
        out = INPUT_VIDEO.split('.')
        INPUT_VIDEO = 'example_clips/'+request.form['inp_video']
        out = out[0]
        OUTPUT_VIDEO = 'output/' + out + '.avi'
        SIZE = 128

        print(MODEL_PATH,PICKLE_PATH,INPUT_VIDEO,OUTPUT_VIDEO,SIZE)
        #load the trained model and label binarizer from disk
        print("[INFO] loading model and label binarizer...")
        model = load_model(MODEL_PATH)
        lb = pickle.loads(open(PICKLE_PATH, "rb").read())

        # initialize the image mean for mean subtraction along with the
        # predictions queue
        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
        Q = deque(maxlen=SIZE)

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        vs = cv2.VideoCapture(INPUT_VIDEO)
        #vs = cv2.VideoCapture(0)
        writer = None
        (W, H) = (None, None)

        count = 0.0
        flag = 0
        start_frame = 0
        end_frame = 0
        status = {}
        annotation = ""
        que = deque()
        # loop over frames from the video file stream
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            count += 1.0
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # clone the output frame, then convert it from BGR to RGB
            # ordering, resize the frame to a fixed 224x224, and then
            # perform mean subtraction
            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)).astype("float32")
            frame -= mean

            # make predictions on the frame and then update the predictions
            # queue
            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            Q.append(preds)
            # perform prediction averaging over the current history of
            # previous predictions
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = lb.classes_[i]
            if len(que) == 30:
               que.popleft()
            if len(que) != 30:
               que.append(label)
            noOfAlerts = que.count("fire") + que.count("accident")
            if que.count("fire") > que.count("accident"):
                caseDetect = "fire"
            else:
                caseDetect = "accident"
            # draw the activity on the output frame
            text = "Alert!!: {}".format(label)

            # Changes starts here
            alert = ["fire", "accident"]

            #currentFrame = 0
            #print(label, flag)
            if len(que) == 30:
                if caseDetect in alert and noOfAlerts > 20:
                    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (0, 0, 255), 5)
                    if flag == 0:
                        annotation = caseDetect
                        start_frame = count - 20
                        flag = 1
                else:
                    if flag == 1:
                        end_frame = count - 10
                        flag = 2

                #name = './frame/frame'+str(currentFrame)+'.jpg'
                #cv2.imwrite(name,output)

            # check if the video writer is None
            if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30,
                    (W, H), True)

            # write the output frame to disk
            writer.write(output)

            # show the output image
            cv2.imshow("Output", output)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # changes made here

        if annotation != "":
            status = sendmail("harshpatel682@gmail.com", "Anomaly Detected!!!", "yes")
            status = status['email_status']

        #total_time = end_time - start_time
        #print("Time is: {}".format(str(datetime.timedelta(seconds=(total_time)))))
        print("count: {}".format(count))
        #print("Frame count: {}".format(f_start))
        # release the file pointers
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        start_frame = start_frame//30
        end_frame = end_frame // 30
    if flag == 1:
            end_frame = count
            end_frame = end_frame // 30
            flag = 2
    print(start_frame, end_frame)
    return render_template('Result1.html', label=annotation, count=count, start_time=start_frame, end_time=end_frame,
     status = status)


if __name__ == '__main__':
    app.run(debug=False)