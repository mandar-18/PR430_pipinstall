*****Readme.md***

Repository to share source code of team pip_install for Smart India Hackathon, 2020

**Objective:**

Detect any unusual activity in a given video, or through the live stream from a camera. Report start and end time of the unusual event in video or clock timings in case of real time detection.

**Approach?**

We collected video data of various normal as well as abnormal events from internet. Extracted frames from data manually associated hand-picked of these frames to certain labels, say accident, fire and normal event. Except the 'NormalEvent' label, we treated every other label as an label which would contain all the frames of associated unusual or anomolous behaviour. Then we built a convolutional neural network which takes every frame from the input video and associates the frame to a label with largest corresponding probability. If labels associated with certain series of frames turns to be abnormal, the video associated with is termed as abnormal. 

**How did we do it?**

For training, we built a head layer over pre-trained CNN using ImageNet weights, Resnet50, which is a famous CNN in image classification. The head layer helped us achieve labelling frames with their largest corresponding probabilities. Though it gives each frame an annotation, it's not enough due to temporal nature of video frames as opposite to spatial-only nature of training images (Note that training images are discrete and have no interrelation at all). We may face flickering since continuous frames in a video may not have same label. Some of them may have other labels, which hampers overall prediction. To overcome, we apply rolling averaging over the predictions, which takes average prediction values of n previous values, the value of n being defined manually on trial and error basis.

**What does it achieve?**

Predicts if events happening in given videos or streams are normal or abnormal. If abnormal, it is monitored until it gets finished. Once finished, authorized personnel would get an email alerting him/her regarding the anomaly. The email would also contain starting and ending timings of anomaliy/ies.

**What's next?**

More research to be done on the labels, as very it can detect presently. More data has to be generated in order to predict more accurately. A distributed architechture can be designed to overcome computing power scarcity.

**How to implement this code?**

- Run predict_videos.py.
- A Flask server will run. Type 127.0.0.1:5000 in your browser.
- Type in the name of video you want to give input to the system. The video to be saved in example_clips folder in your project root. A sample video has been included in the       same.
- The process will start and another window will open which will show video footage along with on screen messages if unusual event occurs.
- For storing the output, just create an output folder in project root. 


**Requirements**

- Python 3
- Tensorflow 2
- Other module requirements are included in requirements.txt
