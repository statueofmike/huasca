""" Face Detection via https://github.com/yeephycho/tensorflow-face-detection
    - http://www.apache.org/licenses/LICENSE-2.0 """


import os
import numpy as np
import tensorflow as tf

from PIL import Image

from ..representation import Detection as _Detection

BASE_DIR = os.path.dirname(__file__)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = BASE_DIR + '/../bin/' + 'face.pb'

class FaceDetector:
    def __init__(self):
        # Load models
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph)
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()

    def detect(self, image,threshold=0.5):
        if isinstance(image,str): # assume a file path
            image = Image.open(image)
        if not isinstance(image,np.ndarray): # assume PIL.Image obj
            image = np.asarray(image)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        # Ratio to real position
        boxes[0, :, [0, 2]] = (boxes[0, :, [0, 2]]*image.shape[0])
        boxes[0, :, [1, 3]] = (boxes[0, :, [1, 3]]*image.shape[1])
        
        boxes,scores = np.squeeze(boxes).astype(int), np.squeeze(scores)
        boxes = boxes[scores > threshold]
        boxes = [(b,a,d,c) for (a,b,c,d) in boxes]
        scores = scores[scores > threshold]
        labels = ['face' for box in boxes]
        return _Detection(boxes, scores, labels, image)

## lazy-loading models
_detector = None

def detect_faces(image,threshold=0.5):
    global _detector
    if not _detector:
        _detector = FaceDetector()
    return FaceDetector().detect(image,threshold)