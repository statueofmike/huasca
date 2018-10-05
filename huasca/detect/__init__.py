""" Module for detection:
      * face detection
      * general object detection """

from .yolo import _TinyYolo
del(yolo)

class ObjectDetector():
    """ Object Detection using the YOLO algorithm """
    def __init__(self):
        raise NotImplementedError("Object Detection Removed for Model Size Contraints")
        self._detector = _TinyYolo()

    def detect(self, frame):
        # Box coordinates are in "Physics matrix" notation from top-left origin: 
        # (Y1,X1,Y2,X2)
        return self._detector.detect(frame)

class FaceDetector():
    def __init__(self):
        raise NotImplementedError("Face Detection not yet implemented")