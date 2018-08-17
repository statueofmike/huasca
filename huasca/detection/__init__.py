""" Module for detection:
      * face detection
      * general object detection """

from .yolo import _TinyYolo
del(yolo)

class ObjectDetector():
    """ Object Detection using the YOLO algorithm """

    def __init__(self):
        self._detector = _TinyYolo()

    def detect(self, frame):
        """
            Main Calling function

                Box coordinates are in "Physics matrix" notation: 
                    ( Y1 , X1 , Y2, X2)
                    (X1,Y1) are box corners from top-left origin
        """
        return self._detector.detect(frame)

class FaceDetector():

    def __init__(self):
        raise NotImplementedError("Face Detection not yet implemented")