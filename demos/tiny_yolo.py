import rtsp
import huasca

obj = huasca.detection.TinyYolo()

_image = rtsp.fetch_image()

annotated,classes,scores = obj.process(_image)

annotated.show()
