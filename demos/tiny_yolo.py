import rtsp
import CVCellar

obj = CVCellar.detection.TinyYolo()

_image = rtsp.fetch_image()

annotated,classes,scores = obj.process(_image)

annotated.show()
