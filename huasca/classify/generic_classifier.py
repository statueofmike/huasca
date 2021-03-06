""" 
    Application of keras implementation of MobileNet trained on ImageNet.
    The MobileNetV2 model takes 14Mb of space.
"""

from ..root import mobilenet

class ObjectClassifier:
    preprocessed = None
    batch_size = 1

    def __init__(self):
        self.model = mobilenet.fully_connected_model()

    def predict(self,input_image_path):
        preprocessed = self.model.preprocess_input(self.model.load_image_file(input_image_path))
        predictions = self.model.predict(preprocessed, batch_size=self.batch_size)
        return self.model.decode_predictions(predictions,top=5)[0]

## lazy-loading models
_classifier = None

def object(input_image_path, verbose=True):
    """ Object detection via Keras Applications Mobilenet V2. Input PIL Image. """
    global _classifier
    if not _classifier:
        if verbose:
            print("Lazy-loading object classifier...")
        _classifier = ObjectClassifier()
        if verbose:
            print("  ...gender and age models loaded.")
    return _classifier.predict(input_image_path)

