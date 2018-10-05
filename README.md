# Huasca

##### Computer vision models OOB (out-of-the-bottle).

                     __
                    [__]
       ___        .+'. '+.
       )_(       /:;/ _.+'\
       + +       +:._   .++
     .+'+'+.  _  |:._     |
    /+::_..+_[_]_+:._CV   |
    )_     /_   _\:._     |
    +;:    )_``'_(:._     +
    +;::+..+;:.._++.____.+'
    `+.._..`+...+'

##### Step into the cellar and select a bottle of computer visions.

  * Object tracking
  * Face classification
    * age
    * gender

## Roadmap

  * v0.1.0 - improve asset loading
  * v0.2.0 - implement basic models to support classification
    * face detection
    * generic object detection
    * style transfer
    * face recognition
  * v0.3.0 - reduce and combine models to save space

## Examples

### Object Detection

    # Get a PIL image from somewhere:
    _image = ...
    
    # Use PIL image as input:
    import huasca

    detector = huasca.detect.ObjectDetector()
    annotated,classes,scores,boxes = detector.detect(_image)

    annotated.show()
    annotated.save('test.png')

  * `annotated`: the input image with annotated boxes and labels drawn on it
  * `classes`: the labels of detected objects
  * `scores`: confidence score for each detected object
  * `boxes`: (x1,y1,x2,y2) coordinates for each box
    * top-left corner is (0,0) and offsets go down/right (physics indexing)

### Object Tracking

    import tracking

    data = json.load(json_data)
    object_log = tracking.track_objects(data)
    output_json = [obj.to_json() for obj in object_log]
