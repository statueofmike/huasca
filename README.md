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

  * Object Detection
  * Object Tracking

## Object Detection

Returns `annotated`,`classes`,`scores`,`boxes`

  * `annotated`: the input image with annotated boxes and labels drawn on it
  * `classes`: the labels of detected objects
  * `scores`: confidence score for each detected object
  * `boxes`: (x1,y1,x2,y2) coordinates for each box
    * top-left corner is (0,0) and offsets go down/right (physics indexing)

### Examples

    # Get a PIL image from somewhere:
    _image = ...
    
    # Use PIL image as input:
    import huasca

    detector = huasca.detection.ObjectDetector()
    annotated,classes,scores,boxes = detector.detect(_image)

    annotated.show()
    annotated.save('test.png')


## Object Tracking

Simple object tracking for computer vision.

Object tracking based on parameterized identity decision:

  * IoU, Intersection-over-Union thresholding
  * Time between object observation
  * Frames between object observation

### Examples

  * Input: json log of detected objects and their bounding boxes
  * Output: list of json-able objects identified in the log history

```
    import tracking

    data = json.load(json_data)
    object_log = tracking.track_objects(data)
    output_json = [obj.to_json() for obj in object_log]
```

#### Tracking Roadmap

  1. momentum modelling
  2. use object class information in identity function

