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

### Object Detection

Returns `annotated`,`classes`,`scores`,`boxes`

  * `annotated`: the input image with annotated boxes and labels drawn on it
  * `classes`: the labels of detected objects
  * `scores`: confidence score for each detected object
  * `boxes`: (x1,y1,x2,y2) coordinates for each box
    * top-left corner is (0,0) and offsets go down/right (physics indexing)

## Examples

### Object Detection

    # Get a PIL image from somewhere:
    import rtsp
    _image = rtsp.Client().read()
    
    # Use PIL image as input:
    import huasca

    model = huasca.detection.TinyYolo()
    annotated,classes,scores,boxes = model.detect(_image)

    annotated.show()
    annotated.save('test.png')
