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

  * Face detection & localization
  * Face classification
    * age
    * gender
  * Object tracking
  * Object classification w/o localization

## Roadmap

  * v0.1.0 - improve asset loading
    * annotations for face/object detection
  * v0.2.0 - reduce and combine models to save space
  * v0.3.0 - implement basic models to support classification
    * face detection
    * generic object detection & localization
    * style transfer
    * face recognition

## Examples

### Detection

Detection results have the following:

  * `boxes`: Boxes follow PIL format of (left, upper, right, lower)
    * top-left corner is (0,0) and offsets go down/right from there (physics indexing)
  * `scores`: confidence score for each detected object
  * `labels`: label description of the object ('face')
  * `portraits`: the object cropped from its source image
  * `base_image`: the source image the objects were found in
  * ~~`annotated`: the source image with objects annotated~~ (not implemented yet)

#### Face Detection

    # Get a PIL image from somewhere:
    image = ...
    
    # Use PIL image as input:
    import huasca

    results = huasca.detect.faces(image)

    results.portraits[0].show()
    annotated.save('test.png')


#### Face Demographics

    # Get a PIL image from somewhere:
    image = ...

    import huasca
    gender,age = huasca.classify.demographics(image)


### Object Tracking

    import huasca

    data = json.load(json_data)
    object_log = huasca.object_tracking.track_objects(data)
    output_json = [obj.to_json() for obj in object_log]
