# Huasca

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

Computer vision deep learning models: effective & useable out-of-the-bottle.

Step into the cellar and select a model for computer visions.

## Examples

### Object Detection

    import rtsp
    import huasca

    obj = huasca.detection.TinyYolo()
    
    _image = rtsp.fetch_image()

    annotated,classes,scores = obj.process(_image)

    annotated.show()
    annotated.save('test.png')
