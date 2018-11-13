
import huasca

res = huasca.detect.faces('tests/images/celeb_group.jpg')
assert len(res.boxes) == 10, "failed to detect celeb faces"
