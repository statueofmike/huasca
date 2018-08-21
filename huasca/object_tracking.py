"""
    Object Tracking Algorithm

    Input: JSON containing a timestamp and list of bounding boxes for objects.

    Object tracking is performed with only a short history. 
    No momentum or long-term modelling is performed.
    Tracking Parameters:
      * iou_threshold
      * time_limit
      * frame_limit

    Output: log of objects tracked with a lifetime over live_time_limit threshold.
"""

import time as _time

class ObjectTracker:
    """ Manages identification and tracking of objects 
        across multiple frames over time. """
    history = []
    temp_objects = []
    object_log = []
    
    ### Object matching parameter

    iou_threshold = 0.8

    ### Define recent history
    time_limit = 5
    frame_limit = 10

    ### Minimum lifetime required for output logging
    live_time_limit = 2

    def get_id(self):
        taken = set([obj.id for frame in self.history for obj in frame.objects if obj.id is not None])
        taken = taken.union(set([obj.id for obj in self.temp_objects if obj.id is not None]))
        #print("taken ids: {}".format(','.join([str(name) for name in taken])))   
        for i in range(100,1000,100):
            #print("checking ids from {} to {}".format(i-100,i))
            new = set(range(i-100,i))
            free = list(new-taken)
            if len(free) > 0:
                return free[0]


    class Frame:
        objects = None

        @property
        def new_objects(self):
            return [obj for obj in self.objects if obj.new_observation]    

        def __init__(self,frame_dict,tracker):
            self.image_id  = frame_dict['image_id']
            self.camera_id = frame_dict['camera_id']
            self.id        = frame_dict['id']

            self.timestamp = float(frame_dict['date_created'])
            self.face_count = int(frame_dict['no_faces'])
            self.objects = [tracker.TrackingObject(bbox,self.timestamp,tracker) for bbox in self._convert_to_frame_of_boxes(frame_dict['windows']) ]

        def _convert_to_frame_of_boxes(self,array):
            """ for any-lenth array or list, give a list of tuples where each tuple is length 4"""
            #print("   input array {}".format(array))
            if type(array) == str:
                array = array.split(',')
            if len(array) <4:
                return []
            output = []
            unfinished = []
            for item in array:
                unfinished.append(int(item))
                if len(unfinished) == 4:
                    output.append(unfinished)
                    unfinished = []
            return output
        
        def __repr__(self):
            report = "frame with {} objects".format(len(self.objects))
            for obj in self.objects:
                report += "\n    "+str(obj)
            return report


    class TrackingObject:
        """ Class represents an object/face located in a Frame """
        @property
        def time_alive(self):
            if self.end_time:
                end = self.end_time
            else:
                end = _time.time()
            return int(end - self.date_created)
        
        def __init__(self,coords,date_created,tracker):
            self.coords = coords
            self.id = tracker.get_id()
            self.date_created = date_created
            self.end_time = date_created
            self.new_observation = True
            tracker.temp_objects.append(self)

        def observe(self,other_obj):
            if other_obj.date_created < self.date_created:
                self.new_observation = False
                self.date_created = other_obj.date_created
                self.id = other_obj.id
            else:
                other_obj.date_created = self.date_created
                other_obj.id = self.id
        
        def __repr__(self):
            report = "id: {} - coords: {}".format(self.id,','.join([str(x) for x in self.coords]))
            report += "\n     lived {} secs ({} to {})\n".format(self.time_alive,self.date_created,self.end_time)
            return report


    class LogObject:
        """ Class to format and serialize for output logging """
        def __init__(self,trackingObject,frame):
            self.latest_coords = trackingObject.coords
            self.date_created = trackingObject.date_created
            self.end_time = trackingObject.end_time
            self.time_alive = trackingObject.time_alive

            self.image_id  = frame.image_id
            self.camera_id = frame.camera_id
            self.id        = frame.id

        def to_json(self):
            return {'latest_coords': self.latest_coords, 'date_created':self.date_created,'end_time':self.end_time,
                    'time_alive':self.time_alive, 'camera_id':self.camera_id}

        def __repr__(self):
            report = "\nlived {} secs ({} to {})\n".format(self.time_alive,self.date_created,self.end_time)
            report += "  last seen at: {}\n".format(','.join([str(x) for x in self.latest_coords]))
            return report


    def object_match(self,objA,objB):
        """ Given two TrackingObjects, return Boolean match """
        # TODO add momentum estimates etc. for more sophisticated tracking
        boxA = objA.coords
        boxB = objB.coords
        try:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            
            interArea = max(0,xB-xA+1) * max(0,yB-yA+1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou>self.iou_threshold:
                return True
        except:
            pass
        return 0


    def frame_update(self,frameA,frameB):
        """ frameB is updated to match objects from frameA. """
        #print("comparing {} new objects with {} old objects".format(len(frameB.new_objects),len(frameA.objects)))
        no_matches = []
        for obja in frameA.objects:
            for objb in frameB.new_objects:
                if self.object_match(obja,objb):
                    #print("we have a match!")
                    objb.observe(obja)
        
        return frameB


    def recent_history(self,new_frame):
        return [frame for frame in self.history[-self.frame_limit:] if (new_frame.timestamp-frame.timestamp) < self.time_limit]


    def process_new_frame(self,new_frame):
        """ History must be a list between length 0-5. new_frame must have 0 or more boxes in it.
            Return new history and list of objects discovered from history analysis. """
        if new_frame.get("no_faces") == 0:
            return []

        new_frame = self.Frame(new_frame,self)

        for old_frame in self.recent_history(new_frame):
            new_frame = self.frame_update(old_frame,new_frame)
            if len(new_frame.new_objects) == 0:
                break

        self.history.append(new_frame)

        if len(self.history) > self.frame_limit:
            retiring = self.history[-(self.frame_limit+1)]
            self.log_objects(retiring,self.history[-self.frame_limit:])
        
        self.temp_objects = []
        
    def flush_final_frames(self):
        N = len(self.history)
        lim = min(len(self.history),self.frame_limit)
        for i in range( N-lim , N-1 ):
            self.log_objects(self.history[-i],self.history[-i+1:])
        self.log_objects(self.history[-1])

    def log_objects(self,frame,exclude_frames=[]):
        """ log objects which are not present in exclude_frames """
        obj_ids = [obj.id for obj in frame.objects]
        exc_ids = [obj.id for _frame in exclude_frames for obj in _frame.objects]
        new_object_ids = list(set(obj_ids) - set(exc_ids))

        for obj in frame.objects:
            if obj.id in new_object_ids and obj.time_alive >= self.live_time_limit:
                self.object_log.append(self.LogObject(obj,frame))


def track_objects(json_data):
    tracker = ObjectTracker()

    for frame in json_data:
        tracker.process_new_frame(frame)

    tracker.flush_final_frames()

    return tracker.object_log


