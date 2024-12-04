import numpy as np

# vim: expandtab:ts=4:sw=4
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    original_ltwh : Optional List
        Bounding box associated with matched detection
    det_class : Optional str
        Classname of matched detection
    det_conf : Optional float
        Confidence associated with matched detection
    instance_mask : Optional 
        Instance mask associated with matched detection
    others : Optional any
        Any supplementary fields related to matched detection

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurrence.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        feature=None,
        original_ltwh=None,
        det_class=None,
        det_conf=None,
        bbox_id=None,
        instance_mask=None,
        others=None,
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Confirmed  # we don't want to wait, mark them confirmed
        self.features = []
        self.latest_feature = None
        if feature is not None:
            self.features.append(feature)
            self.latest_feature = feature

        self._n_init = n_init
        self._max_age = max_age

        self.original_ltwh = original_ltwh
        self.det_class = det_class
        self.det_conf = 0.0  # when we create new track, it has score of 0
        self.bbox_id = bbox_id
        self.instance_mask = instance_mask
        self.others = others

        self.last_seen_ltrb = self.to_ltrb(orig=True, orig_strict=True)
        self.ltrb_history = []
        if self.last_seen_ltrb is not None:
            self.ltrb_history.append(self.last_seen_ltrb)
        self.det_class_history = [self.det_class]
        self.bbox_id_history = [self.bbox_id]
        self.status = 'new'
        self.switch = None


    def to_tlwh(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`. This function is POORLY NAMED. But we are keeping the way it works the way it works in order not to break any older libraries that depend on this.

        Returns
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.
        """
        return self.to_ltwh(orig=orig, orig_strict=orig_strict)

    def to_ltwh(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Params
        ------
        orig : bool
            To use original detection (True) or KF predicted (False). Only works for original dets that are horizontal BBs.
        orig_strict: bool 
            Only relevant when orig is True. If orig_strict is True, it ONLY outputs original bbs and will not output kalman mean even if original bb is not available. 

        Returns
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.

        """
        if orig:
            if self.original_ltwh is None:
                if orig_strict:
                    return None
                # else if not orig_strict, return kalman means below
            else:
                return self.original_ltwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`. This original function is POORLY NAMED. But we are keeping the way it works the way it works in order not to break any older projects that depend on this.
        USE THIS AT YOUR OWN RISK. LIESSSSSSSSSS!
        Returns LIES
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.
        """
        return self.to_ltrb(orig=orig, orig_strict=orig_strict)

    def to_ltrb(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Params
        ------
        orig : bool
            To use original detection (True) or KF predicted (False). Only works for original dets that are horizontal BBs.

        Returns
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.
        """
        ret = self.to_ltwh(orig=orig, orig_strict=orig_strict)
        if ret is not None:
            ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_det_conf(self):
        """
        `det_conf` will be None is there are no associated detection this round
        """
        return self.det_conf

    def get_det_class(self):
        """
        Only `det_class` will be persisted in the track even if there are no associated detection this round.
        """
        return self.det_class

    def get_instance_mask(self):
        '''
        Get instance mask associated with detection. Will be None is there are no associated detection this round
        '''
        return self.instance_mask
    
    def get_det_supplementary(self):
        """
        Get supplementary info associated with the detection. Will be None is there are no associated detection this round.
        """
        return self.others

    def get_feature(self):
        '''
        Get latest appearance feature
        '''
        return self.latest_feature

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.original_ltwh = None
        self.det_conf = 0.0
        self.instance_mask = None
        self.others = None

        self.status = 'missed'

    def update(self, kf, detection, score):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # because of pickling, we don't have these attributes yet
        if not hasattr(self, 'det_class_history'):
            self.det_class_history = []
        if not hasattr(self, 'bbox_id_history'):
            self.bbox_id_history = []

        self.original_ltwh = detection.get_ltwh()
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)
        self.latest_feature = detection.feature
        if detection.class_name is not None:
            self.det_class = detection.class_name
        self.bbox_id = detection.bbox_id
        self.instance_mask = detection.instance_mask
        self.others = detection.others

        self.det_class_history.append(detection.class_name)
        self.bbox_id_history.append(detection.bbox_id)

        self.hits += 1

        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.mark_confirmed()

        self.last_seen_ltrb = self.to_ltrb(orig=True, orig_strict=True)
        self.ltrb_history.append(self.last_seen_ltrb)
        self.status = 'ok'

        self.det_conf = score

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        #         if self.state == TrackState.Tentative:
        #             self.state = TrackState.Deleted
        #         elif self.time_since_update > self._max_age:
        self.state = TrackState.Deleted  # because we don't wand 'missed' tracks, remove them immediately

    def mark_confirmed(self):
        self.state = TrackState.Confirmed

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def to_json(self, round_big_arrays_to=32):
        return {
            'hits': self.hits,
            'age': self.age,
            'time_since_update': self.time_since_update,
            'state': self.state,
            'features': [feat.astype('float64').round(round_big_arrays_to).tolist() for feat in self.features],
            'latest_feature': self.latest_feature.astype('float64').round(round_big_arrays_to).tolist(),
            'last_seen_ltrb': self.last_seen_ltrb.tolist(),
            'ltrb_history': [item.tolist() for item in self.ltrb_history],
            'status': self.status,
            'switch': self.switch,
            'init_kwargs': {
                'mean': self.mean.tolist(),
                'covariance': self.covariance.tolist(),
                'track_id': self.track_id,
                'n_init': self._n_init,
                'max_age': self._max_age,
                'original_ltwh': self.original_ltwh.tolist() if self.original_ltwh is not None else None,
                'det_class': self.det_class,
                'det_conf': self.det_conf,
                'instance_mask': self.instance_mask,
                'others': self.others,
            }
        }

    @staticmethod
    def from_json(data):
        data['init_kwargs']['mean'] = np.array(data['init_kwargs']['mean'])
        data['init_kwargs']['covariance'] = np.array(data['init_kwargs']['covariance'])
        oltwh = data['init_kwargs']['original_ltwh']
        data['init_kwargs']['original_ltwh'] = np.array(oltwh) if oltwh is not None else oltwh
        track_obj = Track(**data['init_kwargs'])

        track_obj.hits = data['hits']
        track_obj.age = data['age']
        track_obj.time_since_update = data['time_since_update']
        track_obj.state = data['state']
        track_obj.features = [np.array(feat) for feat in data['features']]
        track_obj.latest_feature = np.array(data['latest_feature'])
        track_obj.last_seen_ltrb = np.array(data['last_seen_ltrb'])
        track_obj.ltrb_history = [np.array(item) for item in data['ltrb_history']]
        track_obj.status = data['status']
        track_obj.switch = data['switch']
        return track_obj
