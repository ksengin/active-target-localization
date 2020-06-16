from gym import utils
from target_localization.envs.tracking_waypoints_env import TrackingWaypointsEnv
import numpy as np
import torch

class TrackingWaypointsEnvInterface(TrackingWaypointsEnv, utils.EzPickle):
    def __init__(self):
        TrackingWaypointsEnv.__init__(self)
        utils.EzPickle.__init__(self)
