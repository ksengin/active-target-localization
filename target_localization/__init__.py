from gym.envs.registration import register
import logging

logger = logging.getLogger(__name__)

register(
    id='TrackingWaypoints-v0',
    entry_point='target_localization.envs:TrackingWaypointsEnvInterface',
)
