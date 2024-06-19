from .illness import map_illness_real_timeline
from .ett import map_ett_h_real_timeline
from .traffic import map_traffic_real_timeline

from functools import partial

real_time_maps = {
    "illness": map_illness_real_timeline,
    "ett_h_1": map_ett_h_real_timeline,
    "ett_h_2": map_ett_h_real_timeline,
    "traffic": map_traffic_real_timeline,
    "etl": partial(map_traffic_real_timeline, split=True)
}