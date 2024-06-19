from datetime import datetime


def map_traffic_real_timeline(time_dim, split=False):
    if not split:
        datetime_list = [datetime.strptime(date, "%Y-%m-%d") for date in time_dim]
    else:
        datetime_list = [datetime.strptime(date.split(" ")[0], "%Y-%m-%d") for date in time_dim]
    return datetime_list
