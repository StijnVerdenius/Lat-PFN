import typing as T
import pandas as pd
import numpy as np
import torch
import datetime


def cache_context(data_sources: T.List[str], save_path: str, filename: str = None):
    context = []
    for source in data_sources:
        entity = source.split("/")[-1].split(".")[0]
        data = pd.read_csv(source)
        data.set_index([[entity] * len(data)], inplace=True)
        data.index.names = ["entity"]
        context.append(data)
    context = pd.concat(context).reset_index()
    context.to_csv(
        f"{save_path}/{filename if filename is not None else 'context'}.csv",
        index=False,
    )


def calendar_embedding(ts: np.ndarray, device="cuda") -> torch.Tensor:
    """
    Maps the time to a calendar embedding
    """
    if type(ts[0]) == datetime.datetime:
        year = [x.year for x in ts]
        month = [x.month for x in ts]
        day = [x.day for x in ts]
        day_of_week = [x.weekday() + 1 for x in ts]
        day_of_year = [x.timetuple().tm_yday for x in ts]
        return np.stack([year, month, day, day_of_week, day_of_year], axis=-1)
    ts = pd.to_datetime(ts)

    return torch.from_numpy(
        np.stack(
            [ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1
        )
    ).to(device)

def smooth(series, smoother):
    assert smoother % 2 == 1, "smoother must be odd"
    pad = smoother // 2

    series = torch.nn.functional.pad(series, (0, 0, pad, pad, 0, 0), mode="replicate")

    return series.unfold(2, smoother, 1).float().mean(-1)