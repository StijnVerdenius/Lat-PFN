from util.persist import create_directory_if_not_exists

import pandas as pd
import typing as T
from datetime import datetime

def get_univariate_series(data, target_var, split_data):
    target = data[target_var]
    time_dim = data["YEAR"].astype(str) + "-" + data["WEEK"].astype(str)
    if split_data:
        univariate_data = pd.DataFrame({"t": time_dim, "v": target})
    else:
        univariate_data = pd.DataFrame(
            {"t": time_dim, "v": target, "year": data["YEAR"]}
        )

    return univariate_data


def preprocess_illness_dataset(
    raw_data_location: str, target_var: str, save_path: str, split_data: bool = False
):
    data = pd.read_csv(raw_data_location)
    save_dir = f"{save_path}/{target_var.replace(' ', '_')}"
    create_directory_if_not_exists(save_dir)

    if split_data:
        years = data["YEAR"].unique()

        for year in years:
            year_data = data[data["YEAR"] == year]
            univariate_data_year = get_univariate_series(
                year_data, target_var, split_data=split_data
            )
            univariate_data_year.columns = ['t', 'v']
            univariate_data_year.to_csv(
                f"{save_dir}/{year}.csv", index=False
            )
    else:
        univariate_data = get_univariate_series(data, target_var, split_data=split_data)
        univariate_data.to_csv(
            f"{save_path}/{target_var.replace(' ', '_')}_full.csv", index=False
        )

def map_illness_real_timeline(time_dim):
    datetime_list = [datetime.strptime(date, "%Y-%W") for date in time_dim]
    return datetime_list
