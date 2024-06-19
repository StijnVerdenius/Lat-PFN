from util.persist import create_directory_if_not_exists

import pandas as pd
import typing as T
from datetime import datetime


def get_univariate_series(data, target_var):
    target = data[target_var]
    time_dim = data["date"]

    return pd.DataFrame({"t": time_dim, "v": target})


def preprocess_ett_dataset(
    raw_data_location: str,
    targets: T.List[str],
    save_path: str,
    filename: str,
    split_data: bool = False,

):
    data = pd.read_csv(raw_data_location)

    for target_var in targets:
        if split_data:
            data['YEAR'] = pd.DatetimeIndex(data['date']).year
            data['MONTH'] = pd.DatetimeIndex(data['date']).month

            years = data["YEAR"].unique()
            save_dir = f"{save_path}/{filename}/{target_var.replace(' ', '_')}"
            create_directory_if_not_exists(save_dir)

            for year in years:
                year_data = data[data["YEAR"] == year]
                year_data = year_data.drop(columns=["YEAR"])
                months = year_data["MONTH"].unique()
                for month in months:
                    month_data = year_data[year_data["MONTH"] == month]
                    month_data = month_data.drop(columns=["MONTH"])
                    univariate_data_month = get_univariate_series(month_data, target_var)
                    univariate_data_month.columns = ['t', 'v']
                    univariate_data_month.to_csv(f"{save_path}/{filename}/{target_var.replace(' ', '_')}/{year}_{month}.csv", index=False)
        else:
            univariate_data = get_univariate_series(data, target_var)
            univariate_data.to_csv(f"{save_path}/{target_var.replace(' ', '_')}_full.csv", index=False)

def map_ett_h_real_timeline(time_dim):
    datetime_list = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in time_dim]
    return datetime_list