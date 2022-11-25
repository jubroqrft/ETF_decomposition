import pandas as pd
import numpy as np
import scipy.stats as stats
import imageio
import matplotlib.pyplot as plt

from typing import List
from tqdm import tqdm
from pathlib import Path

from qraft_data.data import QraftData


def plots(
    inputs: List[QraftData],
    infer: pd.DataFrame,
    input_names: dict,
    dates: List[pd.Timestamp],
    target_gvkey: str,
    save_path: Path,
):

    # png
    col_names = list(inputs.keys())
    png_path = save_path / "png"
    gif_path = save_path / "gif"

    png_path.mkdir(parents=True, exist_ok=True)
    gif_path.mkdir(parents=True, exist_ok=True)

    # color list
    col_ind = list(infer.columns).index(target_gvkey)
    col_list = ["cornflowerblue" for _ in range(len(infer.columns))]
    col_list[col_ind] = "red"

    # Correlation

    yes_ls = []
    corr_ls = []
    corr_dict = {}

    print("Png files are being generated...")
    for name in tqdm(col_names):
        corr_avg = []

        for index, df in inputs[name].iterrows():
            date = index
            _input = df.values.reshape(-1, 1)
            _infer = infer.loc[date].values.reshape(-1, 1)

            # Correlation
            nas = np.logical_or(np.isnan(_infer), np.isnan(_input))
            res = stats.pearsonr(_infer[~nas], _input[~nas])
            corr = round(res[0], 3)
            # pvalue = res[1]
            corr_avg.append(corr)

            name2 = input_names[name]
            date2 = str(date)[:10]

            plt.scatter(_infer, _input, c=col_list)
            plt.axhline(y=np.nanmean(_input))
            plt.axvline(x=np.nanmean(_infer))

            plt.title(f"{name2} | {date2} | Corr: {corr}")
            plt.ylabel(f"{name2} value")
            plt.xlabel("infer")

            plt.savefig(f"{png_path}/{name}_{date2}.png")
            plt.close()

        corr_avg = np.mean(corr_avg)
        df.corr_avg = corr_avg
        corr_ls.append(corr_avg)
        yes_ls.append(df)
        corr_dict[name] = corr_avg

    # gif
    print("Gif files are being generated...")
    for name in tqdm(col_names):
        filenames = [f"{png_path}/{name}_{str(d)[:10]}.png" for d in dates]
        with imageio.get_writer(
            f"{gif_path}/{name}.gif", mode="I", fps=1
        ) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
