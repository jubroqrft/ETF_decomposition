import pandas as pd
import numpy as np
import pickle
from typing import List, Tuple
from pathlib import Path
from qraft_data.data import QraftData
from kirin import Kirin


def get_infer(infer_path):
    child_files = sorted(infer_path.rglob("*.csv"))

    dates = [
        pd.to_datetime(file.parts[-1].split(".")[0].split("_")[-1])
        for file in child_files
    ]
    files = [pd.read_csv(file, index_col=0) for file in (child_files)]
    infer = pd.concat(files, axis=0)
    infer.index = dates
    return infer, dates


def get_input(
    input_path: Path,
    gvkey_iids: List[str],
    dates: List[pd.Timestamp],
    input_winsorize: Tuple[bool, float],
):

    with open(input_path, "rb") as f:
        inputs = pickle.load(f)

    if_winsorize, winsorize_n = input_winsorize

    if if_winsorize:
        inputs = _winsorize(
            gvkey_iids=gvkey_iids,
            inputs=inputs,
            dates=dates,
            winsorize_n=winsorize_n,
        )

    return inputs


def _winsorize(
    gvkey_iids: List[str],
    inputs: List[QraftData],
    dates: List[pd.Timestamp],
    winsorize_n: float = 0.02,
):
    START_DATE = dates[0]
    END_DATE = dates[-1]
    _inputs = {}
    for data in inputs:
        try:
            if (data.get_tag() == "equity") and (data.name != "sector"):
                _data = data.copy()
                arr = data.values
                mn = np.nanquantile(arr, winsorize_n, axis=1, keepdims=True)
                mx = np.nanquantile(arr, 1 - winsorize_n, axis=1, keepdims=True)
                _data[:] = np.clip(arr, mn, mx)

                _inputs[data.name] = QraftData(
                    data.name, _data.loc[START_DATE:END_DATE][gvkey_iids]
                )
        except KeyError:
            pass

    return _inputs


def get_gvkey_iid(ticker: str = "CI"):
    api = Kirin()
    meta = api.compustat.set_investment_universe()
    meta2 = meta.sort_values(["gvkey_iid", "effdate"], ascending=False)
    gg = meta2.groupby("gvkey_iid")["tic"].first()

    gvkey_iid = gg[gg == ticker].index[0]
    return gvkey_iid


def get_input_name(etf_name: str):
    etf_name = etf_name.lower()
    assert etf_name in ["amom", "qrft", "nvq", "hdiv"]

    n_dict = {}

    if etf_name == "nvq":
        n_dict["pr_1m_0m"] = "Monthly Price Return"
        n_dict["mv"] = "Monthly Market Value"
        n_dict["btm"] = "Book to Market"
        n_dict["mom_12m_1m"] = "12-1M Momentum"
        n_dict["ram_12m_0m"] = "12M Risk-adjusted Momentum"
        n_dict["vol_3m"] = "3M Rolling Volatility"
        n_dict["res_mom_12m_1m_0m"] = "Residual Momentum"
        n_dict["res_vol_6m_3m_0m"] = "Residual Volatility"
        n_dict["at"] = "Asset Turnover"
        n_dict["gpa"] = "Gross Profits to Assets"
        n_dict["rev_surp"] = "Revenue Surprise"
        n_dict["cash_at"] = "Cash to Asset"
        n_dict["op_lev"] = "Operating Leverage"
        n_dict["roe"] = "Return on Equity"
        n_dict["std_u_e"] = "Standardized Unexpected Earnings"

        n_dict["ret_noa"] = "Return on Net Operating Asset"
        n_dict["etm"] = "Earnings to Market"
        n_dict["ia_mv"] = "Intangible Assets to Market Value"
        n_dict["ae_m"] = "Advertising Expense to Market"
        n_dict["ia_ta"] = "Intangible Assets to Total Assets"
        n_dict["rc_a"] = "R&D(Research & Development) Capital to Assets"
        n_dict["r_s"] = "R&D to Sales"
        n_dict["r_a"] = "R&D to Assets"

    elif etf_name == "qrft":
        pass

    elif etf_name == "amom":
        pass

    elif etf_name == "hdiv":
        pass

    return n_dict
