from pathlib import Path

from util import get_input_name, get_gvkey_iid, get_infer, get_input
from plot import plots

# Infer Loading and Preprocessing
ETF = "NVQ"
TICKER = "CI"
infer_path = Path(
    f"/home/jubro/ETF/sr-storage/ETF_historical_Infer/infer_{ETF}/"
)  # where historical infers are stored
input_path = Path(
    f"/home/jubro/ETF/decomposition/ETF_input_analysis/{ETF}/input_data_raw.pkl"
)
input_winsorize = (True, 0.02)
save_path = Path("./plots/winsorized/")

# infer & dates loading
infer, dates = get_infer(infer_path)
gvkey_iids = list(infer.columns)

# Input data loading
inputs = get_input(
    input_path=input_path,
    gvkey_iids=gvkey_iids,
    dates=dates,
    input_winsorize=input_winsorize,
)
# Input real names
input_names = get_input_name(ETF)

# gvkey to ticker mapping
gvkey_iid = get_gvkey_iid(ticker=TICKER)


if __name__ == "__main__":
    plots(
        inputs=inputs,
        infer=infer,
        input_names=input_names,
        dates=dates,
        target_gvkey=gvkey_iid,
        save_path=save_path,
    )
    print("HELLO ")
