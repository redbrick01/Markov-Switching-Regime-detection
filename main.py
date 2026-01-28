import numpy as np
import pandas as pd
import rules
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Settings
ticker = "^GSPC"
start_date = "2006-01-01"
end_date = "2026-01-01"

# Data Load & Preprocess
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
data = data[["Adj Close"]].dropna()

data["ret"] = 100 * np.log(data["Adj Close"] / data["Adj Close"].shift(1))
data["abs_ret"] = np.abs(data["ret"])
data = data.dropna()

# Markov Switching Model
model = MarkovRegression(
    data["abs_ret"],
    k_regimes=2,                    # 2-state
    trend="c",                          # regime별 평균 허용
    switching_variance=True # regime별 분산 허용
)
result = model.fit(maxiter=1000, disp=False)
print(result.summary())
filtered = result.filtered_marginal_probabilities[1]
smoothed = result.smoothed_marginal_probabilities[1]

# Result table
results_threshold = []
results_threshold_persist = []
results_jump = []
results_jump_persist = []

# Parameters
threshold_list = [0.4, 0.5, 0.6, 0.7]
mean_jump_candidates = [0.4, 0.5, 0.6]
k_list = [5, 10, 15, 20]

# Rule 1 + 3 : Threshold, Threshold + Persistence
for threshold in threshold_list:
    regime_refined, stats, title = rules.threshold(data, smoothed, threshold)
    results_threshold.append({"threshold": threshold,**stats})
    for k_val in k_list:
        regime_refined, stats_p = rules.persistence(data, regime_refined, k_val, title)
        results_threshold_persist.append({"threshold": threshold,"k": k_val,**stats_p})

# Rule 2 + 3 : Jump, Jump + Persistence
for mean_jump in mean_jump_candidates:
    regime_refined, stats, title = rules.jump(data, smoothed, mean_jump_threshold=mean_jump)
    results_jump.append({"mean_jump": mean_jump, **stats})
    for k_val in k_list:
        regime_refined, stats_p2 = rules.persistence(data, regime_refined, k_val, title)
        results_jump_persist.append({"mean_jump": mean_jump, "k":k_val, **stats_p2})

# CSV 저장
def flatten_results(results, rule_name):
    rows = []
    for r in results:
        for regime in r["regimes"]:
            row = {
                "rule": rule_name,
                "transitions": r["transitions"],
                **{k: v for k, v in r.items() if k not in ["regimes", "transitions"]},
                **regime
            }
            rows.append(row)
    return pd.DataFrame(rows)
flatten_results(results_threshold, "threshold").to_csv("threshold.csv", index=False)
flatten_results(results_threshold_persist, "threshold+persistence").to_csv("threshold_persistence.csv", index=False)
flatten_results(results_jump, "jump").to_csv("jump.csv", index=False)
flatten_results(results_jump_persist, "jump+persistence").to_csv("jump_persistence.csv", index=False)

