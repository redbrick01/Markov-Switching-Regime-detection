import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 판별 결과 통계량 정리
def analyze_regime(data, regime_series, returns_col='abs_ret'):
    data = data.copy()
    data['regime_temp'] = regime_series
    stats = data.groupby('regime_temp')[returns_col].describe()
    counts = data['regime_temp'].value_counts().sort_index()
    ratios = data['regime_temp'].value_counts(normalize=True).sort_index()
    transitions = ((data['regime_temp'] == 1) &(data['regime_temp'].shift(1) == 0)).sum()
    regime_stats = []

    for regime in sorted(counts.index):
        stat = stats.loc[regime]
        row = {
            "regime": regime,
            "count": counts[regime],
            "ratio": ratios[regime],
            "mean": stat["mean"],
            "std": stat["std"],
            "min": stat["min"],
            "max": stat["max"],
        }
        regime_stats.append(row)

    #     print(
    #         f"Regime {regime}: count={row['count']}, "
    #         f"ratio={row['ratio']:.2%}, mean={row['mean']:.4f}, "
    #         f"std={row['std']:.4f}, min={row['min']:.4f}, max={row['max']:.4f} |"
    #     )
    # print(f"regime change count: {transitions}\n")

    results = {
        "transitions": transitions,
        "regimes": regime_stats
    }
    return results

# 시각화
def plot_regime_highlight(data, regime_series, title):

    r = pd.Series(regime_series, index=data.index).astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data.index, data['abs_ret'],
            color='black', linewidth=0.5)
    change = r.ne(r.shift()).fillna(True)
    starts = r.index[change & (r == 1)]
    change_idx = r.index[change]
    next_change_idx = pd.Series(change_idx, index=change_idx).shift(-1)

    for s in starts:
        e = next_change_idx.loc[s]
        if pd.isna(e):
            e = r.index[-1]
        ax.axvspan(s, e, color='red', alpha=0.3, linewidth=0)

    ax.set_title(title)
    ax.set_ylabel('abs_ret')
    ax.grid(alpha=0.3)
    plt.savefig(f"{title}.png", dpi=300)

# Rule1 : Threshold
# 레짐 1의 사후확률 > Threshold    =>  Regime 1
# 레짐 1의 사후확률 < 1-Threshold =>  Regime 0
# 그 외 구간에서는 이전 시점의 레짐 유지
def threshold(data, regime_prob, threshold):

    regime_refined = (regime_prob > threshold).astype(int)
    for t in range(1, len(regime_prob)):
        if regime_prob.iloc[t] > threshold:
            regime_refined.iloc[t] = 1 
        elif regime_prob.iloc[t] < 1-threshold:
            regime_refined.iloc[t] = 0 
        else:
            regime_refined.iloc[t] = regime_refined.iloc[t-1]
    
    # 출력
    title = f"threshold : {threshold} "
    #plot_regime_highlight(data, regime_refined, title)
    stats = analyze_regime(data, regime_refined, returns_col='abs_ret')
    return regime_refined, stats, title


# Rule2 : Jump
# t 시점에서 Regime 1 사후확률의 변화량 Δp_t에 대해,
# Δp_t > δ      =>  Regime 1
# Δp_t < −δ    =>  Regime 0
def jump(data, regime_prob, mean_jump_threshold):
    regime_refined = pd.Series(0, index=data.index)
    jump = regime_prob.diff() 
    for t in range(1, len(jump)):
        if jump.iloc[t] > mean_jump_threshold:
            regime_refined.iloc[t:] = 1 
        elif jump.iloc[t] < -mean_jump_threshold:
            regime_refined.iloc[t:] = 0 
    abs_ret = data['abs_ret'].copy()

    # 출력
    title = f"jump : {mean_jump_threshold}"
    #plot_regime_highlight(data, regime_refined, title)
    stats = analyze_regime(data, regime_refined, returns_col='abs_ret')
    return regime_refined, stats, title


# Rule3 :  Persistence
# 길이가 k 미만인 짧은 레짐 구간이 동일한 레짐 상태로 둘러싸여 있는 경우 해당 구간을 인접 레짐과 병합
def persistence(data, regime_raw, k, title):
    
    def merge(series, k, max_iter=1_000_000):
        s = series.copy().astype(int)
        n = len(s)
        if n == 0 or k <= 1:
            return s

        arr = s.to_numpy()
        change_pos = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        starts = np.r_[0, change_pos]
        ends = np.r_[change_pos, n]
        values = arr[starts].tolist()
        lengths = (ends - starts).astype(int).tolist()
        it = 0
        i = 1
        while i < len(values) - 1:
            it += 1
            if it > max_iter:
                raise RuntimeError("RLE absorb did not converge")

            if lengths[i] < k and values[i - 1] == values[i + 1]:
                lengths[i - 1] = lengths[i - 1] + lengths[i] + lengths[i + 1]
                del values[i:i + 2]
                del lengths[i:i + 2]
                i = max(i - 1, 1)
            else:
                i += 1

        out = np.repeat(np.array(values, dtype=int), np.array(lengths, dtype=int))
        if len(out) != n:
            raise RuntimeError("RLE decode length mismatch")
        return pd.Series(out, index=s.index, name=s.name)

    # persistence 고려 병합
    regime_refined = merge(regime_raw, k)

    # 출력
    title = f"persistence + {title}, k : {k} "
    #plot_regime_highlight(data, regime_refined, title)
    stats = analyze_regime(data, regime_refined, returns_col='abs_ret')
    return regime_refined, stats
