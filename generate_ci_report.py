"""
Generate aggregated metrics with confidence intervals across random seeds
AND convergence line plots comparing algorithms for the same n.

- 각 run(= seed 하나)에 대해 metrics.py 의 compute_run_metrics 로 요약값 계산
- 같은 (env_type, n, algo_name, hyperparams) 그룹 내에서
  여러 seed 의 분포로부터 mean, std, 95% CI 계산
- n=3, n=4 에 대한 요약 테이블과 CI 막대 그래프(PNG) 생성
- 추가로, 같은 env_type, 같은 n 에 대해
  여러 알고리즘/하이퍼파라미터의 수렴 꺾은선 그래프를 생성
"""

import os
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import (
    scan_logs,
    load_progress_csv,
    compute_run_metrics,
)

# ---------------------------------------------------------------------
# 설정: 어떤 metric 들에 대해 CI를 낼지
# ---------------------------------------------------------------------

METRICS_TO_SUMMARIZE = [
    "mean_final_length",
    "success_rate",
    "mean_coverage_ratio",
    "mean_episode_steps",
    "mean_episode_return",
]

METRIC_LABELS = {
    "mean_final_length": "Mean final length",
    "success_rate": "Success rate",
    "mean_coverage_ratio": "Mean coverage ratio",
    "mean_episode_steps": "Mean episode steps",
    "mean_episode_return": "Mean episode return",
}

# 수렴 그래프용: 어떤 metric을 그릴지
CONVERGENCE_METRICS = {
    "final_length": "Final length",
    "coverage_ratio": "Coverage ratio",
    "episode_return": "Episode return",
    # 필요하면 여기에 "success": "Success rate" 등도 추가 가능
}

# x축 후보 (우선순위 순서)
X_AXIS_CANDIDATES = [
    "time/total_timesteps",
    "time/iterations",
    "episode",
]


# ---------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def hyperparams_signature(hp: Dict[str, Any]) -> str:
    """
    hyperparams 딕셔너리를 일정한 문자열로 변환.
    예: {"learning_rate": 0.0007, "gamma": 0.99} -> "gamma=0.99, learning_rate=0.0007"
    """
    if not hp:
        return "default"
    items = sorted(hp.items())
    return ", ".join(f"{k}={v}" for k, v in items)


def normalize_superperm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    SB3 logger 가 기록한 'superperm/xxx' 컬럼을
    metrics.py 의 compute_run_metrics 가 기대하는 이름으로 변환.

    예:
      superperm/final_length -> final_length
      superperm/success -> success
      superperm/coverage_ratio -> coverage_ratio
      superperm/episode_steps -> episode_steps
      superperm/episode_return -> episode_return
      superperm/duplicate_action_ratio -> duplicate_action_ratio
      superperm/new_perms_total -> new_perms_total
    """
    if df.empty:
        return df

    rename_map = {}
    prefix = "superperm/"

    for col in df.columns:
        if col.startswith(prefix):
            bare = col[len(prefix):]
            rename_map[col] = bare

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def choose_x_axis_column(columns: List[str]) -> Optional[str]:
    """
    주어진 컬럼 리스트에서 X_AXIS_CANDIDATES 중 하나를 선택.
    우선순위: time/total_timesteps > time/iterations > episode
    """
    for cand in X_AXIS_CANDIDATES:
        if cand in columns:
            return cand
    return None


# ---------------------------------------------------------------------
# 1. run(=seed) 단위 metrics 수집 (CI용)
# ---------------------------------------------------------------------

def collect_run_metrics(log_root: str = "logs") -> pd.DataFrame:
    """
    모든 run(각 seed)을 읽어서 run-level metrics DataFrame 으로 반환.
    각 row가 하나의 run 이다.
    """
    runs = scan_logs(log_root)
    records: List[Dict[str, Any]] = []

    for run in runs:
        meta = run["metadata"]
        env_type = meta.get("env_type", "unknown")
        n = meta.get("n", None)
        algo = meta.get("algo_name", "unknown")
        hp = meta.get("hyperparams", {})
        hp_sig = hyperparams_signature(hp)
        seed = meta.get("seed", None)
        wall_time = meta.get("wall_time", None)

        df = load_progress_csv(run["progress_path"])
        if df.empty:
            continue

        # superperm/ 프리픽스 제거
        df = normalize_superperm_columns(df)

        m = compute_run_metrics(df)
        if not m:
            continue

        record = {
            "run_name": run["run_name"],
            "env_type": env_type,
            "n": n,
            "algo_name": algo,
            "hyperparams_sig": hp_sig,
            "seed": seed,
            "wall_time": wall_time,
        }
        for k, v in m.items():
            record[k] = v

        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# 2. 그룹 내에서 mean / std / 95% CI 계산
# ---------------------------------------------------------------------

def add_ci_columns(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    k = int(s.shape[0])
    if k == 0:
        return {
            "num_runs": 0,
            "mean": math.nan,
            "std": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }

    mean = float(s.mean())
    if k > 1:
        std = float(s.std(ddof=1))
    else:
        std = 0.0

    se = std / math.sqrt(k) if k > 0 else math.nan
    z = 1.96  # 95% CI

    if math.isnan(se):
        ci_low = ci_high = math.nan
    else:
        ci_low = mean - z * se
        ci_high = mean + z * se

    return {
        "num_runs": k,
        "mean": mean,
        "std": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def summarize_with_ci(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    run_df: 각 row가 하나의 run 인 DataFrame.
    같은 (env_type, n, algo_name, hyperparams_sig) 그룹 안에서
    METRICS_TO_SUMMARIZE 에 대해 CI 계산.

    리턴: long-format DataFrame (하나의 metric 당 하나의 row)
    컬럼:
      env_type, n, algo_name, hyperparams_sig,
      metric, metric_label, group_label,
      num_runs, mean, std, ci95_low, ci95_high
    """
    if run_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["env_type", "n", "algo_name", "hyperparams_sig"]

    for group_values, g in run_df.groupby(group_cols):
        env_type, n, algo, hp_sig = group_values

        for metric in METRICS_TO_SUMMARIZE:
            if metric not in g.columns:
                continue

            ci_info = add_ci_columns(g[metric])
            if ci_info["num_runs"] == 0:
                continue

            row = {
                "env_type": env_type,
                "n": n,
                "algo_name": algo,
                "hyperparams_sig": hp_sig,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "num_runs": ci_info["num_runs"],
                "mean": ci_info["mean"],
                "std": ci_info["std"],
                "ci95_low": ci_info["ci95_low"],
                "ci95_high": ci_info["ci95_high"],
            }
            row["group_label"] = f"{algo}\n{hp_sig}"

            rows.append(row)

    return pd.DataFrame(rows)


def make_wide_ci(ci_df: pd.DataFrame) -> pd.DataFrame:
    """
    long-format CI 결과를 wide-format 으로 변환.
    한 row가 하나의 (env_type, n, algo_name, hyperparams_sig) 이고
    각 metric 에 대해
      <metric>__mean,
      <metric>__std,
      <metric>__ci95_low,
      <metric>__ci95_high
    컬럼을 갖게 된다.
    """
    if ci_df.empty:
        return pd.DataFrame()

    group_cols = ["env_type", "n", "algo_name", "hyperparams_sig"]
    rows = []

    for group_values, g in ci_df.groupby(group_cols):
        env_type, n, algo, hp_sig = group_values
        row: Dict[str, Any] = {
            "env_type": env_type,
            "n": n,
            "algo_name": algo,
            "hyperparams_sig": hp_sig,
        }

        row["num_runs"] = int(g["num_runs"].iloc[0])

        for _, r in g.iterrows():
            metric = r["metric"]
            prefix = metric
            row[f"{prefix}__mean"] = float(r["mean"])
            row[f"{prefix}__std"] = float(r["std"])
            row[f"{prefix}__ci95_low"] = float(r["ci95_low"])
            row[f"{prefix}__ci95_high"] = float(r["ci95_high"])

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. CI bar chart 그리기
# ---------------------------------------------------------------------
    """
    def plot_ci_bars(
        ci_df: pd.DataFrame,
        out_dir: str,
        n_filter: Optional[List[int]] = None,
    ) -> None:
        
        #long-format CI DataFrame 에서
        #(env_type, n) 별, metric 별로 바 차트 + error bar 를 그림.
        #n_filter 가 주어지면 그 n 값들만 그림.
       
        ensure_dir(out_dir)
        if ci_df.empty:
            print("No CI summary data to plot.")
            return

        for (env_type, n), df_env_n in ci_df.groupby(["env_type", "n"]):
            if n_filter is not None and n not in n_filter:
                continue

            for metric, df_m in df_env_n.groupby("metric"):
                if df_m.empty:
                    continue

                labels = df_m["group_label"].tolist()
                x = np.arange(len(labels))
                means = df_m["mean"].to_numpy()
                lows = df_m["ci95_low"].to_numpy()
                highs = df_m["ci95_high"].to_numpy()

                yerr = np.vstack([means - lows, highs - means])

                plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
                plt.bar(x, means, yerr=yerr, capsize=4)
                plt.xticks(x, labels, rotation=45, ha="right")
                plt.ylabel(METRIC_LABELS.get(metric, metric))
                plt.title(
                    f"env={env_type}, n={n} – {METRIC_LABELS.get(metric, metric)} (95% CI)"
                )
                plt.tight_layout()

                fname = f"env_{env_type}_n{n}_{metric}_ci.png"
                out_path = os.path.join(out_dir, fname)
                plt.savefig(out_path)
                plt.close()
                print(f"Saved CI bar plot: {out_path}")
    """
def plot_ci_bars(
    ci_df: pd.DataFrame,
    out_dir: str,
    n_filter: Optional[List[int]] = None,
) -> None:
    """
    long-format CI DataFrame 에서
    (env_type, n) 별, metric 별로 바 차트 + error bar 를 그림.
    n_filter 가 주어지면 그 n 값들만 그림.

    각 막대 위에는 mean 값을 숫자로 표시해서
    그래프만 봐도 수치 비교가 가능하게 함.
    """
    ensure_dir(out_dir)
    if ci_df.empty:
        print("No CI summary data to plot.")
        return

    for (env_type, n), df_env_n in ci_df.groupby(["env_type", "n"]):
        if n_filter is not None and n not in n_filter:
            continue

        for metric, df_m in df_env_n.groupby("metric"):
            if df_m.empty:
                continue

            labels = df_m["group_label"].tolist()
            x = np.arange(len(labels))
            means = df_m["mean"].to_numpy()
            lows = df_m["ci95_low"].to_numpy()
            highs = df_m["ci95_high"].to_numpy()

            # 대칭 error bar 높이
            yerr = np.vstack([means - lows, highs - means])

            # 막대가 많으면 가로 폭을 자동으로 크게
            fig_width = max(6, len(labels) * 0.9)
            plt.figure(figsize=(fig_width, 4))

            bars = plt.bar(x, means, yerr=yerr, capsize=4)

            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel(METRIC_LABELS.get(metric, metric))
            plt.title(
                f"env={env_type}, n={n} – {METRIC_LABELS.get(metric, metric)} (95% CI)"
            )

            # --- 여기서 각 막대 위에 숫자(평균 값) 표시 ---
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                # 막대 위 약간 띄워서 표시
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{mean_val:.3g}",   # 소수점 3자리 유효숫자
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            plt.tight_layout()

            fname = f"env_{env_type}_n{n}_{metric}_ci.png"
            out_path = os.path.join(out_dir, fname)
            plt.savefig(out_path)
            plt.close()
            print(f"Saved CI bar plot: {out_path}")


# ---------------------------------------------------------------------
# 4. n별 요약 테이블 저장
# ---------------------------------------------------------------------

def save_n_specific_tables(
    ci_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    report_dir: str,
) -> None:
    """
    n 값별로 long-format / wide-format 요약 파일 저장.
    특히 n=3, n=4 에 대해서는 보고서 표로 바로 쓸 수 있음.
    """
    if ci_df.empty:
        return

    for n_val, df_n_long in ci_df.groupby("n"):
        df_n_wide = wide_df[wide_df["n"] == n_val] if not wide_df.empty else pd.DataFrame()

        long_path = os.path.join(report_dir, f"ci_summary_long_n{n_val}.csv")
        df_n_long.to_csv(long_path, index=False)
        print(f"Saved CI (long) summary for n={n_val} to {long_path}")

        if not df_n_wide.empty:
            wide_path = os.path.join(report_dir, f"ci_summary_wide_n{n_val}.csv")
            df_n_wide.to_csv(wide_path, index=False)
            print(f"Saved CI (wide) summary for n={n_val} to {wide_path}")


# ---------------------------------------------------------------------
# 5. 수렴 꺾은선 그래프: 에피소드 / 타임스텝별 성능
# ---------------------------------------------------------------------

def collect_progress_with_meta(log_root: str = "logs") -> pd.DataFrame:
    """
    모든 run 의 progress.csv 를 읽어서, episode/타임스텝 단위 데이터에
    메타 정보(env_type, n, algo_name, hyperparams_sig, seed)를 붙인 큰 DataFrame 생성.
    (수렴 꺾은선 그래프용)
    """
    runs = scan_logs(log_root)
    dfs = []

    for run in runs:
        meta = run["metadata"]
        env_type = meta.get("env_type", "unknown")
        n = meta.get("n", None)
        algo = meta.get("algo_name", "unknown")
        hp = meta.get("hyperparams", {})
        hp_sig = hyperparams_signature(hp)
        seed = meta.get("seed", None)

        df = load_progress_csv(run["progress_path"])
        if df.empty:
            continue

        # superperm/ 프리픽스 제거
        df = normalize_superperm_columns(df)

        # x축 컬럼 선택
        x_col = choose_x_axis_column(list(df.columns))
        if x_col is None:
            # 에피소드 / 타임스텝 정보가 없으면 수렴 그래프에는 사용하지 않음
            continue

        df = df.copy()
        df["env_type"] = env_type
        df["n"] = n
        df["algo_name"] = algo
        df["hyperparams_sig"] = hp_sig
        df["seed"] = seed
        df["group_label"] = f"{algo}\n{hp_sig}"
        df["x"] = df[x_col]
        df["x_source"] = x_col  # 나중에 동일한 x축 기준만 모으기 위해 저장

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def plot_convergence_curves(
    prog_df: pd.DataFrame,
    out_dir: str,
    n_filter: Optional[List[int]] = None,
) -> None:
    """
    같은 env_type, 같은 n 에 대해
    여러 알고리즘/하이퍼파라미터의 수렴 꺾은선 그래프를 그림.

    x축: time/total_timesteps (있으면) 또는 다른 후보
    y축: CONVERGENCE_METRICS 에 정의된 metric 들
    """
    ensure_dir(out_dir)
    if prog_df.empty:
        print("No progress data available for convergence plots.")
        return

    for (env_type, n), df_env_n in prog_df.groupby(["env_type", "n"]):
        if n_filter is not None and n not in n_filter:
            continue

        # 이 (env_type, n) 에서 사용 가능한 x축 타입들
        x_sources = df_env_n["x_source"].value_counts().index.tolist()
        if not x_sources:
            continue

        # 가장 많이 쓰인 x_source 를 기준으로 통일 (보통 time/total_timesteps 일 것)
        x_source = x_sources[0]
        df_env_n = df_env_n[df_env_n["x_source"] == x_source]

        if df_env_n.empty:
            continue

        for metric, metric_label in CONVERGENCE_METRICS.items():
            if metric not in df_env_n.columns:
                continue

            # metric 값이 있는 행만
            df_metric = df_env_n[~df_env_n[metric].isna()]
            if df_metric.empty:
                continue

            plt.figure(figsize=(7, 4))

            # 알고리즘+하이퍼파라미터 그룹별로 한 꺾은선씩
            for group_label, g in df_metric.groupby("group_label"):
                # 같은 x 에 여러 row가 있을 수 있으니 x별 평균을 사용 (seed/step 평균)
                g_mean = g.groupby("x")[metric].mean().sort_index()
                x_vals = g_mean.index.to_numpy()
                y_vals = g_mean.to_numpy()

                plt.plot(x_vals, y_vals, label=group_label)

            plt.xlabel(x_source)
            plt.ylabel(metric_label)
            plt.title(f"env={env_type}, n={n} – {metric_label} convergence")
            plt.legend(fontsize=8)
            plt.tight_layout()

            fname = f"env_{env_type}_n{n}_{metric}_convergence.png"
            out_path = os.path.join(out_dir, fname)
            plt.savefig(out_path)
            plt.close()
            print(f"Saved convergence plot: {out_path}")


# ---------------------------------------------------------------------
# 6. 전체 파이프라인
# ---------------------------------------------------------------------

def generate_ci_report(
    log_root: str = "logs",
    report_dir: str = "report_ci",
    n_plot: Optional[List[int]] = None,
) -> None:
    ensure_dir(report_dir)

    # 1) run-level metrics (seed 단위 요약) 수집
    print("[1/5] Collecting run-level metrics...")
    run_df = collect_run_metrics(log_root=log_root)
    if run_df.empty:
        print("No runs found or no valid metrics.")
        return

    run_metrics_path = os.path.join(report_dir, "run_metrics_raw.csv")
    run_df.to_csv(run_metrics_path, index=False)
    print(f"Saved raw run metrics to {run_metrics_path}")

    # 2) seed 들 묶어서 CI 계산
    print("[2/5] Aggregating across seeds with confidence intervals...")
    ci_df = summarize_with_ci(run_df)
    if ci_df.empty:
        print("No CI summary generated.")
        return

    ci_summary_long_path = os.path.join(report_dir, "ci_summary_long_all.csv")
    ci_df.to_csv(ci_summary_long_path, index=False)
    print(f"Saved CI summary (long, all n) to {ci_summary_long_path}")

    wide_df = make_wide_ci(ci_df)
    if not wide_df.empty:
        ci_summary_wide_path = os.path.join(report_dir, "ci_summary_wide_all.csv")
        wide_df.to_csv(ci_summary_wide_path, index=False)
        print(f"Saved CI summary (wide, all n) to {ci_summary_wide_path}")

    # 3) n 별 long/wide 요약 파일 생성 (특히 n=3,4)
    print("[3/5] Saving per-n summary tables (including n=3,4)...")
    save_n_specific_tables(ci_df, wide_df, report_dir=report_dir)

    # 4) CI 막대 그래프 (알고리즘/하이퍼파라미터 비교)
    print("[4/5] Plotting CI bar charts...")
    plots_dir = os.path.join(report_dir, "ci_plots")
    if n_plot is None:
        n_plot = [3, 4]
    plot_ci_bars(ci_df, out_dir=plots_dir, n_filter=n_plot)

    # 5) 수렴 꺾은선 그래프 (같은 n 에서 여러 알고리즘을 한 그림에)
    print("[5/5] Plotting convergence curves...")
    prog_df = collect_progress_with_meta(log_root=log_root)
    conv_dir = os.path.join(report_dir, "convergence_plots")
    plot_convergence_curves(prog_df, out_dir=conv_dir, n_filter=n_plot)

    print("Done: CI report + convergence plots generated.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs",
        help="Root directory containing run log folders",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="report_ci",
        help="Directory to save CI tables and plots",
    )
    parser.add_argument(
        "--n_plot",
        type=int,
        nargs="*",
        default=[3, 4,5,6],
        help="Values of n to plot CI & convergence charts for (default: 3 4)",
    )
    args = parser.parse_args()

    generate_ci_report(
        log_root=args.log_root,
        report_dir=args.report_dir,
        n_plot=args.n_plot,
    )
