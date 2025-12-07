"""Aggregate metrics and extract top-3 superpermutations."""

import os
import json
import csv
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from envs.utils import canonicalize_superperm


def scan_logs(log_root: str = "logs") -> List[Dict[str, Any]]:
    """
    Scan logs directory for all run directories.
    
    Returns:
        List of run metadata dicts
    """
    runs = []
    log_path = Path(log_root)
    
    if not log_path.exists():
        print(f"Log directory {log_root} does not exist.")
        return runs
    
    for run_dir in log_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        metadata_path = run_dir / "metadata.json"
        progress_path = run_dir / "progress.csv"
        
        if not progress_path.exists():
            continue
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        runs.append({
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "metadata": metadata,
            "progress_path": str(progress_path),
        })
    
    return runs


def load_progress_csv(csv_path: str) -> pd.DataFrame:
    """Load progress CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()


def extract_sequences_from_progress(df: pd.DataFrame, run_dir: str) -> List[Dict[str, Any]]:
    """
    Extract sequences from progress CSV.
    
    Note: progress.csv may not have sequences directly.
    We'll need to check if sequences are stored elsewhere or reconstruct from episodes.
    For now, we'll return empty list and note that sequences should be extracted
    from episode info during training.
    """
    sequences = []
    
    # If progress.csv has sequence column, use it
    if "sequence" in df.columns:
        for idx, row in df.iterrows():
            if row.get("success", 0) == 1:
                try:
                    seq_str = row["sequence"]
                    if isinstance(seq_str, str):
                        # Try to parse as Python literal (list)
                        seq = ast.literal_eval(seq_str)
                    else:
                        seq = seq_str
                    sequences.append({
                        "episode": int(row.get("episode", idx)),
                        "final_length": int(row.get("final_length", 0)),
                        "episode_return": float(row.get("episode_return", 0.0)),
                        "sequence": seq,
                    })
                except (ValueError, SyntaxError, TypeError):
                    # Skip if parsing fails
                    pass
    
    return sequences


def extract_sequences_from_episodes(run_dir: str, n: int) -> List[Dict[str, Any]]:
    """
    Extract sequences from episode logs.
    
    Looks for episode JSON files in the episodes/ subdirectory.
    """
    sequences = []
    run_path = Path(run_dir)
    
    # Check for episode files in episodes/ subdirectory
    episodes_dir = run_path / "episodes"
    if episodes_dir.exists():
        episode_files = list(episodes_dir.glob("episode_*.json"))
        for ep_file in sorted(episode_files):
            try:
                with open(ep_file, "r") as f:
                    ep_data = json.load(f)
                    if ep_data.get("success", False) and "sequence" in ep_data:
                        sequences.append({
                            "episode": ep_data.get("episode", 0),
                            "final_length": ep_data.get("final_length", 0),
                            "episode_return": ep_data.get("episode_return", 0.0),
                            "sequence": ep_data["sequence"],
                        })
            except Exception as e:
                print(f"Error reading {ep_file}: {e}")
                pass
    
    return sequences


def get_top3_superperms(
    sequences: List[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    """
    Get top-3 shortest distinct superpermutations.
    
    Args:
        sequences: List of sequence dicts with keys: episode, final_length, episode_return, sequence
        n: Alphabet size
        
    Returns:
        List of top-3 superpermutation dicts
    """
    if not sequences:
        return []
    
    # Canonicalize and deduplicate
    seen = set()
    unique_seqs = []
    
    for seq_data in sequences:
        seq = seq_data["sequence"]
        if not isinstance(seq, list):
            continue
        
        canonical = tuple(canonicalize_superperm(seq, n))
        if canonical not in seen:
            seen.add(canonical)
            unique_seqs.append({
                "canonical_sequence": list(canonical),
                "original_sequence": seq,
                "final_length": seq_data["final_length"],
                "episode_return": seq_data.get("episode_return", 0.0),
                "episode": seq_data.get("episode", 0),
            })
    
    # Sort by final_length, then by episode_return (descending)
    unique_seqs.sort(key=lambda x: (x["final_length"], -x["episode_return"]))
    
    # Take top 3
    top3 = unique_seqs[:3]
    
    # Add rank
    for i, item in enumerate(top3, 1):
        item["rank"] = i
    
    return top3


def compute_run_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregated metrics for a run."""
    if df.empty:
        return {}
    
    metrics = {}
    
    # Final length statistics
    if "final_length" in df.columns:
        metrics["mean_final_length"] = float(df["final_length"].mean())
        metrics["min_final_length"] = float(df["final_length"].min())
        metrics["max_final_length"] = float(df["final_length"].max())
        metrics["std_final_length"] = float(df["final_length"].std())
    
    # Success rate
    if "success" in df.columns:
        success_count = int(df["success"].sum())
        total_episodes = len(df)
        metrics["success_rate"] = float(success_count / total_episodes) if total_episodes > 0 else 0.0
        metrics["success_count"] = success_count
        metrics["total_episodes"] = total_episodes
    
    # Coverage ratio
    if "coverage_ratio" in df.columns:
        metrics["mean_coverage_ratio"] = float(df["coverage_ratio"].mean())
    
    # Episode steps
    if "episode_steps" in df.columns:
        metrics["mean_episode_steps"] = float(df["episode_steps"].mean())
    
    # Episode return
    if "episode_return" in df.columns:
        metrics["mean_episode_return"] = float(df["episode_return"].mean())
        metrics["std_episode_return"] = float(df["episode_return"].std())
        metrics["min_episode_return"] = float(df["episode_return"].min())
        metrics["max_episode_return"] = float(df["episode_return"].max())
    
    # Steps to first success
    if "success" in df.columns and "episode_steps" in df.columns:
        success_episodes = df[df["success"] == 1]
        if not success_episodes.empty:
            first_success_idx = success_episodes.index[0]
            metrics["episodes_to_first_success"] = int(first_success_idx)
            # Cumulative steps to first success
            metrics["steps_to_first_success"] = int(df.loc[:first_success_idx, "episode_steps"].sum())
        else:
            metrics["episodes_to_first_success"] = None
            metrics["steps_to_first_success"] = None
    
    return metrics


def update_metrics_for_run(
    run_dir: str,
    log_root: str = "logs",
    output_dir: str = "metrics",
) -> None:
    """
    Update metrics for a single run (incremental update).
    
    This function updates the aggregated metrics after a single run completes.
    It reads existing metrics and updates them with the new run's data.
    
    Args:
        run_dir: Directory of the completed run
        log_root: Root directory containing run logs
        output_dir: Directory to save aggregated metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata and progress for this run
    run_path = Path(run_dir)
    metadata_path = run_path / "metadata.json"
    progress_path = run_path / "progress.csv"
    
    if not progress_path.exists():
        return
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    env_type = metadata.get("env_type", "unknown")
    n = metadata.get("n", 0)
    algo_name = metadata.get("algo_name", "unknown")
    hyperparams = metadata.get("hyperparams", {})
    
    # Create group key
    hp_sig = "_".join(f"{k}{v}" for k, v in sorted(hyperparams.items())) if hyperparams else "default"
    group_key = (env_type, n, algo_name, hp_sig)
    group_name = f"{env_type}_n{n}_{algo_name}_{hp_sig}"
    
    # Load existing summary if exists
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    existing_summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            existing_summaries = json.load(f)
    
    # Find existing summary for this group
    existing_summary = None
    existing_idx = None
    for idx, summary in enumerate(existing_summaries):
        if (summary.get("env_type") == env_type and
            summary.get("n") == n and
            summary.get("algo_name") == algo_name and
            summary.get("hyperparams") == hp_sig):
            existing_summary = summary
            existing_idx = idx
            break
    
    # Get all runs in this group
    all_runs = scan_logs(log_root)
    group_runs = []
    for run in all_runs:
        run_meta = run["metadata"]
        run_env_type = run_meta.get("env_type", "unknown")
        run_n = run_meta.get("n", 0)
        run_algo = run_meta.get("algo_name", "unknown")
        run_hp = run_meta.get("hyperparams", {})
        run_hp_sig = "_".join(f"{k}{v}" for k, v in sorted(run_hp.items())) if run_hp else "default"
        
        if (run_env_type == env_type and run_n == n and 
            run_algo == algo_name and run_hp_sig == hp_sig):
            group_runs.append(run)
    
    # Aggregate metrics for this group
    all_metrics = []
    all_sequences = []
    
    for run in group_runs:
        df = load_progress_csv(run["progress_path"])
        if df.empty:
            continue
        
        # Compute run-level metrics
        run_metrics = compute_run_metrics(df)
        run_metrics["run_name"] = run["run_name"]
        run_metrics["seed"] = run["metadata"].get("seed", None)
        run_metrics["wall_time"] = run["metadata"].get("wall_time", None)
        all_metrics.append(run_metrics)
        
        # Extract sequences
        sequences = extract_sequences_from_progress(df, run["run_dir"])
        if not sequences:
            sequences = extract_sequences_from_episodes(run["run_dir"], n)
        
        for seq in sequences:
            seq["run_name"] = run["run_name"]
            all_sequences.append(seq)
    
    if not all_metrics:
        return
    
    # Aggregate across runs
    metrics_df = pd.DataFrame(all_metrics)
    
    summary = {
        "env_type": env_type,
        "n": n,
        "algo_name": algo_name,
        "hyperparams": hp_sig,
        "num_runs": len(group_runs),
        "mean_final_length": float(metrics_df["mean_final_length"].mean()) if "mean_final_length" in metrics_df.columns else None,
        "min_final_length": float(metrics_df["min_final_length"].min()) if "min_final_length" in metrics_df.columns else None,
        "max_final_length": float(metrics_df["max_final_length"].max()) if "max_final_length" in metrics_df.columns else None,
        "mean_success_rate": float(metrics_df["success_rate"].mean()) if "success_rate" in metrics_df.columns else None,
        "mean_coverage_ratio": float(metrics_df["mean_coverage_ratio"].mean()) if "mean_coverage_ratio" in metrics_df.columns else None,
        "mean_episode_steps": float(metrics_df["mean_episode_steps"].mean()) if "mean_episode_steps" in metrics_df.columns else None,
        "mean_episode_return": float(metrics_df["mean_episode_return"].mean()) if "mean_episode_return" in metrics_df.columns else None,
        "total_wall_time": float(metrics_df["wall_time"].sum()) if "wall_time" in metrics_df.columns else None,
    }
    
    # Update or add summary
    if existing_idx is not None:
        existing_summaries[existing_idx] = summary
    else:
        existing_summaries.append(summary)
    
    # Save updated summary
    with open(summary_path, "w") as f:
        json.dump(existing_summaries, f, indent=2)
    
    # Save CSV version
    summary_df = pd.DataFrame(existing_summaries)
    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # Extract and save top-3 superpermutations for this group
    top3 = get_top3_superperms(all_sequences, n)
    if top3:
        top3_path = os.path.join(output_dir, f"{group_name}_top3_superperms.json")
        with open(top3_path, "w") as f:
            json.dump(top3, f, indent=2)
        print(f"Updated metrics and top-3 for {group_name} ({len(group_runs)} runs)")


def aggregate_metrics(log_root: str = "logs", output_dir: str = "metrics") -> None:
    """
    Aggregate metrics from all runs and extract top-3 superpermutations.
    
    Args:
        log_root: Root directory containing run logs
        output_dir: Directory to save aggregated metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    runs = scan_logs(log_root)
    print(f"Found {len(runs)} runs")
    
    # Group runs by (env_type, n, algo_name, hyperparams)
    grouped_runs = defaultdict(list)
    
    for run in runs:
        metadata = run["metadata"]
        env_type = metadata.get("env_type", "unknown")
        n = metadata.get("n", 0)
        algo_name = metadata.get("algo_name", "unknown")
        hyperparams = metadata.get("hyperparams", {})
        
        # Create group key
        hp_sig = "_".join(f"{k}{v}" for k, v in sorted(hyperparams.items())) if hyperparams else "default"
        group_key = (env_type, n, algo_name, hp_sig)
        
        grouped_runs[group_key].append(run)
    
    # Aggregate metrics for each group
    summary_rows = []
    
    for group_key, group_runs in grouped_runs.items():
        env_type, n, algo_name, hp_sig = group_key
        
        # Aggregate metrics across all runs in group
        all_metrics = []
        all_sequences = []
        
        for run in group_runs:
            df = load_progress_csv(run["progress_path"])
            if df.empty:
                continue
            
            # Compute run-level metrics
            run_metrics = compute_run_metrics(df)
            run_metrics["run_name"] = run["run_name"]
            run_metrics["seed"] = run["metadata"].get("seed", None)
            run_metrics["wall_time"] = run["metadata"].get("wall_time", None)
            all_metrics.append(run_metrics)
            
            # Extract sequences (try multiple methods)
            sequences = extract_sequences_from_progress(df, run["run_dir"])
            if not sequences:
                sequences = extract_sequences_from_episodes(run["run_dir"], n)
            
            for seq in sequences:
                seq["run_name"] = run["run_name"]
                all_sequences.append(seq)
        
        if not all_metrics:
            continue
        
        # Aggregate across runs
        metrics_df = pd.DataFrame(all_metrics)
        
        summary = {
            "env_type": env_type,
            "n": n,
            "algo_name": algo_name,
            "hyperparams": hp_sig,
            "num_runs": len(group_runs),
            "mean_final_length": float(metrics_df["mean_final_length"].mean()) if "mean_final_length" in metrics_df.columns else None,
            "min_final_length": float(metrics_df["min_final_length"].min()) if "min_final_length" in metrics_df.columns else None,
            "max_final_length": float(metrics_df["max_final_length"].max()) if "max_final_length" in metrics_df.columns else None,
            "mean_success_rate": float(metrics_df["success_rate"].mean()) if "success_rate" in metrics_df.columns else None,
            "mean_coverage_ratio": float(metrics_df["mean_coverage_ratio"].mean()) if "mean_coverage_ratio" in metrics_df.columns else None,
            "mean_episode_steps": float(metrics_df["mean_episode_steps"].mean()) if "mean_episode_steps" in metrics_df.columns else None,
            "mean_episode_return": float(metrics_df["mean_episode_return"].mean()) if "mean_episode_return" in metrics_df.columns else None,
            "total_wall_time": float(metrics_df["wall_time"].sum()) if "wall_time" in metrics_df.columns else None,
        }
        
        summary_rows.append(summary)
        
        # Extract top-3 superpermutations for this group
        top3 = get_top3_superperms(all_sequences, n)
        
        if top3:
            # Save top-3 for this group
            group_name = f"{env_type}_n{n}_{algo_name}_{hp_sig}"
            top3_path = os.path.join(output_dir, f"{group_name}_top3_superperms.json")
            with open(top3_path, "w") as f:
                json.dump(top3, f, indent=2)
            print(f"Saved top-3 superpermutations for {group_name}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved metrics summary to {summary_path}")
    
    # Also save as JSON
    summary_json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Saved metrics summary to {summary_json_path}")


if __name__ == "__main__":
    aggregate_metrics()

