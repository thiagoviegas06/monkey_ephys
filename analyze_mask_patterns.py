#!/usr/bin/env python3
"""Analyze masking patterns in test data to understand span distributions and structure."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_mask_structure(test_dir, session_id):
    """
    Analyze masking patterns for a single session.
    
    Returns:
        dict with keys: session_id, n_trials, n_masked_trials, mask_spans,
                       avg_span, span_distribution, channel_coverage, etc.
    """
    mask_path = Path(test_dir) / f"{session_id}_mask.npy"
    info_path = Path(test_dir) / f"{session_id}_trial_info.npz"
    
    if not mask_path.exists():
        print(f"  {session_id}: mask not found, skipping")
        return None
    
    mask = np.load(str(mask_path))  # (N, 96) bool
    info = np.load(str(info_path))
    starts = info["start_bins"].astype(int)
    ends = info["end_bins"].astype(int)
    
    N, n_channels = mask.shape
    n_trials = len(starts)
    
    # Identify which time bins have any masking
    any_masked = mask.any(axis=1)  # (N,)
    
    # Find contiguous masked regions
    diff = np.diff(any_masked.astype(int))
    starts_masked = np.where(diff == 1)[0] + 1
    ends_masked = np.where(diff == -1)[0] + 1
    
    # Handle edges
    if any_masked[0]:
        starts_masked = np.concatenate([[0], starts_masked])
    if any_masked[-1]:
        ends_masked = np.concatenate([ends_masked, [N]])
    
    mask_spans = list(zip(starts_masked, ends_masked))  # List of (start_bin, end_bin) tuples
    
    # For each contiguous mask region, compute stats
    span_stats = []
    for m_start, m_end in mask_spans:
        span_len = m_end - m_start
        chans_in_span = mask[m_start:m_end].sum(axis=0)  # per-channel count
        n_chans_masked = (chans_in_span > 0).sum()  # how many unique channels
        
        # Find which trial(s) this span overlaps
        overlapping_trials = []
        for t_idx, (t_start, t_end) in enumerate(zip(starts, ends)):
            if m_start < t_end and m_end > t_start:
                overlap_len = min(m_end, t_end) - max(m_start, t_start)
                overlapping_trials.append((t_idx, t_start, t_end, overlap_len))
        
        span_stats.append({
            "start_bin": int(m_start),
            "end_bin": int(m_end),
            "span_len": int(span_len),
            "n_channels_masked": int(n_chans_masked),
            "max_channels_at_bin": int(chans_in_span.max()),
            "overlapping_trials": overlapping_trials,
        })
    
    # Trials with any masking in them
    masked_trials = set()
    for t_idx, (t_start, t_end) in enumerate(zip(starts, ends)):
        if mask[t_start:t_end].any():
            masked_trials.add(t_idx)
    
    n_masked_trials = len(masked_trials)
    
    # Trial-level masking stats + channel analysis
    trial_mask_stats = []
    global_channel_freq = np.zeros(n_channels, dtype=int)
    
    for t_idx, (t_start, t_end) in enumerate(zip(starts, ends)):
        trial_mask = mask[t_start:t_end]
        if trial_mask.any():
            # How many bins in this trial are masked?
            masked_bins = (trial_mask.any(axis=1)).sum()
            trial_span = t_end - t_start
            
            # Which channels are masked in this trial (at any bin)?
            channels_masked = np.flatnonzero(trial_mask.any(axis=0))
            n_chans = len(channels_masked)
            
            # Are the same channels masked throughout, or do they vary per bin?
            # Check if the set of masked channels is constant across all bins
            channel_consistency = "constant"  # default
            if masked_bins > 0:
                masked_per_bin = [frozenset(np.flatnonzero(row)) for row in trial_mask]
                unique_masks = len(set(masked_per_bin))
                if unique_masks > 1:
                    channel_consistency = "varying"
            
            # For each bin in this trial, how many channels are masked?
            channels_per_bin = trial_mask.sum(axis=1)
            
            # Track global channel frequency
            global_channel_freq += trial_mask.sum(axis=0)
            
            trial_mask_stats.append({
                "trial_idx": int(t_idx),
                "trial_start": int(t_start),
                "trial_end": int(t_end),
                "trial_len": int(trial_span),
                "masked_bins": int(masked_bins),
                "n_channels": int(n_chans),
                "pct_masked": float(masked_bins / trial_span),
                "channels_masked": ",".join(str(c) for c in channels_masked),
                "channel_consistency": channel_consistency,
                "avg_channels_per_bin": float(channels_per_bin.mean()),
                "max_channels_in_bin": int(channels_per_bin.max()),
            })
    
    # Channel frequency statistics
    channels_ever_masked = np.flatnonzero(global_channel_freq > 0)
    n_channels_ever_masked = len(channels_ever_masked)
    
    # Distribution of how often each channel is masked
    channel_freq_nonzero = global_channel_freq[channels_ever_masked]
    
    # Count trials with constant vs varying channel masking
    channel_consistency_counts = {"constant": 0, "varying": 0}
    for stat in trial_mask_stats:
        consistency = stat["channel_consistency"]
        if consistency in channel_consistency_counts:
            channel_consistency_counts[consistency] += 1
    
    result = {
        "session_id": session_id,
        "n_bins": int(N),
        "n_trials": int(n_trials),
        "n_masked_trials": int(n_masked_trials),
        "pct_trials_masked": float(100 * n_masked_trials / n_trials) if n_trials > 0 else 0.0,
        "mask_spans": mask_spans,
        "span_stats": span_stats,
        "trial_mask_stats": trial_mask_stats,
        "n_contiguous_regions": len(mask_spans),
        "total_masked_bins": int(any_masked.sum()),
        "pct_bins_masked": float(100 * any_masked.sum() / N) if N > 0 else 0.0,
        "global_channel_freq": global_channel_freq,
        "n_channels_ever_masked": int(n_channels_ever_masked),
        "channels_ever_masked": list(int(c) for c in channels_ever_masked),
        "channel_consistency_counts": channel_consistency_counts,
    }
    
    if span_stats:
        span_lens = [s["span_len"] for s in span_stats]
        result["avg_span_len"] = float(np.mean(span_lens))
        result["median_span_len"] = float(np.median(span_lens))
        result["min_span_len"] = int(min(span_lens))
        result["max_span_len"] = int(max(span_lens))
    
    if len(channel_freq_nonzero) > 0:
        result["avg_channel_freq"] = float(channel_freq_nonzero.mean())
        result["median_channel_freq"] = float(np.median(channel_freq_nonzero))
        result["min_channel_freq"] = int(channel_freq_nonzero.min())
        result["max_channel_freq"] = int(channel_freq_nonzero.max())
    
    return result


def print_session_summary(result):
    """Pretty-print summary for one session."""
    if result is None:
        return
    
    print(f"\n{'='*70}")
    print(f"Session: {result['session_id']}")
    print(f"{'='*70}")
    print(f"  Total bins: {result['n_bins']:,}")
    print(f"  Total trials: {result['n_trials']}")
    print(f"  Masked trials: {result['n_masked_trials']} / {result['n_trials']} ({result['pct_trials_masked']:.1f}%)")
    print(f"  Total masked bins: {result['total_masked_bins']:,} ({result['pct_bins_masked']:.2f}%)")
    print(f"  Contiguous mask regions: {result['n_contiguous_regions']}")
    
    if result['span_stats']:
        print(f"\n  Span length stats:")
        print(f"    Mean: {result['avg_span_len']:.1f} bins")
        print(f"    Median: {result['median_span_len']:.1f} bins")
        print(f"    Range: [{result['min_span_len']}, {result['max_span_len']}]")
    
    # Channel-level summary
    print(f"\n  Channel statistics:")
    print(f"    Unique channels ever masked: {result['n_channels_ever_masked']} / 96")
    if result.get('avg_channel_freq'):
        print(f"    Avg times a channel is masked: {result['avg_channel_freq']:.1f}")
        print(f"    Median: {result['median_channel_freq']:.1f}, Range: [{result['min_channel_freq']}, {result['max_channel_freq']}]")
    
    # Channel consistency
    const_count = result['channel_consistency_counts']['constant']
    vary_count = result['channel_consistency_counts']['varying']
    print(f"\n  Channel consistency within trials:")
    print(f"    Trials with constant channels: {const_count}")
    print(f"    Trials with varying channels: {vary_count}")
    
    if result['trial_mask_stats']:
        print(f"\n  Masked trials breakdown (first 10):")
        for stat in result['trial_mask_stats'][:10]:
            consistency_marker = "[const]" if stat['channel_consistency'] == "constant" else "[vary]"
            print(
                f"    Trial {stat['trial_idx']:3d}: "
                f"len={stat['trial_len']:4d}, masked_bins={stat['masked_bins']:4d} "
                f"({stat['pct_masked']*100:5.1f}%), chans={stat['n_channels']} "
                f"({stat['avg_channels_per_bin']:.1f} avg/bin) {consistency_marker}"
            )
        if len(result['trial_mask_stats']) > 10:
            print(f"    ... and {len(result['trial_mask_stats']) - 10} more masked trials")


def main():
    parser = argparse.ArgumentParser(description="Analyze masking patterns in test data")
    parser.add_argument("--test-dir", type=str, default="kaggle_data/test",
                       help="Path to test data directory")
    parser.add_argument("--session-id", type=str, default=None,
                       help="Analyze specific session (default: all test sessions)")
    parser.add_argument("--output-csv", type=str, default=None,
                       help="Save session summary to CSV")
    parser.add_argument("--output-trials-csv", type=str, default=None,
                       help="Save per-trial masking stats to CSV")
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    
    # Find all test sessions
    if args.session_id:
        session_ids = [args.session_id]
    else:
        mask_files = sorted(test_dir.glob("*_mask.npy"))
        session_ids = [f.stem.replace("_mask", "") for f in mask_files]
    
    print(f"Analyzing {len(session_ids)} session(s)...")
    
    results = []
    all_trial_stats = []
    
    for session_id in session_ids:
        result = analyze_mask_structure(str(test_dir), session_id)
        if result:
            results.append(result)
            print_session_summary(result)
            all_trial_stats.extend(result['trial_mask_stats'])
    
    # Cross-session summary
    print(f"\n{'='*70}")
    print("CROSS-SESSION SUMMARY")
    print(f"{'='*70}")
    if results:
        masked_bins_list = [r['total_masked_bins'] for r in results]
        masked_trials_list = [r['n_masked_trials'] for r in results]
        
        print(f"  Sessions analyzed: {len(results)}")
        print(f"  Total masked bins across all: {sum(masked_bins_list):,}")
        print(f"  Total masked trials across all: {sum(masked_trials_list)}")
        print(f"  Avg bins masked per session: {np.mean(masked_bins_list):.0f}")
        print(f"  Avg trials masked per session: {np.mean(masked_trials_list):.1f}")
    
    # Collect span length data
    all_spans = []
    for result in results:
        all_spans.extend([s['span_len'] for s in result['span_stats']])
    
    if all_spans:
        print(f"\n  All mask span lengths (across all sessions):")
        print(f"    Count: {len(all_spans)}")
        print(f"    Mean: {np.mean(all_spans):.1f} bins")
        print(f"    Median: {np.median(all_spans):.1f} bins")
        print(f"    Std: {np.std(all_spans):.1f} bins")
        print(f"    Range: [{min(all_spans)}, {max(all_spans)}]")
        
        # Distribution
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(all_spans, p)
            print(f"    P{p:2d}: {val:.1f} bins")
    
    # Save results
    if args.output_csv and results:
        df = pd.DataFrame([
            {
                'session_id': r['session_id'],
                'n_bins': r['n_bins'],
                'n_trials': r['n_trials'],
                'n_masked_trials': r['n_masked_trials'],
                'pct_trials_masked': r['pct_trials_masked'],
                'total_masked_bins': r['total_masked_bins'],
                'pct_bins_masked': r['pct_bins_masked'],
                'n_contiguous_regions': r['n_contiguous_regions'],
                'avg_span_len': r.get('avg_span_len', None),
                'median_span_len': r.get('median_span_len', None),
                'min_span_len': r.get('min_span_len', None),
                'max_span_len': r.get('max_span_len', None),
            }
            for r in results
        ])
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved session summary to: {args.output_csv}")
    
    if args.output_trials_csv and all_trial_stats:
        df = pd.DataFrame(all_trial_stats)
        df.to_csv(args.output_trials_csv, index=False)
        print(f"Saved trial-level stats to: {args.output_trials_csv}")
    
    # Cross-session channel analysis
    print(f"\n  Channel analysis across all sessions:")
    all_channels_masked = set()
    all_consistent = 0
    all_varying = 0
    
    for result in results:
        all_channels_masked.update(result['channels_ever_masked'])
        all_consistent += result['channel_consistency_counts']['constant']
        all_varying += result['channel_consistency_counts']['varying']
    
    print(f"    Total unique channels masked (across all sessions): {len(all_channels_masked)} / 96")
    print(f"    Channels never masked: {96 - len(all_channels_masked)}")
    print(f"    Trials with constant channel masking: {all_consistent}")
    print(f"    Trials with varying channel masking: {all_varying}")
    if all_consistent + all_varying > 0:
        pct_const = 100.0 * all_consistent / (all_consistent + all_varying)
        print(f"    Proportion constant: {pct_const:.1f}%")


if __name__ == "__main__":
    main()
