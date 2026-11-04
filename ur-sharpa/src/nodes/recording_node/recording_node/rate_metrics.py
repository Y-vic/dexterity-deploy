"""Actual source/timeline cadence, without turning repeated cache reads into frames."""
import numpy as np

RATE_TOLERANCE = .02
MAX_GAP_PERIODS = 1.5
MIN_VALID_RATE_HZ = 30. * (1. - RATE_TOLERANCE)


def cadence(timestamps_ns, frequency, duration_s, *, window_start_ns=None):
    stamps = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    stamps = stamps[stamps > 0]
    target = float(frequency)
    minimum = target * (1. - RATE_TOLERANCE)
    if len(stamps):
        advancing = np.r_[True, stamps[1:] > np.maximum.accumulate(stamps[:-1])]
        unique = stamps[advancing]
    else:
        unique = stamps
    intervals = np.diff(unique)
    boundary_gaps = 0
    if window_start_ns is not None:
        # A source sample retained before recording can precede the first
        # timeline tick. Only the intersection of its gap with the capture
        # window is a recording stall. Keep all source values/counts intact.
        clipped = unique[1:]-np.maximum(unique[:-1], int(window_start_ns))
        boundary_gaps = int(np.count_nonzero(clipped != intervals))
        intervals = clipped[clipped > 0]
    gaps_ms = intervals.astype(np.float64) / 1e6
    period_ms = 1000. / target
    source_hz = len(unique) / duration_s if duration_s > 0 else 0.
    large_gaps = int(np.count_nonzero(gaps_ms > period_ms * MAX_GAP_PERIODS))
    backwards = int(np.count_nonzero(np.diff(stamps) < 0))
    return dict(target_hz=target, minimum_hz=minimum, source_hz=round(source_hz, 3),
                unique_frames=len(unique), repeated_frames=int(len(stamps)-len(unique)),
                boundary_gaps_clipped=boundary_gaps,
                backwards=backwards, gap_limit_ms=round(period_ms * MAX_GAP_PERIODS, 3),
                gap_count=large_gaps, max_gap_ms=round(float(gaps_ms.max()), 3) if gaps_ms.size else 0.,
                p95_gap_ms=round(float(np.percentile(gaps_ms, 95)), 3) if gaps_ms.size else 0.,
                stable=bool(source_hz >= minimum and not large_gaps and not backwards))
