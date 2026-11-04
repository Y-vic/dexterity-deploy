#!/usr/bin/env python3
"""Read receipt timing and completeness without ROS or opening hardware."""

import argparse
import json
from pathlib import Path
import sqlite3


def inspect(episode):
    directory = Path(episode).resolve() / "raw"
    database = directory / "messages.sqlite3"
    result = {"episode": str(Path(episode).resolve()), "topics": {}}
    metadata = directory / "messages_meta.json"
    result["journal"] = json.loads(metadata.read_text()) if metadata.exists() else None
    # Read-only mode also prevents accidentally creating an empty database.
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        topics = [r[0] for r in connection.execute("SELECT DISTINCT topic FROM messages")]
        for topic in topics:
            row = connection.execute("""
                WITH samples AS (
                    SELECT *,
                        received_monotonic_ns - LAG(received_monotonic_ns) OVER (ORDER BY id) AS receive_gap,
                        source_timestamp_ns - LAG(source_timestamp_ns) OVER (ORDER BY id) AS source_gap,
                        receive_sequence - LAG(receive_sequence) OVER (ORDER BY id) AS sequence_gap
                    FROM messages WHERE topic = ?
                )
                SELECT COUNT(*), MAX(received_monotonic_ns)-MIN(received_monotonic_ns),
                    MAX(receive_gap), SUM(CASE WHEN receive_gap > 50000000 THEN 1 ELSE 0 END),
                    COUNT(DISTINCT CASE WHEN source_timestamp_ns > 0 THEN source_timestamp_ns END),
                    SUM(CASE WHEN source_gap = 0 AND source_timestamp_ns > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN source_gap < 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN sequence_gap > 1 THEN sequence_gap-1 ELSE 0 END)
                FROM samples
            """, (topic,)).fetchone()
            count, span, gap, slow, unique, repeat, backwards, missing = row
            result["topics"][topic] = {
                "written": count,
                "receipt_rate_hz": (count - 1) * 1e9 / span if span else None,
                "max_receipt_gap_ms": gap / 1e6 if gap is not None else None,
                "receipt_gaps_over_50ms": slow,
                "unique_positive_source_timestamps": unique,
                "repeated_positive_source_timestamps": repeat,
                "source_timestamp_regressions": backwards,
                "internal_receive_sequence_gaps": missing,
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="saved episode directory containing raw/")
    args = parser.parse_args()
    print(json.dumps(inspect(args.episode), ensure_ascii=False, indent=2))
