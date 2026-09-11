"""One-off: repair indentation mangled by shell heredoc transport.

Every indent in the source files is a multiple of 4; the transport dropped 3
spaces from some continuation lines (8 -> 5, 12 -> 9). Snap each leading indent
up to the nearest multiple of 4.
"""

import sys

for path in sys.argv[1:]:
    lines = []
    for line in open(path).read().split("\n"):
        stripped = line.lstrip(" ")
        if not stripped:
            lines.append("")
            continue
        n = len(line) - len(stripped)
        if n % 4:
            n = ((n + 3) // 4) * 4
        lines.append(" " * n + stripped)
    open(path, "w").write("\n".join(lines))
    print(f"reindented {path}")
