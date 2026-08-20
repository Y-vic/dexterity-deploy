# deps/ — Reproducible external SDK downloads

Both embodiments depend on external SDKs. Today each machine's README hand-lists
some of them, versions drift, and re-installing on a fresh machine is guesswork.

This directory replaces that with three files:

- `manifest.yaml` — the ONE list of every SDK we actually use, versions pinned.
- `fetch.sh`     — downloads exactly what the manifest says, verifies sha256,
                   caches under `cache/<name>-<version>/`. Never installs
                   system-wide, never overwrites a previous version.
- `lock/`        — per-SDK provenance records: source URL, upstream commit or
                   release tag, sha256, the human who pinned it, the date.

## Design rules

1. **Pinning is by content hash, not by tag.** Upstream can retag; sha256 can't
   lie. Every entry in the manifest has `sha256`.
2. **Versions coexist.** Every install lives under
   `deps/cache/<name>-<version>/`; the current-in-use symlink is per-embodiment
   and set explicitly, never as a side effect of a fetch.
3. **No system package manager writes here.** apt/dnf/brew stays in
   `docs/system_deps.md` as documented prerequisites, not as automation.
4. **No pip install into system Python.** Wheels are downloaded and placed
   next to the SDK; each embodiment's venv points at the wheel path.
5. **The manifest is authoritative even for absent SDKs.** If PND does not
   need `ur_rtde`, the manifest still lists `ur_rtde` with `embodiments: [ur]`
   so anyone reading the manifest sees the full picture.

## Usage

```bash
# On the UR host — only fetch what UR needs
./deps/fetch.sh --embodiment ur

# On the PND robot
./deps/fetch.sh --embodiment pnd-robot

# On the PND workstation
./deps/fetch.sh --embodiment pnd-workstation

# Verify caches match the manifest without fetching
./deps/fetch.sh --verify
```

## Adding a new SDK

1. Add an entry to `manifest.yaml` with source URL, version, sha256, embodiment
   list, unpack instructions.
2. Run `./deps/fetch.sh --embodiment <yours>` locally, confirm the resulting
   `cache/<name>-<version>/` layout is what you expected.
3. Commit `manifest.yaml`, `lock/<name>.lock`. Do NOT commit `cache/`.
4. Update the embodiment package's `README.md` if the SDK requires environment
   variables or dlopen paths beyond what the fetcher already does.

## Directory layout after fetching

```
deps/
├── manifest.yaml
├── fetch.sh
├── cache/                          # gitignored
│   ├── zed_sdk-4.1.2/
│   ├── ur_rtde-1.5.7/
│   ├── pnd_lowstate_dds-0.9.3/
│   ├── webrtc_native-m119/
│   └── sharpa_sdk-2.3.0/
└── lock/
    ├── zed_sdk.lock
    ├── ur_rtde.lock
    ├── pnd_lowstate_dds.lock
    ├── webrtc_native.lock
    └── sharpa_sdk.lock
```
