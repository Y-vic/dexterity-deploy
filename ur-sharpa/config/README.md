# Model server profiles

`model_servers.json` is the only workstation-side mapping between a dashboard
server choice and a remote checkout. The dashboard discovers models from
`<repo_root>/scripts/deploy/*/launch.sh`; there is no model alias registry in
ROS.

Every task config uses schema `sharpa.deploy.task_config.v2` and contains one
flat `runs` list. Each run must carry the exact launcher directory name in
`model` and a globally unique, immutable `run_id`. A selected run starts as:

```bash
bash scripts/deploy/<model>/launch.sh <task_name> <run_id>
```

Add or replace task runs on the model server with
`scripts/deploy/task_config/register_run.py`. Duplicate `run_id` values are
rejected unless `--replace` is passed intentionally.

For a new server, add one profile with its SSH alias, code checkout, asset
root, task-config root, and optional environment file. The dashboard can switch
profiles only while the current model is stopped or faulted. Its `/model/status`
message then changes both lifecycle SSH commands and the policy client's SSH
tunnel target.

On PS `/home/ps/dexterity` is the Dexhand checkout and asset root, matching the
single-root layout used on BAAI. Checkpoints, datasets, logs, and third-party
repositories live under that checkout; machine-specific variables are in
`.env.local`.
