# Upstream provenance

This directory vendors
[`JonathanGabor/evilgenie_inspect`](https://github.com/JonathanGabor/evilgenie_inspect)
at commit `f5d8a2ca5d92ecf5fd1e857695c54e609312e1bd`.

The upstream benchmark is MIT licensed; its license is retained in
[`LICENSE`](LICENSE). This copy was ported from an internal fork, which added the bubblewrap sandbox and the tested Tinker tool-call
bridge. In this repository that bridge is merged into `sdf.models.provider` so
the same implementation serves both base models and SDF adapters. `task.py`,
`eval.json`, and the scalar/secondary-metric adapter are the SDF-specific glue.
