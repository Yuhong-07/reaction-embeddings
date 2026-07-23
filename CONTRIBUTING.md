# Contributing

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Change requirements

1. Keep raw data immutable and write transformations to `data/interim` or `data/processed`.
2. Update schemas, serialization, coverage reports, and tests together when fields change.
3. Do not use EC numbers as main embedding inputs.
4. Preserve the independent EC evaluation pool when comparing auxiliary-loss weights.
5. Rebuild versioned caches when graph features or preprocessing semantics change.
6. Run the relevant validation scripts after tests pass.
7. Append the reason, files, compatibility impact, validation, outputs, and rollback plan to `agent.md`.

## Pull requests

A pull request should contain one coherent change, a short validation summary, and no generated model/data binaries. If an experiment creates deliverables, publish their manifests and checksums and store the large files outside ordinary Git history.
