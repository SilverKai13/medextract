# Documentation index

Start with [../README.md](../README.md) for what the project is and the
measured results. This directory holds everything else.

## Orientation

| Document | Read it when you want |
|---|---|
| [PROJECT.md](PROJECT.md) | What this is, why it exists, what is in and out of scope |
| [HANDOVER.md](HANDOVER.md) | To pick the project up cold and be productive in an hour |
| [RESULTS.md](RESULTS.md) | **Every measurement, in full**, with the command that produced each |
| [PROGRESS.md](PROGRESS.md) | The dated narrative of the runs that produced them |

## Product and requirements

| Document | Covers |
|---|---|
| [PRD.md](PRD.md) | Users, jobs to be done, success metrics, explicit non-goals |
| [REQUIREMENTS.md](REQUIREMENTS.md) | Numbered FR/NFR, each traced to the test that proves it |
| [SPEC.md](SPEC.md) | Endpoint-by-endpoint contract: payloads, status codes, edge cases |

## Design and implementation

| Document | Covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, data flow, deployment topology |
| [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) | The algorithms that are easy to get subtly wrong, and how each is pinned |
| [DATA_MODEL.md](DATA_MODEL.md) | Every structure that crosses a boundary |
| [DATABASE.md](DATABASE.md) | Qdrant collection design, the tracking store, on-disk artefacts |
| [DECISIONS.md](DECISIONS.md) | Decision record — the reasoning, and what each choice costs |

## Operations

| Document | Covers |
|---|---|
| [ENVIRONMENT.md](ENVIRONMENT.md) | Setup, every environment variable, the reference machine, troubleshooting |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, compose, CI/CD, image budget, running it |
| [SECURITY.md](SECURITY.md) | Threat model, clinical safety caveats, known gaps |
| [TESTING.md](TESTING.md) | Strategy, what is covered, what deliberately is not |
| [CODE_STYLE.md](CODE_STYLE.md) | Conventions, especially the comment philosophy |

## Status and history

| Document | Covers |
|---|---|
| [../CHANGELOG.md](../CHANGELOG.md) | Released changes |
| [TASKS.md](TASKS.md) | Execution breakdown of the work done, with status |
| [TODO.md](TODO.md) | Remaining work, prioritised |
| [BUGS.md](BUGS.md) | Every defect found and fixed, with root cause |

## Generated

Written by scripts, never edited by hand. Regenerating them is the only
supported way to change them.

| File | Produced by |
|---|---|
| [sweep_table.md](sweep_table.md) | `.\make.ps1 report` |
| [benchmark.md](benchmark.md) | `.\make.ps1 benchmark` |
| `benchmark.json` | `.\make.ps1 benchmark` — raw results with hardware |

The four `<!-- MEDEXTRACT:*:START -->` blocks in the top-level README are
likewise rewritten by `readme_sync.sync_readme`.
