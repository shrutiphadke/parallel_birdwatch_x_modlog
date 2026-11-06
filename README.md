# Reproducibility
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Run from the repository root with Python. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Linux or macOS:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run the current dataset with:

```powershell
typer src.pipeline run reproduce --dataset reddit --input data/processed/final_intersection_dataset.csv
```

K-fold evaluation is not enabled by default. To run it as well:

```powershell
typer src.pipeline run reproduce --dataset reddit --input data/processed/final_intersection_dataset.csv --k-fold
```

Every method is enabled by default. Use the method's name once with
`--exclude-method` (`-x`) to opt out. For example, this keeps Team Formation
enabled but skips Bandit:

```sh
typer src.pipeline run reproduce --dataset reddit --input data/processed/final_intersection_dataset.csv --exclude-method bandit
```

Repeat the option to omit more methods, for example `-x bandit -x sef`.
Individual baselines can be skipped the same way (`-x BL4_reddit_score`). The
same option is available on `run methods`. In an advanced JSON config, use the
optional field:

```json
{
  "run": {
    "methods": ["all"],
    "exclude_methods": ["bandit", "sef"]
  }
}
```

## Inputs

You supply at most four inputs. Everything else (splits, causal features,
embeddings, NormVio scores, results, figures) is derived and cached by the
pipeline. Flat tables may be CSV, Parquet, a JSON array, JSONL, or NDJSON. IDs
must match across all inputs.

| Input | Columns | Needed by |
|---|---|---|
| **votes** (required) | `username`, `community`, `item_id`, `timestamp`, `vote`, `label` | every method |
| **items** | `item_id`, `title` + `selftext` and/or `text`, `score`, optional `timestamp` | SEF, NVSE, Team Formation (texts); BL4/BL5 (`score`) |
| **user_contributions/** | one `<username>.jsonl` per user | NVSE, SEF, Team Formation (causal features, SEF user histories) |
| **user_metadata** (optional) | `username`, `account_created_utc` | causal features (account tenure) |

What each method needs, at a glance:

| Method | votes | items | user_contributions | models |
|---|:-:|:-:|:-:|---|
| Baselines BL1–BL3, Community Notes, VAR, MA-QSMF | ✓ | | | |
| Baselines BL4–BL5 | ✓ | `score` | | |
| NVSE | ✓ | text | ✓ | NormVio checkpoints + BERT |
| SEF | ✓ | text | ✓ | sentence-transformer |
| Team Formation (5 methods) | ✓ | text | ✓ | sentence-transformer |

### One folder for everything

The simplest way to provide inputs is one folder with conventional names:

```text
my_inputs/
  votes.csv                 # or .parquet/.json/.jsonl
  items.parquet             # texts + scores in one table
  user_contributions/       # <username>.jsonl files
  user_metadata.csv         # optional
```

```sh
typer src.pipeline run reproduce --dataset my_dataset --inputs path/to/my_inputs
```

Any explicit option (`--input`, `--items`, `--user-contributions`,
`--user-metadata`) overrides the folder. Without `--inputs`, the conventional
locations for the dataset are used:

```text
data/raw/<dataset>/votes.*                       # (reddit: data/processed/final_intersection_dataset.csv)
data/interim/<dataset>/auxiliary/items.parquet
data/raw/<dataset>/user_contributions/           # (reddit: data/raw/user_contributions/user_contributions/)
data/processed/<dataset>/user_metadata.csv       # (reddit: data/processed/user_metadata.csv)
```

All paths are defined in one place, `src/layout.py`; every method's standalone
CLI defaults come from it too.

### Votes

Each row is one vote with string `username`, `community`, and `item_id`,
`float64` Unix-seconds `timestamp`, `vote` in `{-1,+1}`, and an item-constant
`label` in `{-1,+1}` for the moderator decision.

### Items

One row per voted item. Provide `title` and `selftext`, or a single `text`
column, or all three: whichever is missing is derived (`text = title\n\nselftext`;
a `text`-only table feeds NormVio as the post body). `score` is the Reddit score
used by BL4/BL5. Without a `score` column, skip those two baselines with
`-x BL4_reddit_score -x BL5_subreddit`.

Older two-file folders keep working: `post_texts.parquet` and
`moderated_posts_scores.parquet` are still recognized (and override
`items.parquet` individually).

To download items for a votes table from Arctic Shift (one request per post
for texts and score together):

```sh
typer src.pipeline run prepare auxiliary --section items --input-csv path/to/votes.csv --dataset my_dataset
```

### User contributions

Required when causal features must be built for NVSE, SEF, or Team Formation.
They are also the default source for SEF's semantic user histories: the
pipeline converts them locally to `user_documents.parquet`, without making
Arctic Shift requests. Download the JSONL archive
[here](https://drive.google.com/file/d/1RNWxXLgP8cpWInrWxeoUFdHQ0D3GKgzR/view?usp=drive_link)
and extract it to the conventional location above, or pass
`--user-contributions DIR`.

Each line is a Reddit JSON object with `subreddit`, numeric `created_utc`, and
`id` or canonical `name`. A post has `title`; a comment has `body` and canonical
`parent_id`. Numeric `score` is optional:

```json
{"id":"abc123","subreddit":"example","created_utc":1700000000,"title":"Post title","score":4}
{"name":"t1_def456","subreddit":"example","created_utc":1700000100,"body":"Reply","parent_id":"t1_xyz789","score":2}
```

The conversion selects up to 150 documents per user, split between recent posts
and comments in the vote-table time window. Change the cap with
`--n-user-documents N` in `reproduce`. To top up users below the cap from
Arctic Shift, add `--download-user-documents`; otherwise user histories stay
fully local.

### User account metadata

Optional: without it, causal features are still generated but account tenure
is unavailable.

## Models

To generate NVSE scores, download the nine NormVio checkpoints
[here](https://drive.google.com/file/d/1khNw_M4HcJOQe2gftfXMybraYChVt3T7/view?ts=6a05d126)
and place them at:

```text
external/normvio/normvio_redditmodels/<category>/finetuned_model.pt
```

The categories are `spam`, `meta-rules`, `content`, `harassment`, `hatespeech`,
`format`, `off-topic`, `trolling`, and `incivility`.

Hugging Face automatically downloads NormVio's
`DeepPavlov/bert-base-cased-conversational` and SEF's
`sentence-transformers/all-MiniLM-L6-v2` on first use.

## Pipeline stages

`reproduce` runs these stages as a dependency graph. With `--n-workers > 1`,
independent preparation stages run concurrently, and every method waits for
all of them:

| Stage | Waits for |
|---|---|
| `prepare-dataset` (splits, incl. K-fold) | — |
| `prepare-user-documents-local` | — |
| `prepare-causal-user-features` | — |
| `prepare-kfold` (only when splits already exist) | `prepare-dataset` |
| `prepare-nvse-scores` (GPU) | `prepare-dataset` |
| `prepare-sef-cache` (GPU) | `prepare-dataset`, `prepare-kfold`, `prepare-user-documents-local` |
| `methods` | every preparation stage |
| `graphs`, `kfold-methods` | `methods` |
| `kfold-graphs` | `kfold-methods` |

`prepare-sef-cache` encodes post texts and causal user profiles once, for every
split (main and K-fold). SEF and all Team Formation methods read it, so SEF then
runs as one CPU process per split and Team Formation can never start before the
embeddings it needs exist. Each stage is skipped when its output is current.

## Parallel method execution

`run methods` expands the benchmark into independent method/split jobs. All 18
reported methods participate: the five baselines are separate jobs, MA/QSMF is
split into its fixed- and learned-rho branches (preserving their shared base
fits), SEF runs one process per split, and the five Team Formation methods are
scheduled independently. Defaults remain serial for reproducibility.

For a large two-GPU server, a suitable starting point is:

```sh
python -m src.pipeline run methods \
  --n-workers 16 \
  --threads-per-worker 4 \
  --memory-workers 2 \
  --device cuda \
  --gpu-device 0 \
  --gpu-device 1
```

GPU jobs are limited to one per listed device. Team Formation jobs have a
separate memory semaphore controlled by `--memory-workers` (default 1, which
runs the 65 Team Formation jobs one at a time). Each run manifest
(`results/<dataset>/reproducibility/latest_methods_run.json`) records every
job's `peak_rss_mb`; use it to raise `--memory-workers` as far as the machine's
RAM allows. For reference, on the Intersection split Team Formation jobs peak
at roughly 0.4–0.8 GB (Role-NSGA2 is the largest); jobs on the Full split need
several times more. Native BLAS/OpenMP thread counts are capped per subprocess to
prevent nested oversubscription.

The same scheduling options are accepted by `reproduce`, where they also
control how many preparation stages run at once.

## CLI options

### `reproduce`

| Option | Meaning |
|---|---|
| `--inputs DIR` | Folder of conventionally named inputs (see **Inputs**). |
| `--input PATH` | Votes table; overrides `--inputs`. Mutually exclusive with `--config`. |
| `--items PATH` | Per-item texts + scores table; overrides `--inputs`. |
| `--dataset NAME` | Namespace separating splits, caches, results, and reports. Default: `reddit`. |
| `--method NAME`, `-m NAME` | Select a method; repeat for several. Omit to run all. Choices: `baselines`, `community-notes`, `var`, `nvse`, `sef`, `ma-qsmf`, `team-formation`. |
| `--exclude-method NAME`, `-x NAME` | Opt out by method name, for example `bandit`, `sef`, or `BL4_reddit_score`; repeat for several. |
| `--user-contributions DIR` | Local directory of per-user JSONL files; also used to derive `user_documents.parquet`. |
| `--user-metadata PATH` | Local account-metadata table. |
| `--user-documents PATH` | Local historical user-document table (normally derived). |
| `--post-texts PATH`, `--external-scores PATH` | Legacy overrides for one half of `--items`. |
| `--download-user-documents` | Top up users below the configured document cap through Arctic Shift. Default: disabled. |
| `--n-user-documents N` | Maximum documents per user for semantic embeddings. Default: 150. |
| `--k-fold` | Also run five-fold evaluation and summaries. |
| `--force` | Rebuild splits, causal features, NVSE scores, and the semantic cache. |
| `--graphs` / `--no-graphs` | Enable or skip final figures. Default: enabled. |
| `--device cpu|cuda|cuda:N` | Execution device. Default: `cpu`. |
| `--gpu-device INDEX` | GPU scheduling lane for the method stage; repeat once per GPU. |
| `--n-workers N` | Maximum concurrent jobs/stages. Default: 1. |
| `--threads-per-worker N` | BLAS/OpenMP threads assigned to each subprocess. Default: 1. |
| `--memory-workers N` | Maximum concurrent Team Formation jobs. Default: 1. |
| `--dry-run` | Validate and print the plan without executing it. |
| `--config PATH`, `-c PATH` | Optional advanced JSON configuration (`inputs.directory`, `inputs.items`, and every path above). |

### `run methods`

All path options default to the `--dataset` layout.

| Option | Meaning |
|---|---|
| `--dataset NAME` | Dataset whose conventional paths fill unset options. Default: `reddit`. |
| `--items PATH` | Per-item texts + scores table. |
| `--n-workers N` | Maximum concurrent jobs. `0` selects a conservative affinity-aware value; default: 1. |
| `--threads-per-worker N` | BLAS/OpenMP threads assigned to each subprocess; default: 1. |
| `--memory-workers N` | Maximum concurrent Team Formation jobs; default: 1. |
| `--device cpu|cuda|cuda:N` | Device used by MA-QSMF. |
| `--gpu-device INDEX` | GPU scheduling lane; repeat once per GPU. |
| `--prepare-cache` / `--no-prepare-cache` | Build/verify the SEF/Team Formation embedding cache first (a no-op when current). Default: on. |
| `--k-fold` | Run the materialized K-fold split collection and render its graphs after all jobs finish. |
| `--continue-on-error` | Finish independent jobs and report every failure. |
| `--dry-run` | Print the complete method/split execution matrix without running it. |

### `prepare` subcommands

Every `prepare` subcommand takes `--dataset NAME`, which fills all unset paths
from that dataset's layout and writes its run manifest under
`data/interim/<dataset>/`.

| Command | Purpose |
|---|---|
| `prepare dataset` | Clean the votes and write every split (including K-fold). |
| `prepare kfold` | Rebuild only the K-fold collections. |
| `prepare user-features` | Build causal per-vote user-history features. |
| `prepare user-documents` | Convert contributions to `user_documents.parquet` (`--download` tops up from Arctic Shift). |
| `prepare nvse-cache` | Score every post with the NormVio models once. |
| `prepare sef-cache` | Encode post texts and user profiles for SEF/Team Formation (`--posts-only` for Team Formation alone). |
| `prepare auxiliary` | Download `items.parquet` (`--section items`) and/or user documents from Arctic Shift. |
