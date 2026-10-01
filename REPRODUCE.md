# Reproducing the results of the paper

This file lists every step from the public raw data to the tables and figures
of the paper and its Supplementary Information (SI). All reported results were
produced on Windows 11 with Python 3.12.10 and the library versions pinned in
`requirements.txt`. The commands below use forward slashes; they work unchanged
on Windows, Linux and macOS. Run them from the root of the repository.

```
pip install -r requirements.txt
```

Every experiment writes only CSV and JSON files under `results/revision_v5/`.
The folder names start with an experiment id (`T1.4`, `T2.2`, ...); the table at
the end of this file says what each id contains. Every experiment program has a
unit test `tools/test_<name>.py` that runs in a few seconds on synthetic data.

## 1. Data

| Scenario | Source | Prepared file |
|---|---|---|
| `youtube_hourly` | Kaggle, "YouTube videos viewCount every hour" (CC0), file `count_observation_upload.csv` | `data/datasets/youtube_hourly.csv` |
| `taxi_hourly`, `taxi_30min`, `taxi_5min` | NYC TLC Trip Record Data, Yellow Taxi, January to November 2025 (`yellow_tripdata_2025-01.parquet` ... `-11.parquet`) | `data/datasets/yellow_taxi_2025_all_{hourly,30min,5min}.csv` |
| `movielens_daily`, `movielens_weekly` | GroupLens MovieLens 32M (ML-32M), file `ratings.csv` | `data/datasets/movielens_v5_{daily,weekly}.csv` |

Put the raw files in `data/raw/youtube/`, `data/raw/yellow_taxi/2025/` and
`data/raw/movielens/`, then build the prepared files:

```
python prepare_data.py --dataset youtube --input data/raw/youtube/count_observation_upload.csv --output data/datasets/youtube_hourly.csv --youtube-min-views-per-video 50
python prepare_data.py --dataset yellow_taxi --input "data/raw/yellow_taxi/2025/*.parquet" --output data/datasets/yellow_taxi_2025_all_hourly.csv --yellow_taxi-granularity hourly --yellow_taxi-min-trips-per-location 1 --yellow_taxi-start-date 2025-01-01
python prepare_data.py --dataset yellow_taxi --input "data/raw/yellow_taxi/2025/*.parquet" --output data/datasets/yellow_taxi_2025_all_30min.csv --yellow_taxi-granularity 30min --yellow_taxi-min-trips-per-location 1 --yellow_taxi-start-date 2025-01-01
python prepare_data.py --dataset yellow_taxi --input "data/raw/yellow_taxi/2025/*.parquet" --output data/datasets/yellow_taxi_2025_all_5min.csv --yellow_taxi-granularity 5min --yellow_taxi-min-trips-per-location 1 --yellow_taxi-start-date 2025-01-01
python tools/prepare_movielens_v5.py --raw data/raw/movielens/ratings.csv --start 1998-01-01 --end 2023-10-12 --min-obs 24 --data-out data/datasets --meta-out results/revision_v5/T3.9_movielens/data_prep
```

Every prepared file has the columns `timestamp,item_id,count`. The MD5 sums of
the files used in the paper are:

| File | MD5 (written on Windows) | MD5 with LF line ends (Linux, macOS) |
|---|---|---|
| `youtube_hourly.csv` | `7b1ad171ae827e44be1aeb30cf359570` | `a688190f76d783cdcd39c2c554cceb46` |
| `yellow_taxi_2025_all_hourly.csv` | `2e392096919c670cfdf31501a1a4ee04` | `d1922ee624e863a9f46cf2268a73193c` |
| `yellow_taxi_2025_all_30min.csv` | `7d6c8afc41127880ee58794569e60430` | `0a74be837c7945dc6855405879e38859` |
| `yellow_taxi_2025_all_5min.csv` | `e5259e7e9f99a042ae6132a11a20e373` | `827f1755856c5672b0c86ba62a1f4ee6` |
| `movielens_v5_daily.csv` | `349c336bb43bbe0c841072f918f8b450` | same |
| `movielens_v5_weekly.csv` | `3a3fab484684c3d3f4dc525589efdf7d` | same |

pandas writes Windows line ends on Windows; the content is the same.
The three taxi files keep all 262 pickup zones (the smallest zone has 5 trips,
so any `--yellow_taxi-min-trips-per-location` from 1 to 5 gives the same file).
The start date removes trips dated before 2025 that the January file contains.

The MovieLens licence does not allow redistribution, so only the script is provided. See
`data/README.md` for the conversion rules.

## 2. Settings of the six scenarios

The same values are used by every experiment below.

| Scenario | Prepared file | `--min-obs` | `--block` | `--block-sens` |
|---|---|---|---|---|
| `youtube_hourly` | `youtube_hourly.csv` | 50 | 24 | (none) |
| `taxi_hourly` | `yellow_taxi_2025_all_hourly.csv` | 24 | 168 | 24 |
| `taxi_30min` | `yellow_taxi_2025_all_30min.csv` | 24 | 336 | 48 |
| `taxi_5min` | `yellow_taxi_2025_all_5min.csv` | 24 | 2016 | 288 |
| `movielens_daily` | `movielens_v5_daily.csv` | 24 | 7 | 28 |
| `movielens_weekly` | `movielens_v5_weekly.csv` | 24 | 4 | 13 |

`--min-obs` is the threshold of the causal item list. `--block` is the block
length (in slots) of the block bootstrap and of the block-level Wilcoxon test;
`--block-sens` is a shorter block reported as a sensitivity check.

In the commands below, `<s>` is the scenario name, `<file>` the prepared file,
and `<min>`, `<block>`, `<sens>` the values of this table. Leave out
`--block-sens <sens>` for YouTube. Steps 3 to 13 use the four YouTube and taxi
scenarios; step 14 is MovieLens.

## 3. Main runs, default windows (Tables 2 and 3, Figures 2 and 3, Tables S12-S15, Figure S6)

Baselines use 7 slots and the three wavelet-based methods 64 slots.

```
python tools/run_v5_eval.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/<s>
python tools/stats_report.py --run results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/<s> --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/causal/T1.4_protocol_v5/<s>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/causal
```

Leakage audit of these runs:

```
python tools/leakage_audit.py --causal-universe --run-root results/revision_v5/T1.5_causal_universe --out results/revision_v5/T1.5_leakage_audit/causal
```

## 4. Window sweep and the equal 64-slot window (Table 3, Figure S1)

Every method with N in {7, 16, 32, 64, 128} (wavelet-based methods from 16),
plus SMA, an equivalent EWMA and Holt. The folder `W064` is the equal-window
configuration of the paper.

```
python tools/run_window_sweep.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T2.2_window_sweep/<s>
python tools/run_window_sweep.py --collect results/revision_v5/T2.2_window_sweep
python tools/stats_report.py --run results/revision_v5/T2.2_window_sweep/<s>/W<NNN> --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T2.2_window_sweep/<s>/W<NNN>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/T2.2_window_sweep
```

`W<NNN>` is `W016`, `W032`, `W064` and `W128`.

## 5. Responsiveness to real surges (Section 4.10, Figure 4, Tables S29-S31, Figure S10)

```
python tools/run_responsiveness.py --data data/datasets/<file> --min-obs <min> --causal-universe --scenario <s> --block <block> --control-default results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/<s> --control-equal64 results/revision_v5/T2.2_window_sweep/<s>/W064 --out results/revision_v5/T2.4_responsiveness/<s>
python tools/run_responsiveness.py --collect results/revision_v5/T2.4_responsiveness
```

## 6. Decomposition level J (Table S5)

```
python tools/run_level_sweep.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.1_level_sweep/<s>
python tools/run_level_sweep.py --collect results/revision_v5/T3.1_level_sweep
python tools/stats_report.py --run results/revision_v5/T3.1_level_sweep/<s>/<W>/<m> --reference J3 --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T3.1_level_sweep/<s>/<W>/<m>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/T3.1_level_sweep
```

`<W>` is `W064` and `W032`; `<m>` is `WSPI` and `DTCWT+AF`.

## 7. alpha x beta grid and the 30/70 selection (Section 4.9, Tables S27-S28, Figure S9)

The selection of J uses the level sweep of step 6.

```
python tools/run_param_grid.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.2_param_grid/<s>
python tools/run_param_grid.py --collect results/revision_v5/T3.2_param_grid
python tools/run_param_grid.py --select results/revision_v5/T3.2_param_grid --level-root results/revision_v5/T3.1_level_sweep
python tools/stats_report.py --run results/revision_v5/T3.2_param_grid/selection/test_split/<s>/<p> --reference selected --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T3.2_param_grid/<s>/<p>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/T3.2_param_grid
```

`<p>` is `alpha_beta` and `J`.

## 8. Ablation and fusion function (Section 4.8, Table 4, Tables S25-S26)

```
python tools/run_ablation_v5.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.3_ablation/<s>
python tools/run_ablation_v5.py --collect results/revision_v5/T3.3_ablation
python tools/stats_report.py --run results/revision_v5/T3.3_ablation/<s>/<f> --reference WSPI --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T3.3_ablation/<s>/<f>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/T3.3_ablation
```

`<f>` is `ablation` and `fusion`.

## 9. Padding and boundary extension (Section 3.4, Table S2)

```
python tools/run_padding_v5.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.4_padding/<s>
python tools/run_padding_v5.py --collect results/revision_v5/T3.4_padding
python tools/stats_report.py --run results/revision_v5/T3.4_padding/<s>/ext/<m> --reference symmetric --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T3.4_padding/<s>/<m>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/T3.4_padding
```

`<m>` is `WSPI`, `DTCWT+AF` and `DWT+AF`.

## 10. Wider perturbations (Section 4.3, Tables S16-S18, Figure S5)

```
python tools/run_robustness_v5.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.5_robustness/<s>
python tools/run_robustness_v5.py --collect results/revision_v5/T3.5_robustness
python tools/run_robustness_v5.py --stats results/revision_v5/T3.5_robustness --stats-out results/revision_v5/T1.6_stats/T3.5_robustness
```

The 5-minute taxi run is long; it can be split with `--methods` (for example
`--methods WSPI`, then `--methods DTCWT+AF`, then the other seven) into the same
`--out` folder. The audit of the robustness test of the main tables:

```
python tools/run_robustness_v5.py --audit <s> --data data/datasets/<file> --min-obs <min> --causal-universe --audit-out results/revision_v5/T3.5_robustness
```

## 11. Shift invariance (Section 4.8, Figure S8)

```
python tools/run_shift_test.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.6_shift_invariance/<s>
python tools/run_shift_test.py --synthetic --out results/revision_v5/T3.6_shift_invariance/synthetic
python tools/run_shift_test.py --collect results/revision_v5/T3.6_shift_invariance
python tools/run_shift_test.py --stats results/revision_v5/T3.6_shift_invariance --stats-out results/revision_v5/T1.6_stats/T3.6_shift_invariance
```

## 12. Relation of R and W_E (Section 3.5, Table S3, Figure S2)

```
python tools/run_feature_relation.py --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.7_feature_relation/<s>
python tools/run_feature_relation.py --collect results/revision_v5/T3.7_feature_relation
```

## 13. Run time and memory (Section 4.7, Tables S20-S24, Figure S7)

The measurements use one thread (the program sets it).

```
python tools/run_runtime_v5.py --hardware
python tools/run_runtime_v5.py --bench --out results/revision_v5/T3.8_runtime/bench
python tools/run_runtime_v5.py --memory --out results/revision_v5/T3.8_runtime/memory
python tools/run_runtime_v5.py --real --data data/datasets/<file> --min-obs <min> --causal-universe --out results/revision_v5/T3.8_runtime/real/<s>
python tools/run_runtime_v5.py --collect results/revision_v5/T3.8_runtime
```

## 14. MovieLens (Tables 2 and 3, Table S9, Figures S3 and S4)

```
python tools/run_v5_eval.py --data data/datasets/movielens_v5_<g>.csv --min-obs 24 --causal-universe --out results/revision_v5/T3.9_movielens/default/movielens_<g>
python tools/run_window_sweep.py --data data/datasets/movielens_v5_<g>.csv --min-obs 24 --causal-universe --windows 64 --out results/revision_v5/T3.9_movielens/equal64/movielens_<g>
python tools/stats_report.py --run results/revision_v5/T3.9_movielens/default/movielens_<g> --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T3.9_movielens/default/movielens_<g>
python tools/stats_report.py --run results/revision_v5/T3.9_movielens/equal64/movielens_<g>/W064 --block <block> --block-sens <sens> --out results/revision_v5/T1.6_stats/T3.9_movielens/equal64/movielens_<g>
python tools/stats_report.py --collect results/revision_v5/T1.6_stats/T3.9_movielens
```

`<g>` is `daily` (block 7, sensitivity 28) or `weekly` (block 4, sensitivity 13).

## 15. YouTube data provenance (Section 4.2, Table S8)

```
python tools/audit_youtube_provenance.py --raw data/raw/youtube/count_observation_upload.csv --processed data/datasets/youtube_hourly.csv --v5-run results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/youtube_hourly --out results/revision_v5/T3.11_youtube_provenance
python tools/audit_youtube_provenance.py --summary --out results/revision_v5/T3.11_youtube_provenance
```

## 16. Tables and figures

These programs read `results/revision_v5` through `scripts/result_paths.json`
(logical name -> result folder); `<folder>` is any output folder outside
`results`.

```
python scripts/generate_tables.py --part all --results results/revision_v5 --out <folder>
python scripts/generate_figures.py --target paper --results results/revision_v5 --out <folder>
python scripts/generate_figures.py --target si --results results/revision_v5 --out <folder>
python scripts/generate_figures.py --target thesis --results results/revision_v5 --out <folder>
python scripts/generate_figures.py --target draft --results results/revision_v5 --out <folder>
```

`generate_tables.py --part main` (or `all`) also writes the source CSVs of the
main tables and figures (`values_all.csv`, `tests_all.csv`, `control.csv`) to
`results/revision_v5/T3.10_main_figures/` (the folder `figure_data` of
`result_paths.json`); `--data-out` sets another folder.

The output file names are the names used in the LaTeX source; the table and
figure numbers follow from the order in the LaTeX source.

| Paper element | Program | Main source files (`results/revision_v5/...`) |
|---|---|---|
| Main result tables (`tab_main_default`, `tab_main_equal64`); SI tables of intervals and tests (`si_ci_*`, `si_tests_*`) | `generate_tables.py` | `T3.10_main_figures/values_all.csv`, `tests_all.csv` |
| Other SI tables (`si_*.tex`) | `generate_tables.py --part si` | listed in its output `si_tables_sources.csv` |
| Paper figures (`fig_*.pdf`) | `generate_figures.py --target paper` | `T3.10_main_figures/`, `T2.4_responsiveness/` |
| SI figures (`si_*.pdf`) | `generate_figures.py --target si` | read by the figure functions of the program |
| Run-time table | values of | `T3.8_runtime/runtime_paper_table.csv` |
| Ablation table | values of | `T3.3_ablation/ablation_summary.csv`, `T1.6_stats/T3.3_ablation/` |
| Sensitivity table | values of | `T3.2_param_grid/grid_summary.csv` (test split) |
| Responsiveness table | values of | `T2.4_responsiveness/responsiveness_summary.csv` |

The pipeline figure is drawn in TikZ in the LaTeX source of the paper.

## 17. Experiment ids

| Id | Folder under `results/revision_v5/` | Content |
|---|---|---|
| T1.4, T1.5 | `T1.5_causal_universe/T1.4_protocol_v5/` | Main runs of the paper (protocol V5, causal item list) |
| T1.5 | `T1.5_leakage_audit/` | Leakage audit |
| T1.6 | `T1.6_stats/` | Confidence intervals and paired tests of every experiment |
| T2.2 | `T2.2_window_sweep/` | Window sweep; `W064` = equal-window configuration |
| T2.4 | `T2.4_responsiveness/` | Responsiveness to real surges |
| T3.1 | `T3.1_level_sweep/` | Decomposition level J |
| T3.2 | `T3.2_param_grid/` | alpha x beta grid and 30/70 selection |
| T3.3 | `T3.3_ablation/` | Ablation and fusion functions |
| T3.4 | `T3.4_padding/` | Padding and boundary extension |
| T3.5 | `T3.5_robustness/` | Wider perturbations |
| T3.6 | `T3.6_shift_invariance/` | Shift invariance |
| T3.7 | `T3.7_feature_relation/` | Relation of R and W_E |
| T3.8 | `T3.8_runtime/` | Run time and memory |
| T3.9 | `T3.9_movielens/` | MovieLens runs and data profile |
| T3.10 | `T3.10_main_figures/` | Source values of the main tables and figures |
| T3.11 | `T3.11_youtube_provenance/` | YouTube data provenance |

`evaluation/fast_evaluator.py` in `compat` mode reproduces the protocol of the
first submission of the paper; the protocol of the revised paper is
`evaluation/protocol_v5.py`.
