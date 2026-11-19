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

# Popularity prediction (release 4.1)

Sections 18 to 25 reproduce the forecasting study: the held-out data, the runs,
the paired tests, the decision rules, and the tables, figures and decision files. They use the six scenarios of section 2
as development data and four held-out scenarios. The results go to
`results/prediction/` (CSV and JSON only). The commands are for the root of the
repository, with the library versions of `requirements.txt` (this includes
`statsmodels`, `scikit-learn` and `joblib`).

## 18. Held-out data

| Scenario | Source | Prepared file |
|---|---|---|
| `taxi2024_hourly`, `taxi2024_30min`, `taxi2024_5min` | NYC TLC Trip Record Data, Yellow Taxi, January to December 2024 (`yellow_tripdata_2024-01.parquet` ... `-12.parquet`) | `data/datasets/yellow_taxi_2024_all_{hourly,30min,5min}.csv` |
| `wikipedia_daily` | Extended Wikipedia Web Traffic Daily Dataset (with missing values), Monash forecasting repository on Zenodo, DOI `10.5281/zenodo.7370977`, file `web_traffic_extended_dataset_with_missing_values.tsf` (unzip the downloaded archive) | `data/datasets/wikipedia_daily.csv` |

Put the raw files in `data/raw/yellow_taxi/2024/` and `data/raw/wikipedia/`, then:

```
python prepare_data.py --dataset yellow_taxi --input "data/raw/yellow_taxi/2024/*.parquet" --output data/datasets/yellow_taxi_2024_all_hourly.csv --yellow_taxi-granularity hourly --yellow_taxi-min-trips-per-location 1 --yellow_taxi-start-date 2024-01-01 --yellow_taxi-end-date "2024-12-31 23:59:59"
python prepare_data.py --dataset yellow_taxi --input "data/raw/yellow_taxi/2024/*.parquet" --output data/datasets/yellow_taxi_2024_all_30min.csv --yellow_taxi-granularity 30min --yellow_taxi-min-trips-per-location 1 --yellow_taxi-start-date 2024-01-01 --yellow_taxi-end-date "2024-12-31 23:59:59"
python prepare_data.py --dataset yellow_taxi --input "data/raw/yellow_taxi/2024/*.parquet" --output data/datasets/yellow_taxi_2024_all_5min.csv --yellow_taxi-granularity 5min --yellow_taxi-min-trips-per-location 1 --yellow_taxi-start-date 2024-01-01 --yellow_taxi-end-date "2024-12-31 23:59:59"
python prepare_data.py --dataset wikipedia --input data/raw/wikipedia/web_traffic_extended_dataset_with_missing_values.tsf --output data/datasets/wikipedia_daily.csv
```

The Wikipedia converter keeps the rows `all-access_all-agents` of the 803 days
from 1 July 2015 to 10 September 2017 and, in file order, every fourth of these
rows (39,402 rows, 9,851 items). The step, 4, is the smallest k for which the kept
rows times 803 days do not exceed 9,256,350, the number of result rows of the
largest taxi scenario at horizon 1; the rule is in `data/README.md`. The taxi
files keep all 263 zones. The MD5 sums of the files used in the study are:

| File | Rows | Items | MD5 (written on Windows) |
|---|---|---|---|
| `yellow_taxi_2024_all_hourly.csv` | 1,062,021 | 263 | `30f2ddce3b4f723f6640d7d1597246a6` |
| `yellow_taxi_2024_all_30min.csv` | 1,715,758 | 263 | `35d1a4f0e4f90305966e92d388317112` |
| `yellow_taxi_2024_all_5min.csv` | 5,909,825 | 263 | `408030ee8fcc74e410954bdc0a17de05` |
| `wikipedia_daily.csv` | 7,424,757 | 9,851 | `99597c30d9d0e661917f7aea2f3dfc05` |

The item threshold `--min-obs` is 24 for the three taxi files and 50 for
Wikipedia; the settings of the six development scenarios are in
`tools/run_prediction_plan.py` (`SCENARIOS` and `CONFIRM_SCENARIOS`) and equal
the table of section 2.

## 19. Selection of the gate on synthetic data, and unit tests

The settings of the forecast gate (the number of backtest origins and the source
of the backtest) are chosen on synthetic series only:

```
python experiments/select_gate_synthetic.py --out results/prediction/gate_selection_public
```

Every program of the study has a unit test that runs in a few seconds on
synthetic data and prints `ALL OK`:

| Test | Program tested |
|---|---|
| `tools/test_wspi_forecast.py` | `methods/wspi_forecast.py` (forecasters, gates, bounds, exact reduction to WSPI) |
| `tools/test_level_methods.py`, `tools/test_mechanism_control.py` | WSPI-M, WSPI-H, WSPI-HL and plans D and E |
| `tools/test_supplementary_methods.py`, `tools/test_learned_baseline.py` | WSPI-S, WSPI-FS, SMA, EWMA, sensitivity, GBM and GBM+W, plans C, C_sens and C_nlms |
| `tools/test_prediction.py`, `tools/test_prediction_resume.py` | batched scorers, runner, resume and bound check, plan A and B |
| `tools/test_prediction_stats.py`, `tools/test_prediction_stats_part.py`, `tools/test_prediction_stats_references.py`, `tools/test_prediction_stats_run_b.py`, `tools/test_prediction_stats_run_d.py`, `tools/test_prediction_stats_block_rule.py` | `tools/run_prediction_stats.py`: paired tests, parts, references, runs B and D, block rule |
| `tools/test_prediction_confirm.py`, `tools/test_wikipedia_converter.py` | held-out data: plans with `--confirm`, paired tests with `--confirm`, Wikipedia converter |
| `tools/test_select_gate_synthetic.py` | `experiments/select_gate_synthetic.py` (reproduces the stored selection) |
| `tools/test_check_reproduction.py`, `tools/test_estimate_runtime.py` | `tools/check_reproduction.py` and the run-time estimators |
| `tools/test_compare_prediction_outputs.py` | `tools/compare_prediction_outputs.py` (section 25) |

```
python tools/test_wspi_forecast.py
python tools/test_prediction.py
```

and so on for every file of the table.

## 20. Runs

`tools/run_prediction_plan.py` starts one job for every scenario and horizon
(1, 8 and 24 slots) as cores and memory allow. If the computer stops, give the
same command again: finished jobs end in seconds and interrupted jobs resume.
`--dry-run` prints the commands only. Every job writes
`results/prediction/run_<plan>/<scenario>/h<h>/` and a log under
`logs/prediction_plan/`.

| Plan | Methods | Scenarios |
|---|---|---|
| `A` | WSPI, WSPI-FG, WSPI-F2, WSPI-F-YW, WSPI-F, WSPI-FT, Persistence, Holt, ARYW, MA64 (with the gate profile and the bound check) | six development scenarios |
| `B` | WSPI, WSPI-FA, ARIMA(1,0,1) | `youtube_hourly`, `taxi_hourly` |
| `C` | WSPI, WSPI-S, WSPI-FS1, WSPI-FS1.5, WSPI-FS2, WSPI-FS3 (the number is `kappa`), SMA8 to SMA64, EWMA8 to EWMA64, GBM, GBM+W | six development scenarios |
| `C_sens` | WSPI and the 18 settings of WSPI-F2 (`c` in 2, 4, 8; `gamma` in 0.1, 0.3, 0.5; `phi_d` in 0.8, 1) | six development scenarios |
| `C_nlms` | convergence of the NLMS forecaster (`experiments/nlms_convergence.py`), horizon 1 | `youtube_hourly`, `taxi_hourly` |
| `D` | WSPI, WSPI-M, WSPI-H | six development scenarios |
| `E` | WSPI, WSPI-H, WSPI-HL | six development scenarios |

```
python tools/run_prediction_plan.py --plan A --jobs 6 --mem-gb 10
python tools/run_prediction_plan.py --plan B --jobs 6
python tools/run_prediction_plan.py --plan C --jobs 6 --mem-gb 10
python tools/run_prediction_plan.py --plan C_sens --jobs 6 --mem-gb 10
python tools/run_prediction_plan.py --plan C_nlms --jobs 2
python tools/run_prediction_plan.py --plan D --jobs 6 --mem-gb 12
python tools/run_prediction_plan.py --plan E --jobs 6 --mem-gb 12
```

`--jobs` is the number of parallel jobs and `--mem-gb` the limit of the summed
estimated peak memory (MovieLens daily needs about 6 GB). Plan A took about 6
hours of method time and plan B, with one maximum-likelihood fit per item and
window, about 99 hours (the study used six parallel jobs). The estimators
`tools/estimate_runtime.py` and `tools/estimate_plan_runtime.py` predict the time
from a short preview run.

## 21. Paired tests on the development data

`tools/run_prediction_stats.py` calls `tools/stats_report.py` on every run
folder: circular block bootstrap (B = 10,000, seed 42), block-level two-sided
Wilcoxon test with Holm correction, block length `max(b, h)` slots for the base
block `b` of the scenario and the horizon `h`. Before any test it checks that the
runs are paired (same windows and items in every method, and WSPI equal across
plans). The first 30 % of the common windows of plan A are the tuning part and
the rest the test part. Run the commands in this order; the test part of plan C
needs the selection of `kappa` written by the tuning part.

```
python tools/run_prediction_stats.py --root results/prediction
python tools/run_prediction_stats.py --root results/prediction --part tune --runs C
python tools/run_prediction_stats.py --root results/prediction --part test --runs C C_sens
python tools/run_prediction_stats.py --root results/prediction --part test --runs C --references WSPI-F-YW --out results/prediction/stats_explore
python tools/run_prediction_stats.py --root results/prediction --part test --runs B
python tools/run_prediction_stats.py --root results/prediction --part tune --runs D
python tools/run_prediction_stats.py --root results/prediction --part test --runs D
```

| Output folder | Content |
|---|---|
| `stats/` | Plans A and B on all windows |
| `stats_tune/`, `stats_test/` | Plan C on the tuning part (selection of `kappa`) and on the test part (with `C_sens`) |
| `stats_explore/` | The test part with WSPI-F-YW as an additional reference |
| `stats_test_B/` | Plan B on the test part |
| `stats_tune_D/`, `stats_test_D/` | The level methods of plan D on the tuning and test parts |

Every folder holds `all_paired_tests.csv` (the column `verdict` is `ref_better`,
`ref_worse` or `n.s.`), `all_method_summary.csv`, `part_windows.csv`,
`window_check.csv` and `stats_plan.json`. The options of the program are
described in the header of `tools/run_prediction_stats.py`.

## 22. Runs and tests on the held-out data

The same programs run on the four held-out scenarios with `--confirm`. The runs
go to `results/prediction/confirm/` and use the `kappa` selected on the
development data.

```
python tools/run_prediction_plan.py --confirm --plan A --jobs 6 --mem-gb 12
python tools/run_prediction_plan.py --confirm --plan D --jobs 6 --mem-gb 12
python tools/run_prediction_plan.py --confirm --plan C --jobs 6 --mem-gb 12
python tools/run_prediction_plan.py --confirm --plan E --jobs 6 --mem-gb 12
python tools/run_prediction_stats.py --root results/prediction --confirm --part tune
python tools/run_prediction_stats.py --root results/prediction --confirm --part test
```

The outputs are `confirm/stats_tune/` and `confirm/stats_test/`.

## 23. Mechanism control and block-length sensitivity

Plan E scores WSPI-H without the structural factor (WSPI-HL). Its paired tests
(reference WSPI-HL against WSPI-H and MA64, test part) are:

```
python tools/run_prediction_stats.py --root results/prediction --part test --runs E
python tools/run_prediction_stats.py --root results/prediction --confirm --part test --runs E
```

with the outputs `stats_test_E/` and `confirm/stats_test_E/`.

The block length is also tested with `--block-rule window`, which uses the block
`max(b, N + h - 1)` with the window length `N = 64`, so that the windows of two
blocks share less of their input. Every call needs an `--out` folder of its own:

```
python tools/run_prediction_stats.py --root results/prediction --part test --runs C C_sens --block-rule window --out results/prediction/stats_test_window
python tools/run_prediction_stats.py --root results/prediction --part test --runs D --block-rule window --out results/prediction/stats_test_D_window
python tools/run_prediction_stats.py --root results/prediction --part test --runs B --block-rule window --out results/prediction/stats_test_B_window
python tools/run_prediction_stats.py --root results/prediction --confirm --part test --block-rule window --out results/prediction/confirm/stats_test_window
python tools/run_prediction_stats.py --root results/prediction --part test --runs E --block-rule window --out results/prediction/stats_test_E_window
python tools/run_prediction_stats.py --root results/prediction --confirm --part test --runs E --block-rule window --out results/prediction/confirm/stats_test_E_window
```

## 24. Decision rules and folder index

`tools/prediction_decision.py` applies the rules fixed for plans A and B (the
comparison of the forecast gate with the structural gate, the claims of a gain
over the best time-domain forecast, and the comparison with ARIMA). It reads only
the `verdict` column of the paired tests:

```
python tools/prediction_decision.py --root results/prediction
```

The output is `results/prediction/decision/`.

| Folder under `results/prediction/` | Content |
|---|---|
| `run_A/` ... `run_E/`, `run_C_sens/`, `run_C_nlms/` | Window-by-window results of the plans of section 20 |
| `confirm/run_A/` ... `confirm/run_E/` | The same plans on the held-out data |
| `stats*/`, `confirm/stats*/` | Paired tests of sections 21 to 23 |
| `decision/` | Output of `tools/prediction_decision.py` |
| `decision_test/`, `decision_explore/`, `decision_paper/`, `decision_D/`, `confirm/decision/` | Output of `tools/prediction_test_analysis.py` (section 25) |
| `decision_E/` | Output of `tools/prediction_mechanism_read.py` (section 25) |
| `decision_review/` | Output of `tools/prediction_block_sensitivity.py` (section 25) |
| `prediction_numbers.csv` | Output of `tools/prediction_numbers.py` (section 25) |
| `gate_selection*/` | Selection of the gate on synthetic data |
| `repro/`, `smoke*/` | Checks and short test runs (not used for any result) |

## 25. Tables, figures and decision files

The programs of this section read result files only (CSV and JSON under
`results/prediction/`). They compute no statistic: the verdicts, means and
intervals come from `tools/run_prediction_stats.py` (sections 21 to 23). They
need `pandas`, `numpy` and `matplotlib` and write into folders of their own,
never into the run folders. Do sections 20 to 24 first, then run the commands
below in this order.

**Numbers file.** `tools/prediction_numbers.py` collects the numbers of plans A
and B (means, intervals, run times, gate profile, score bounds) into
`results/prediction/prediction_numbers.csv`. It stops with exit code 1 if a
folder is missing, if a mean differs from the run summary, or if a score bound
was violated.

```
python tools/prediction_numbers.py
```

**Decision files.** Where two parts are listed, run the tuning part first: the
test part reads its output.

| Command (all with `--root results/prediction`) | Output under `results/prediction/` |
|---|---|
| `python tools/prediction_test_analysis.py` | `decision_test/` |
| `python tools/prediction_test_analysis.py --explore` | `decision_explore/` |
| `python tools/prediction_test_analysis.py --paper` | `decision_paper/` |
| `python tools/prediction_test_analysis.py --run-d tune`, then `--run-d test` | `decision_D/` |
| `python tools/prediction_test_analysis.py --confirm tune`, then `--confirm test` | `confirm/decision/` |
| `python tools/prediction_mechanism_read.py` | `decision_E/` |
| `python tools/prediction_block_sensitivity.py --block`, then `--tables` | `decision_review/` |

`--block` reads the reports of section 23 that were written with
`--block-rule window`. With `--figures DIR` the last program also writes four
supplementary Markdown tables into `DIR`. The header of each program lists its
inputs and outputs. One more program gives an aggregate view of all
wavelet-domain methods on the test part (average rank over the four main
metrics, first place by metric, gain against the best baseline of all methods,
and the pairs of one model in the two domains):

```
python tools/prediction_aggregate_analysis.py results/prediction OUTPUT_FOLDER
```

**Tables and figures.** `tools/prediction_figures.py` writes Markdown tables
(`tab_*.md`), LaTeX tables (`tab_*.tex`), figures (`fig_*.pdf` and `.png`) and a
manifest (`figures_manifest*.json`) that names the result files each item was
made from. Each option writes its own set; the header of the program lists the
files of every option. The `--points` run replaces the figures of the plain run
that have the same names with the final versions (points with error bars), so
run the plain command first and the `--points` command after it into the same
folder.

```
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --manuscript --si
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --test-part
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --test-part --points
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --paper-test
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --paper-test --points
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --confirm-tables
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --si-test
python tools/prediction_figures.py --root results/prediction --out paper_figures/prediction --frame
```

`--out` must not lie inside `results/`.

**Check against stored copies.** `tools/compare_prediction_outputs.py` compares
every file of one folder with the file of the same name in another folder:

```
python tools/compare_prediction_outputs.py paper_figures/prediction STORED_FOLDER
python tools/compare_prediction_outputs.py results/prediction_check/decision_test results/prediction/decision_test
```

Tables, CSV and JSON files must be identical; line endings, the one-line note
`<!-- Generated by ... -->` of a table and the `created` and `root` fields of a
JSON file are not compared. A JSON file that holds the SHA-256 of another file
(for example `screen_tune.json` in `decision_D/`) matches only if that file has
the line endings of the stored run. PNG files are compared pixel by pixel; their
size can change by a few pixels with the version of `matplotlib` or the fonts of
the computer, so a differing PNG file is listed as a warning (`--strict-png` makes
it an error). PDF files are not compared. The program exits with 1 if any table,
CSV or JSON file differs.
