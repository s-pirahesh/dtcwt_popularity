# Data converters

Each converter turns one raw dataset into the standard format
`timestamp,item_id,count` (see [../README.md](../README.md)). They are called
through `prepare_data.py` at the root of the repository:

```
python prepare_data.py --list
python prepare_data.py --dataset <name> --input <raw file or pattern> --output <csv> [options]
```

| Name | File | Options (default) |
|---|---|---|
| `youtube` | `youtube_converter.py` | `--youtube-granularity` (`none`), `--youtube-min-views-per-video` (100; the paper uses 50), `--youtube-start-date`, `--youtube-end-date` |
| `yellow_taxi` | `yellow_taxi_converter.py` | `--yellow_taxi-granularity` (`15min`; the paper uses `hourly`, `30min`, `5min`), `--yellow_taxi-min-trips-per-location` (100; the paper uses 1), `--yellow_taxi-start-date` (the paper uses 2025-01-01), `--yellow_taxi-end-date`, `--yellow_taxi-extract-features` |
| `movielens` | `movielens_converter.py` | general MovieLens converter; the paper uses `tools/prepare_movielens_v5.py` instead |

Generic options of every converter: `--output-format` (`csv`, `parquet`,
`feather`), `--quiet`, `--no-validate`.

## Adding a converter

Subclass `BaseConverter` in `base_converter.py`, set `DATASET_NAME`, describe
the options in `get_specific_params()`, implement `convert_file()` so that it
returns a DataFrame with the columns `timestamp`, `item_id` and `count`, and
register the class with `ConverterFactory.register()`. The options then
appear in `prepare_data.py` automatically.
