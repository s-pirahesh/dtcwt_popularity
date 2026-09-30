# Data

The raw data are public but are not part of this repository. Put them here:

```
data/raw/youtube/count_observation_upload.csv       Kaggle, "YouTube videos viewCount every hour" (CC0)
data/raw/yellow_taxi/2025/yellow_tripdata_2025-MM.parquet   NYC TLC Trip Record Data, Yellow Taxi, 01 to 11
data/raw/movielens/ratings.csv                      GroupLens MovieLens 32M (ML-32M)
```

The prepared files go to `data/datasets/` (not kept in Git). The commands that
build them are in [REPRODUCE.md](../REPRODUCE.md), section 1.

## Standard format

Every prepared file is a CSV with one row per item and time slot that has at
least one request:

| Column | Meaning |
|---|---|
| `timestamp` | Start of the time slot (`YYYY-MM-DD HH:MM:SS`, UTC for MovieLens) |
| `item_id` | Item identifier (video, pickup zone or movie) |
| `count` | Number of requests of the item in the slot |

Slots without a row have zero requests; the evaluation fills them with zeros.

## How each dataset is built

**YouTube** (`data/converters/youtube_converter.py`). The signal is the column
`viewCount_diff` of the raw file, that is, the new views of a video in each
hour. Rows with a missing or negative value are removed. Videos with fewer
than 50 views in total are removed (`--youtube-min-views-per-video 50`).
Result: 1,485 videos over 695 hours (7 May to 5 June 2018).

**NYC Yellow Taxi** (`data/converters/yellow_taxi_converter.py`). Each trip is
one request of its pickup zone (`PULocationID`) at its pickup time
(`tpep_pickup_datetime`). Trips are counted per zone and slot of 1 hour,
30 minutes or 5 minutes, from 1 January to 30 November 2025. The file keeps
all 262 zones.

**MovieLens** (`tools/prepare_movielens_v5.py`). The signal is the number of
ratings a movie receives in a day or in a week (Monday to Sunday, UTC, only
complete weeks); the rating value is not used. The period is 1 January 1998 to
12 October 2023, because the years before 1998 have long gaps without ratings.
The weekly and daily files keep the movies with at least 24 ratings in the
period; this pre-filter has no effect on the causal item list used in the
evaluation.

The item list of the evaluation is built causally from these files: an item
enters at slot *t* only when its total count up to *t* reaches the threshold
`--min-obs` (50 for YouTube, 24 for the other datasets).
