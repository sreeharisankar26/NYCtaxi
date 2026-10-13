#!/usr/bin/env bash
# Downloads 2024 NYC TLC yellow taxi trip records (12 months) and the taxi zone lookup into data/raw.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
base="https://d37ci6vzurychx.cloudfront.net"
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  f="yellow_tripdata_2024-$m.parquet"
  [ -f "data/raw/$f" ] && { echo "skip $f"; continue; }
  curl -fSL -o "data/raw/$f" "$base/trip-data/$f"
done
[ -f data/raw/taxi_zone_lookup.csv ] || curl -fSL -o data/raw/taxi_zone_lookup.csv "$base/misc/taxi_zone_lookup.csv"
# compare against the checksums of the files used for the reported results
(cd data/raw && sha256sum -c ../raw_checksums.sha256) || echo "WARNING: checksums differ (TLC may have republished files)"
