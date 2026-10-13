# Downloads 2024 NYC TLC yellow taxi trip records (12 months) and the taxi zone lookup into data\raw,
# then compares SHA-256 hashes with data\raw_checksums.sha256 (the files used for the reported results).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\download.ps1
$ErrorActionPreference = "Stop"
$base = "https://d37ci6vzurychx.cloudfront.net"
$out = Join-Path $PSScriptRoot "..\data\raw"
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($m in 1..12) {
    $f = "yellow_tripdata_2024-{0:D2}.parquet" -f $m
    $dest = Join-Path $out $f
    if (Test-Path $dest) { Write-Host "skip $f"; continue }
    Write-Host "get  $f"
    Invoke-WebRequest -Uri "$base/trip-data/$f" -OutFile $dest -UseBasicParsing
}
$z = Join-Path $out "taxi_zone_lookup.csv"
if (-not (Test-Path $z)) { Invoke-WebRequest -Uri "$base/misc/taxi_zone_lookup.csv" -OutFile $z -UseBasicParsing }

$sums = Join-Path $PSScriptRoot "..\data\raw_checksums.sha256"
$bad = 0
foreach ($line in Get-Content $sums) {
    $hash, $name = $line -split "\s+", 2
    $actual = (Get-FileHash (Join-Path $out $name) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $hash) { Write-Warning "$name differs from the reported run"; $bad++ }
}
if ($bad -eq 0) { Write-Host "All checksums match." }
