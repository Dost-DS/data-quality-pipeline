"""Download the NYC TLC Yellow Taxi Trip Records for 2024.""" 
import os
import time
import urllib.request

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

start = time.time()
total_bytes = 0

for month in range(1, 13):
    filename = f"yellow_tripdata_2024-{month:02d}.parquet"
    out_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(out_path):
        print(f"exists, skipping: {filename}")
        total_bytes += os.path.getsize(out_path)
        continue

    print(f"downloading {filename} ...", end=" ", flush=True)
    urllib.request.urlretrieve(f"{BASE_URL}/{filename}", out_path)
    size = os.path.getsize(out_path)
    total_bytes += size
    print(f"{size / 1024 / 1024:.1f} MB")

elapsed = time.time() - start
print(f"\n12 files, {total_bytes / 1024 / 1024:.1f} MB total, {elapsed:.0f}s")
