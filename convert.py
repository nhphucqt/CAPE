# %%
import msgpack
import json
from pathlib import Path

# %%
result_dir = Path("result_records")
output_dir = Path("result_records_msgpack")

# %%
output_dir.mkdir(exist_ok=True, parents=True)
print(list(result_dir.glob("*.json")))
for result_file in result_dir.glob("*.json"):
    output_file = (output_dir / result_file.name).with_suffix(".msgpack")
    if output_file.exists():
        print(f"Skipping {result_file} as {output_file} already exists")
        continue
    with open(result_file, "r") as f:
        data = json.load(f)
    with open(output_file, "wb") as f:
        msgpack.pack(data, f)

# %%



