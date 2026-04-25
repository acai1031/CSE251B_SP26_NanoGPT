import glob
import numpy as np
from pathlib import Path

src = Path("./external/build-nanogpt/edu_fineweb10B")
dst = Path("./external/nanoGPT/data/fineweb")
dst.mkdir(parents=True, exist_ok=True)

def write_bin(pattern, out_file, max_shards=None):
    files = sorted(glob.glob(str(src / pattern)))
    if max_shards:
        files = files[:max_shards]

    print(f"Writing {out_file} from {len(files)} shards")
    with open(dst / out_file, "wb") as f:
        for p in files:
            arr = np.load(p).astype(np.uint16)
            arr.tofile(f)
            print(p, arr.shape)

write_bin("edufineweb_train_*.npy", "train.bin")
# write_bin("edufineweb_val_*.npy", "val.bin", max_shards=1)