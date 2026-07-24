from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .openniji import OpenNijiImageCache, default_openniji_jsonl, read_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch OpenNiji URL images into the Adeleine /data cache")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--image_cache", type=Path, default=Path("/data/shasegawa/adeleine/openniji/images"))
    parser.add_argument("--max_records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl = args.jsonl or default_openniji_jsonl(args.hf_home)
    records = read_records(jsonl, args.image_cache, max_records=args.max_records)
    cache = OpenNijiImageCache(args.image_cache)
    ok = 0
    failed = 0
    for record in tqdm(records, total=len(records)):
        try:
            cache.get(record, download=True)
            ok += 1
        except Exception:
            failed += 1
    print({"ok": ok, "failed": failed, "image_cache": str(args.image_cache)})


if __name__ == "__main__":
    main()
