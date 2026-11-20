"""Fetch a few BDD100K driving videos without downloading the whole 1.97 TB archive.

    python tools/fetch_bdd100k.py --list                       # the splits, how many videos, a few names
    python tools/fetch_bdd100k.py --count 5 [--split val] [--seed 0] [--out runs/bdd100k]
    python tools/fetch_bdd100k.py --names b1c66a42-6f7d68ca ...

BDD100K's 100,000 driving videos (about 40 s each, 1280 x 720, 30 fps, from dashboard cameras in
US cities) come as one zip, bdd100k_videos.zip. The download host in the BDD100K docs,
dl.cv.ethz.ch, no longer resolves; the Hugging Face dataset linxxx3/bdd100k_videos holds the same
zip byte for byte (md5 253d9a2f9d89d2b09d8d93f397aecdd7, the one the docs give) in 19 parts.

A zip lists its files at its end. This reads that list (about 10 MB, kept in <out>/central-directory.bin),
then asks for the bytes of each chosen video only (HTTP range requests), checks each against its
CRC-32, and writes <out>/videos/<split>/<name>.mov. Python's standard library only.

The data is under the BDD100K license (written to <out>/LICENSE.txt): use for education, research and
not-for-profit purposes; commercial use needs a licence from UC Berkeley.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

BASE = "https://huggingface.co/datasets/linxxx3/bdd100k_videos/resolve/main/bdd100k_videos"
LICENSE = "https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst"
AGENT = {"User-Agent": "vla-drive-fetch/1.0"}


class Archive:
    """The split zip as one byte range: read(start, length) across its parts."""

    def __init__(self):
        manifest = json.loads(self._get(f"{BASE}/manifest.json"))
        self.size = manifest["original_size_bytes"]
        self.part_size = manifest["part_size_bytes"]
        self.count = manifest["part_count"]
        self.naming = manifest["part_naming"].rsplit("-", 1)[0]          # bdd100k_videos.zip.part

    @staticmethod
    def _get(url, first=None, last=None, tries=4) -> bytes:
        headers = dict(AGENT)
        if first is not None:
            headers["Range"] = f"bytes={first}-{last}"
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    data = response.read()
                if first is not None and len(data) != last - first + 1:
                    raise IOError(f"asked for {last - first + 1} bytes, got {len(data)}")
                return data
            except (OSError, IOError) as error:
                if attempt == tries - 1:
                    raise
                print(f"  retrying ({error})", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        raise AssertionError

    def read(self, start: int, length: int) -> bytes:
        out = bytearray()
        while length > 0:
            part, offset = divmod(start, self.part_size)
            n = min(length, self.part_size - offset)
            out += self._get(f"{BASE}/{self.naming}-{part:05d}", offset, offset + n - 1)
            start, length = start + n, length - n
        return bytes(out)


def central_directory(archive: Archive, cache: Path) -> bytes:
    """The zip's list of files (ZIP64: the archive is over 4 GB)."""
    if cache.is_file():
        return cache.read_bytes()
    tail_start = archive.size - 65536
    tail = archive.read(tail_start, 65536)
    end = tail.rfind(b"PK\x05\x06")
    locator = tail.rfind(b"PK\x06\x07", 0, end)
    if end < 0 or locator < 0:
        raise ValueError("no ZIP64 end of central directory in the last 64 KB")
    record_at = struct.unpack_from("<Q", tail, locator + 8)[0]
    record = archive.read(record_at, 56)
    if record[:4] != b"PK\x06\x06":
        raise ValueError("the ZIP64 locator does not point at a ZIP64 record")
    entries, size, offset = struct.unpack_from("<QQQ", record, 32)
    print(f"the archive lists {entries} entries; reading its directory ({size / 1e6:.1f} MB)")
    data = archive.read(offset, size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


def entries(directory: bytes) -> list[dict]:
    """name, method, crc, compressed and uncompressed size, local header offset, for each file."""
    out, at = [], 0
    while at + 46 <= len(directory) and directory[at:at + 4] == b"PK\x01\x02":
        (method, crc, csize, usize, nlen, elen, clen, lho) = (
            *struct.unpack_from("<H", directory, at + 10), *struct.unpack_from("<IIIHHH", directory, at + 16),
            struct.unpack_from("<I", directory, at + 42)[0])
        name = directory[at + 46:at + 46 + nlen].decode("utf-8", "replace")
        extra = directory[at + 46 + nlen:at + 46 + nlen + elen]
        e = 0
        while e + 4 <= len(extra):                                      # ZIP64 sizes and offset
            tag, length = struct.unpack_from("<HH", extra, e)
            if tag == 0x0001:
                values, v = [], e + 4
                for field in ("usize", "csize", "lho"):
                    if {"usize": usize, "csize": csize, "lho": lho}[field] == 0xFFFFFFFF:
                        values.append((field, struct.unpack_from("<Q", extra, v)[0]))
                        v += 8
                for field, value in values:
                    usize, csize, lho = (value if field == "usize" else usize, value if field == "csize" else csize,
                                         value if field == "lho" else lho)
            e += 4 + length
        if not name.endswith("/"):
            out.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, lho=lho))
        at += 46 + nlen + elen + clen
    return out


def fetch(archive: Archive, entry: dict, path: Path) -> float:
    """The entry's bytes, unpacked and checked against its CRC-32, written to `path`. Returns MB/s."""
    started = time.perf_counter()
    header = archive.read(entry["lho"], 30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"no local header for {entry['name']}")
    nlen, elen = struct.unpack_from("<HH", header, 26)
    data = archive.read(entry["lho"] + 30 + nlen + elen, entry["csize"])
    if entry["method"] == 8:
        data = zlib.decompress(data, -15)
    elif entry["method"] != 0:
        raise ValueError(f"{entry['name']}: compression method {entry['method']} is not stored or deflate")
    if zlib.crc32(data) & 0xFFFFFFFF != entry["crc"] or len(data) != entry["usize"]:
        raise ValueError(f"{entry['name']}: the CRC-32 or the size does not match; not written")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return entry["csize"] / 1e6 / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("runs/bdd100k"))
    parser.add_argument("--list", action="store_true", help="the splits and a few names, nothing downloaded")
    parser.add_argument("--split", default="val", help="train, val or test")
    parser.add_argument("--count", type=int, default=0, help="this many videos of --split, picked with --seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--names", nargs="*", default=[], help="video names (with or without .mov)")
    args = parser.parse_args()

    archive = Archive()
    files = entries(central_directory(archive, args.out / "central-directory.bin"))
    by_split: dict[str, list[dict]] = {}
    for f in files:
        parts = f["name"].split("/")
        by_split.setdefault(parts[-2] if len(parts) > 1 else "", []).append(f)
    if args.list or not (args.count or args.names):
        for split, items in sorted(by_split.items()):
            mb = [f["usize"] / 1e6 for f in items]
            print(f"{split:8s} {len(items):6d} videos, {sum(mb) / 1e3:7.1f} GB, median {sorted(mb)[len(mb) // 2]:.1f} MB"
                  f"   e.g. {', '.join(Path(f['name']).stem for f in items[:3])}")
        return 0

    if args.names:
        wanted = {n.removesuffix(".mov") for n in args.names}
        chosen = [f for f in files if Path(f["name"]).stem in wanted]
        missing = wanted - {Path(f["name"]).stem for f in chosen}
        if missing:
            print(f"not in the archive: {', '.join(sorted(missing))}", file=sys.stderr)
    else:
        if args.split not in by_split:
            print(f"no split {args.split!r}; there are {', '.join(sorted(by_split))}", file=sys.stderr)
            return 1
        chosen = random.Random(args.seed).sample(by_split[args.split], min(args.count, len(by_split[args.split])))

    if not (args.out / "LICENSE.txt").is_file():
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "LICENSE.txt").write_bytes(archive._get(LICENSE))
    for f in chosen:
        path = args.out / "videos" / f["name"].split("/")[-2] / Path(f["name"]).name
        if path.is_file() and path.stat().st_size == f["usize"]:
            print(f"  {path} (already here)")
            continue
        speed = fetch(archive, f, path)
        print(f"  {path}  {f['usize'] / 1e6:.1f} MB  CRC-32 ok  {speed:.1f} MB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
