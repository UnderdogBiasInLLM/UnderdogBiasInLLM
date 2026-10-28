"""Retail marketplace (VerdictBench): sample construction and per-dispute records.

The VerdictBench release (see ``data/download/marketplace.md``) ships 6,000 disputes,
each in a folder ``splits2/<split>/<case_id>/`` holding ``case_data.json`` and the
attached image (``.png``) and video (``.mp4``) files. The release is read in place:
``raw_dir`` may contain

* the numbered multipart archive ``splits2_2.zip.001``, ``.002``, ... as downloaded
  (directly or in a ``data/`` subfolder); it wraps a stored inner ``splits2.zip``;
* the inner ``splits2.zip`` after extracting the outer archive; or
* the extracted ``splits2/`` folder.

Screening (Appendix C.1 of the paper), applied to the pooled train/val/test partitions:

1. all 17 juror votes present                                  (6,000 -> 5,829)
2. buyer and seller rationale-list lengths equal the vote counts (-> 5,815)
3. no media attachment at or after the first adjudication or
   end-of-record cutoff in the dispute-handling timeline          (-> 5,491)
4. parsed transaction price in RMB, 0 < price < 1,000,000         (-> 5,489)

Each retained dispute is one adjudication unit and its own inference cluster. The
real outcome is the 17-juror majority (buyer = underdog, seller = stronger party).

``load_records`` returns the pre-decision material a prompt needs: claims, product
information, chat messages and dispute-handling evidence truncated before the first
adjudication, in source order, with video attachments omitted, plus the fixed
deduplication / 100-image cap plan for the few records with more than 100 images and
the two actual-juror opinions used by the opinion conditions.
"""
from __future__ import annotations

import bisect
import hashlib
import io
import json
import math
import random
import re
import struct
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pandas as pd

# ----------------------------------------------------------------------------- constants
ADJUDICATION_MARKERS = ("小法庭出判决结果", "客服判决，执行方案：")
END_OF_RECORD_MARKER = "没有更多了"
CHAT_VOTE_MARKER = "投票结果"
TIMESTAMP = re.compile(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})")
PRICE = re.compile(r"[¥￥]\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
MAX_PRICE = 1_000_000
N_JURORS = 17

SCREENING = (  # (step, expected disputes remaining)
    ("released disputes", 6000),
    ("17 juror votes", 5829),
    ("vote and rationale counts consistent", 5815),
    ("no media at or after the cutoff", 5491),
    ("price in RMB 0-1,000,000", 5489),
)

MAX_IMAGES = 100
IMAGE_GROUPS = ("goods", "buyer", "seller", "unknown")  # provenance strata for the cap

# Opinion conditions show one actual juror per side, drawn with a fixed seed from the
# jurors on that side who wrote a non-empty rationale.
JUROR_SELECTION_SEED = 20260902

UNIT_COLUMNS = ["unit_id", "cluster_id", "weight", "real_underdog",
                "buyer_votes", "seller_votes", "bilateral_opinion"]


# ----------------------------------------------------------------------------- release access
class _Multipart(io.RawIOBase):
    """Numbered archive parts exposed as one seekable read-only stream."""

    def __init__(self, parts, start=0, length=None):
        self.parts = list(parts)
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts, total = [], 0
        for size in self.sizes:
            self.starts.append(total)
            total += size
        self.start = start
        self.total = total - start if length is None else length
        self.position, self.handle, self.index = 0, None, None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.total
        self.position = min(max(offset, 0), self.total)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.total - self.position
        size = min(size, self.total - self.position)
        chunks = []
        while size:
            physical = self.start + self.position
            index = bisect.bisect_right(self.starts, physical) - 1
            if index != self.index:
                if self.handle:
                    self.handle.close()
                self.handle, self.index = self.parts[index].open("rb"), index
            offset = physical - self.starts[index]
            self.handle.seek(offset)
            block = self.handle.read(min(size, self.sizes[index] - offset))
            if not block:
                break
            chunks.append(block)
            self.position += len(block)
            size -= len(block)
        return b"".join(chunks)

    def close(self):
        if self.handle:
            self.handle.close()
        super().close()


class Release:
    """Read-only, in-place access to the VerdictBench release (see module docstring).

    Members are addressed by their archive path, e.g. ``splits2/test/1/case_data.json``.
    Every public function of this module accepts either a directory or an open Release
    (opening the release once avoids re-reading the archive index).
    """

    def __init__(self, raw_dir):
        raw_dir = Path(raw_dir)
        self._zip = self._stream = None
        self._root = None
        for folder in (raw_dir, raw_dir / "data"):
            parts = sorted(p for p in folder.glob("splits2_2.zip.*") if p.suffix[1:].isdigit())
            if parts:
                numbers = [int(p.suffix[1:]) for p in parts]
                if numbers != list(range(1, len(parts) + 1)):
                    raise FileNotFoundError("Incomplete multipart archive in " + str(folder))
                self._zip, self._stream = self._open_multipart(parts)
                break
        else:
            if (raw_dir / "splits2.zip").is_file():
                self._zip = zipfile.ZipFile(raw_dir / "splits2.zip")
            elif (raw_dir / "splits2").is_dir():
                self._root = raw_dir
            else:
                raise FileNotFoundError("VerdictBench release not found in " + str(raw_dir))
        self._names = None if self._zip is None else self._zip.namelist()
        self._goods = None

    @staticmethod
    def _open_multipart(parts):
        stream = _Multipart(parts)
        outer = zipfile.ZipFile(stream)
        info = outer.getinfo("splits2.zip")
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("Expected a stored inner splits2.zip")
        stream.seek(info.header_offset)
        header = struct.unpack("<4s5H3L2H", stream.read(zipfile.sizeFileHeader))
        if header[0] != b"PK\x03\x04":
            raise RuntimeError("Invalid inner archive header")
        start = info.header_offset + zipfile.sizeFileHeader + header[-2] + header[-1]
        outer.close()
        stream.close()
        inner = _Multipart(parts, start=start, length=info.file_size)
        return zipfile.ZipFile(inner), inner

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._zip is not None:
            self._zip.close()
        if self._stream is not None:
            self._stream.close()

    def names(self):
        if self._names is None:
            self._names = sorted(p.relative_to(self._root).as_posix()
                                 for p in (self._root / "splits2").rglob("*") if p.is_file())
        return self._names

    def read(self, member):
        if self._zip is not None:
            return self._zip.read(member)
        return (self._root / member).read_bytes()

    def case_files(self):
        """Every ``case_data.json`` member, in sorted path order."""
        return sorted(n for n in self.names() if n.endswith("/case_data.json"))

    def goods_images(self, case_dir):
        """Product-image file names of one dispute folder, in numeric order."""
        if self._goods is None:
            self._goods = {}
            for name in self.names():
                path = PurePosixPath(name)
                if path.name.startswith("image_goods_") and path.suffix.lower() == ".png":
                    self._goods.setdefault(str(path.parent), []).append(path.name)
        goods = self._goods.get(case_dir, [])
        return sorted(goods, key=lambda n: int(PurePosixPath(n).stem.rsplit("_", 1)[-1]))


@contextmanager
def _release(raw_dir):
    """Use an open Release as is, or open (and afterwards close) one from a directory."""
    if isinstance(raw_dir, Release):
        yield raw_dir
    else:
        with Release(raw_dir) as release:
            yield release


def _unit_id(member):
    split, case_id = PurePosixPath(member).parts[1:3]
    return f"{split}-{case_id}"


def _case_dir(unit_id):
    split, case_id = unit_id.split("-", 1)
    return f"splits2/{split}/{case_id}"


# ----------------------------------------------------------------------------- screening
def normalize_rationale(text):
    """Collapse whitespace and drop the masked juror-name prefix ("***: ...")."""
    normalized = " ".join(str(text).split())
    for separator in (":", "："):
        prefix, found, remainder = normalized.partition(separator)
        if found and "***" in prefix:
            return remainder.strip()
    return normalized


def parse_price(case):
    """First RMB amount in the product text, or None."""
    texts = case.get("fixed_infos", {}).get("product_info", {}).get("text_infos", []) or []
    for text in texts:
        match = PRICE.search(str(text))
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def media_after_cutoff(case):
    """Number of media attachments at or after the first adjudication/end-of-record record."""
    evidences = case.get("evidences", []) or []
    cutoff = len(evidences)
    for index, evidence in enumerate(evidences):
        text = evidence.get("evidence_text", "") or ""
        if any(marker in text for marker in ADJUDICATION_MARKERS) or END_OF_RECORD_MARKER in text:
            cutoff = index
            break
    return sum(len(e.get("image_video_paths", []) or []) for e in evidences[cutoff:])


def screen(raw_dir):
    """One row per released dispute with the screening variables and step flags."""
    rows = []
    with _release(raw_dir) as release:
        for member in release.case_files():
            case = json.loads(release.read(member))
            details = case["case_details"]
            buyer_votes, seller_votes = map(int, details["votes"])
            buyer_reasons = details.get("reviewer4buyer_views", []) or []
            seller_reasons = details.get("reviewer4seller_views", []) or []
            price = parse_price(case)
            rows.append({
                "unit_id": _unit_id(member),
                "buyer_votes": buyer_votes,
                "seller_votes": seller_votes,
                "complete_votes": buyer_votes + seller_votes == N_JURORS,
                "consistent_rationales": (len(buyer_reasons) == buyer_votes
                                          and len(seller_reasons) == seller_votes),
                "no_media_after_cutoff": media_after_cutoff(case) == 0,
                "price_in_range": price is not None and 0 < price < MAX_PRICE,
            })
    frame = pd.DataFrame(rows)
    if not frame.unit_id.is_unique:
        raise ValueError("Duplicate dispute identifiers in the release")
    return frame


def screening_counts(screened):
    """Disputes remaining after each cumulative screening step."""
    keep = pd.Series(True, index=screened.index)
    counts = [("released disputes", int(keep.sum()))]
    for (step, _), column in zip(SCREENING[1:], ("complete_votes", "consistent_rationales",
                                                  "no_media_after_cutoff", "price_in_range")):
        keep &= screened[column]
        counts.append((step, int(keep.sum())))
    return counts


def build_units(raw_dir):
    """The 5,489 retained disputes with real outcome (17-juror majority) and vote counts.

    Columns: unit_id, cluster_id (= unit_id), weight (1), real_underdog (1 if the
    majority favors the buyer), buyer_votes, seller_votes, bilateral_opinion (1 if
    both sides received at least one vote; these 4,662 disputes form the opinion
    conditions' sample).
    """
    screened = screen(raw_dir)
    counts = screening_counts(screened)
    if [n for _, n in counts] != [n for _, n in SCREENING]:
        raise ValueError(f"Screening counts {counts} differ from the paper's {list(SCREENING)}")
    kept = screened[screened.complete_votes & screened.consistent_rationales
                    & screened.no_media_after_cutoff & screened.price_in_range]
    units = pd.DataFrame({
        "unit_id": kept.unit_id,
        "cluster_id": kept.unit_id,
        "weight": 1.0,
        "real_underdog": (kept.buyer_votes > kept.seller_votes).astype(int),
        "buyer_votes": kept.buyer_votes,
        "seller_votes": kept.seller_votes,
        "bilateral_opinion": ((kept.buyer_votes > 0) & (kept.seller_votes > 0)).astype(int),
    })
    return units[UNIT_COLUMNS].reset_index(drop=True)


# ----------------------------------------------------------------------------- records
def _pre_decision(case):
    """Evidence records and chat messages before the first adjudication, in source order.

    The evidence timeline stops at the first record containing an adjudication marker
    or the end-of-record marker. The chat stops at the first vote-result message or at
    the first message time-stamped at or after the adjudication time.
    """
    evidence, cutoff = [], None
    for item in case.get("evidences", []):
        text = item.get("evidence_text", "")
        positions = [text.find(m) for m in ADJUDICATION_MARKERS if m in text]
        if positions:
            first = min(positions)
            preceding = list(TIMESTAMP.finditer(text[:first]))
            if preceding:
                cutoff = tuple(map(int, preceding[-1].groups()))
            else:
                following = TIMESTAMP.search(text[first:])
                if following:
                    cutoff = tuple(map(int, following.groups()))
            break
        if END_OF_RECORD_MARKER in text:
            break
        evidence.append(item)
    chat = []
    for message in case.get("fixed_infos", {}).get("chat_history", []):
        if CHAT_VOTE_MARKER in message:
            break
        stamp = TIMESTAMP.search(message)
        if cutoff and stamp and tuple(map(int, stamp.groups())) >= cutoff:
            break
        chat.append(message)
    return evidence, chat


def _side(text):
    text = text.lstrip()
    return "buyer" if text.startswith("买家") else "seller" if text.startswith("卖家") else "unknown"


def select_jurors(case):
    """One actual juror rationale per side (fixed seed), or None if that side has none."""
    details = case["case_details"]
    selected = {}
    for side, field in (("BUYER", "reviewer4buyer_views"), ("SELLER", "reviewer4seller_views")):
        reasons = [normalize_rationale(r) for r in details.get(field, []) or []]
        candidates = [r for r in reasons if r]
        if not candidates:
            selected[side] = None
            continue
        material = f"{JUROR_SELECTION_SEED}|select-{side}|{case['case_hash']}".encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
        selected[side] = candidates[rng.randrange(len(candidates))]
    return selected


def _allocate(counts, cap):
    """Largest-remainder quotas over provenance strata; every nonempty stratum keeps >= 1."""
    total = sum(counts.values())
    if total <= cap:
        return dict(counts)
    raw = {g: cap * counts.get(g, 0) / total for g in IMAGE_GROUPS}
    out = {g: max(1, math.floor(raw[g])) if counts.get(g, 0) else 0 for g in IMAGE_GROUPS}
    while sum(out.values()) > cap:
        g = min((g for g in IMAGE_GROUPS if out[g] > 1),
                key=lambda g: (raw[g] - out[g], IMAGE_GROUPS.index(g)))
        out[g] -= 1
    while sum(out.values()) < cap:
        g = max((g for g in IMAGE_GROUPS if out[g] < counts.get(g, 0)),
                key=lambda g: (raw[g] - out[g], -IMAGE_GROUPS.index(g)))
        out[g] += 1
    return out


def _even_indices(length, count):
    """``count`` evenly spaced positions in ``range(length)``, endpoints included."""
    if count == 0:
        return []
    if count == 1:
        return [(length - 1) // 2]
    return [(i * (length - 1) * 2 + count - 1) // (2 * (count - 1)) for i in range(count)]


def image_cap_plan(images, read_bytes, cap=MAX_IMAGES):
    """Fixed rule for records with more than ``cap`` images (None otherwise).

    Images with identical decoded RGBA pixels are reduced to their first occurrence;
    if more than ``cap`` distinct images remain, quotas proportional to the number of
    distinct images per stratum (product, buyer, seller, unattributed evidence) are
    filled with evenly spaced images within each stratum. Returns the retained image
    ordinals and, for each duplicate, the ordinal of the image it repeats.
    """
    if len(images) <= cap:
        return None
    from PIL import Image  # only needed for the few records above the cap

    seen, duplicate_of, distinct = {}, {}, []
    for image in images:
        with Image.open(io.BytesIO(read_bytes(image))) as decoded:
            rgba = decoded.convert("RGBA")
            key = (rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest())
        if key in seen:
            duplicate_of[image["ordinal"]] = seen[key]
        else:
            seen[key] = image["ordinal"]
            distinct.append(image)
    strata = {g: [x for x in distinct if x["group"] == g] for g in IMAGE_GROUPS}
    quotas = _allocate({g: len(v) for g, v in strata.items()}, cap)
    kept = sorted(strata[g][i]["ordinal"] for g in IMAGE_GROUPS
                  for i in _even_indices(len(strata[g]), quotas[g]))
    return {"kept": kept, "duplicate_of": duplicate_of}


def load_records(raw_dir, unit_ids, image_dir=None):
    """Per-unit prompt material, keyed by unit_id.

    Each record holds the dispute title, buyer and seller claims, product texts,
    product images, pre-decision chat messages, pre-decision evidence records (text and
    images in source order; videos omitted), the image-cap plan (or None), the vote
    counts and one selected actual-juror rationale per side.

    Images are dicts ``{"ordinal", "group", "member", "path"}``: ``ordinal`` numbers the
    retained images 1, 2, ... in prompt order, ``member`` is the archive path. If
    ``image_dir`` is given, every image is extracted to ``image_dir/<unit_id>/<file>``
    and ``path`` holds that local path; otherwise ``path`` is None.
    """
    records = {}
    with _release(raw_dir) as release:
        for unit_id in unit_ids:
            case_dir = _case_dir(unit_id)
            case = json.loads(release.read(case_dir + "/case_data.json"))
            if f"{PurePosixPath(case_dir).parts[1]}-{case['case_id']}" != unit_id:
                raise ValueError("Unexpected dispute folder for " + unit_id)
            details = case["case_details"]
            product = case.get("fixed_infos", {}).get("product_info", {})
            ordinal = 0

            def image(name, group):
                nonlocal ordinal
                ordinal += 1
                return {"ordinal": ordinal, "group": group, "member": f"{case_dir}/{name}", "path": None}

            product_images = [image(name, "goods") for name in release.goods_images(case_dir)]
            evidence_items, chat = _pre_decision(case)
            evidence, videos = [], 0
            for item in evidence_items:
                text = item.get("evidence_text", "")
                group = _side(f"{text}")
                images = []
                for path in item.get("image_video_paths", []):
                    name = PurePosixPath(path).name
                    suffix = PurePosixPath(name).suffix.lower()
                    if suffix == ".mp4":
                        videos += 1  # video attachments are omitted
                    elif suffix == ".png":
                        images.append(image(name, group))
                    else:
                        raise ValueError(f"Unexpected attachment type in {unit_id}: {name}")
                evidence.append({"text": text, "images": images})
            all_images = product_images + [x for e in evidence for x in e["images"]]
            plan = image_cap_plan(all_images, lambda x: release.read(x["member"]))
            if image_dir is not None:
                target = Path(image_dir) / unit_id
                target.mkdir(parents=True, exist_ok=True)
                for x in all_images:
                    local = target / PurePosixPath(x["member"]).name
                    if not local.exists():
                        local.write_bytes(release.read(x["member"]))
                    x["path"] = str(local)
            buyer_votes, seller_votes = map(int, details["votes"])
            records[unit_id] = {
                "unit_id": unit_id,
                "title": case.get("basecase_info", {}).get("title", ""),
                "buyer_claim": details.get("buyer_claim", ""),
                "seller_claim": details.get("seller_claim", ""),
                "product_texts": product.get("text_infos", []),
                "product_images": product_images,
                "chat": chat,
                "evidence": evidence,
                "omitted_videos": videos,
                "image_cap": plan,
                "buyer_votes": buyer_votes,
                "seller_votes": seller_votes,
                "juror_opinions": select_jurors(case),
            }
    return records
