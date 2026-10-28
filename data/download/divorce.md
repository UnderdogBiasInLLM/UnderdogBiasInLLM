# Divorce property: China Judgments Online decisions (cncases archive)

**Source.** Published civil judgments of China Judgments Online, in the annual full-text
archives distributed by the third-party project `cncases/cases`
(https://github.com/cncases/cases). The paper uses the complete
**2009–2018 and 2021** archives.

**Personal information.** The judgments contain names and other personal information of
litigants. This repository does not redistribute any judgment text, party name or prompt
built from one; it ships only the derived claim and case identifiers
(`data/derived/units_divorce.csv`) and code. Keep the downloaded archives, the build cache
and every generated prompt local and out of version control.

## Pinned versions

- `cncases/cases` repository commit `bd8b086476acbe4158fb139197ac17c76d92c607`.
- The archives are distributed by the torrent file `810air.torrent` in that repository:
  SHA-256 `97a24015061f779d76758fbcc8fefae98b81ca578b3a1c8422a74ce4a46dabe1`,
  BitTorrent info-hash `3a7559524a1b73f94f2e8d3920583ded2ef24237`, torrent name
  `裁判文书全量数据（已完成）` (41 files, 101,319,452,767 bytes).

Only the eleven annual ZIPs below are needed (about 61 GB; select them in the torrent
client). Each must match the size and SHA-256 recorded for the paper. Each ZIP holds one
UTF-8 CSV per month with the columns 原始链接, 案号, 案件名称, 法院, 所属地区, 案件类型,
案件类型编码, 来源, 审理程序, 裁判日期, 公开日期, 当事人, 案由, 法律依据, 全文.

| file | bytes | SHA-256 | CSV rows | divorce-related rows |
|---|---:|---|---:|---:|
| `2009年裁判文书数据.zip` | 84,726,043 | `b8b1e1d4c3be4f19a33323c2a4c537fb409e32f3a3e101cdb0281c799b681527` | 73,797 | 2,481 |
| `2010年裁判文书数据.zip` | 212,355,227 | `c6507341f3651b828f96bfaa156342be317edb24bad5400ba21d6f23a40344b1` | 155,190 | 15,122 |
| `2011年裁判文书数据.zip` | 128,542,106 | `d8aedeba761b83242a8bd3e94f1cabfb7e6283f64a5481ef12bca6d0c375e447` | 99,943 | 9,390 |
| `2012年裁判文书数据.zip` | 218,429,588 | `08d6aacd0126701bef240cd221d02897677cde0f82aab243b7c81fc7faa0eb71` | 165,219 | 15,146 |
| `2013年裁判文书数据.zip` | 1,543,026,562 | `10925fa4104694d7665747f320278b6420a5956efd8fe96db247d622bc65ad49` | 991,687 | 96,249 |
| `2014年裁判文书数据.zip` | 7,150,687,964 | `bf71394ed41c8813c268ba5c7c48f7b2c136818c6671a74002b9d846a8a0225c` | 4,929,300 | 465,138 |
| `2015年裁判文书数据.zip` | 10,073,743,020 | `2a8d16882981bc494a83c34a71d9a38fe301a1d6bd9f58e4963187532f25b665` | 7,206,461 | 572,702 |
| `2016年裁判文书数据.zip` | 11,935,695,545 | `1684a3f5664b85ffffc27af44392e2372ee01257e8c47144f6755277590bff31` | 8,997,815 | 522,259 |
| `2017年裁判文书数据.zip` | 13,292,189,254 | `1c588f260f96549c26ef521ed642382a0145422e3a130ce3b29db6005265b92e` | 11,856,934 | 735,656 |
| `2018年裁判文书数据.zip` | 7,207,980,350 | `9c9374572d73a7b8747f47b1a12316c22ba4c4269eac046551035bc360f08271` | 6,586,750 | 415,572 |
| `2021年裁判文书数据.zip` | 9,116,248,703 | `06f242a5c6039486b47fb6272586b304dc6f7ec3d73c57aa0dd7a21aee144b81` | 9,611,124 | 472,254 |
| **total** | | | **50,674,220** | **3,321,969** |

## Placement and use

Put the ZIPs (any sub-folder layout) in a folder outside version control, e.g.
`data/raw/divorce/`, and run from the repository root:

```bash
python -m src.build.divorce data/raw/divorce --out units_divorce_rebuilt.csv
```

The first run streams every CSV of the eleven ZIPs once and caches the divorce-related
rows as `data/raw/divorce/divorce_rows/<year>.jsonl.gz` (about 1.7 GB; it contains
judgment text, so keep it local); later stages read only the cache. Pass
`--cache <folder>` to put it elsewhere. `cache_divorce_rows(raw_dir, cache, verify_hash=True)`
additionally checks each archive's SHA-256. The build prints its counts against the paper:

3,321,969 divorce-related rows → 3,162,013 court-docket records → 38,333 eligible cases →
3,022 fully granted or denied property claims (2,601 cases) → 2,610 candidate records
(2,362 cases) → **1,826 retained claims in 1,696 cases**, with 774 records in the review
queue and 10 merged duplicate fragments.

```python
from src.build.divorce import build_units, load_records
from src.prompts.divorce import build_request
from src.parse.divorce import read_decision

units = build_units("data/raw/divorce")              # equals data/derived/units_divorce.csv
records = load_records("data/raw/divorce", units.unit_id)   # dict: unit_id -> source fields
request = build_request(records[units.unit_id[0]], "C0")    # {"system": ..., "parts": [{"text": ...}]}
```

The docstring of `src/build/divorce.py` documents the screening rules and the twelve fixed
claim-level decisions (two excluded non-prayer
targets and ten reviewed opposition phrases, the latter identified only by hash).
