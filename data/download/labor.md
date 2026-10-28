# Labor: CPEE (labor-service disputes)

**Source.** CPEE, the Chinese civil-judgment corpus of *CPEE: Civil Case Judgment
Prediction Centering on the Trial Mode of Essential Elements* (Zhao et al., CIKM 2022):
158,625 records with party statements, court fact summaries, plaintiff pleas and
plea-level rulings (0 = denied, 1 = partially granted, 2 = granted).

- Code and data: https://github.com/LiliizZ/CPEE (file `CPEE_dataset.zip`)
- License: the CPEE repository contains no license file, so this repository does not
  redistribute any of its content. Obtain it from the authors' repository and cite the
  paper above.

## Download

Download `CPEE_dataset.zip` from the repository page (or clone the repository, which
contains the archive). The archive holds `CPEE_dataset/train_cpee.json`,
`val_cpee.json` and `test_cpee.json` (about 475 MB uncompressed); macOS metadata entries
under `__MACOSX/` are ignored.

Expected SHA-256 of the three files:

| file | SHA-256 |
|---|---|
| `train_cpee.json` | `a8302807af4c3b7ecdf99d722c426164c584db3f1a765218f12b2e5645d272b6` |
| `val_cpee.json` | `c746fc9057ef6f21324cfb845510f02998a3be9b49bee48ac25c3f9def86c943` |
| `test_cpee.json` | `725abea7c84bd6a8d8b1605def47c8862e09f3ce8704aa0a7840b7c5392cd630` |

## Placement

Put the download in a folder outside version control, e.g. `data/raw/labor/`, and pass
it as `raw_dir` to `src/build/labor.py`. Any of these is accepted, and nothing needs to
be extracted:

- `CPEE_dataset.zip` itself, or a folder containing it;
- a folder containing the extracted `train_cpee.json`, `val_cpee.json`, `test_cpee.json`,
  directly or inside a `CPEE_dataset/` subfolder.

## Use

```bash
python -m src.build.labor data/raw/labor            # prints the screening counts vs the paper
```

```python
from src.build.labor import build_units, load_records
from src.prompts.labor import build_request

units = build_units(raw_dir)                       # 5,562 pleas in 3,359 source cases
records = load_records(raw_dir, units.unit_id)
request = build_request(records[0], "C0")          # {"system": ..., "parts": [{"text": ...}]}
```

The screening chain (Appendix C.3) is 158,625 records → 11,119 labor-service records →
5,374 with a defendant statement → 4,944 after exact deduplication (training, validation,
test order) → 3,702 records with a substantive defendant statement (6,309 pleas) → 5,562
fully granted or fully denied pleas after excluding 747 partially granted pleas.
