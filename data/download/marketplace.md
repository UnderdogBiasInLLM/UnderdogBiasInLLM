# Retail marketplace: VerdictBench

**Source.** VerdictBench, the 6,000-dispute release accompanying *CyberJurors: A
Multi-Agent Simulation Task for E-Commerce Disputes Verdict* (Sun et al., 2026).

- Dataset: https://huggingface.co/datasets/piggi/VerdictBench
- Code: https://github.com/YanhuiS/CyberJurors
- License: CC BY-NC 4.0. The dataset is gated: request access on the dataset page,
  state your affiliation and intended (non-commercial, academic) use, and wait for
  approval. The terms prohibit commercial use and re-identification of disputants.
  This repository does not redistribute any of its content.

## Download

After access is granted, log in with a Hugging Face token and download the dataset
repository, for example:

```bash
huggingface-cli login
huggingface-cli download piggi/VerdictBench --repo-type dataset --local-dir <raw_dir>
```

The release is about 250 GB: a multipart ZIP archive `data/splits2_2.zip.001`,
`.002`, ... that wraps a stored inner archive `splits2.zip`. Each dispute is a
folder `splits2/<split>/<case_id>/` with `case_data.json` and its image (`.png`)
and video (`.mp4`) attachments.

## Placement

Pass the download folder as `raw_dir` to `src/build/marketplace.py`. Any of these
layouts is accepted, and nothing needs to be extracted:

- the numbered parts `splits2_2.zip.001`, `.002`, ..., either directly in `raw_dir`
  or in `raw_dir/data/` (read in place);
- the inner `splits2.zip` in `raw_dir` (after extracting the outer archive);
- the extracted `splits2/` folder in `raw_dir`.

## Use

```python
from src.build.marketplace import Release, build_units, load_records
from src.prompts.marketplace import build_request

with Release(raw_dir) as release:
    units = build_units(release)                      # 5,489 disputes
    records = load_records(release, units.unit_id, image_dir=image_dir)
request = build_request(records[unit_id], "C0", model="qwen3.5-flash")
```

`build_units` checks the screening chain against the paper (6,000 → 5,829 → 5,815 →
5,491 → 5,489). `load_records` extracts only the images used in the prompts to
`image_dir` (omit `image_dir` to keep archive member names instead of local paths).
Records with more than 100 images need Pillow (`pip install pillow`) to apply the
image deduplication rule.
