# Data sources and licenses

The repository does not redistribute any case text. It ships only identifiers,
real-outcome labels, numeric features, and the models' coded decisions. The source
data must be obtained from the original providers.

| Setting | Source | Terms | What this repository contains |
|---|---|---|---|
| Retail marketplace | VerdictBench (https://huggingface.co/datasets/piggi/VerdictBench; code at https://github.com/YanhuiS/CyberJurors) | CC BY-NC 4.0; access requires accepting the dataset's conditions; not for re-identification of disputants | Dispute identifiers, juror vote counts, real outcomes, coded model decisions |
| Housing | Housing Ombudsman Service published decisions | Public decisions of the Housing Ombudsman Service | Report and complaint-head identifiers, recorded findings, numeric record features, coded model decisions |
| Labor | CPEE (https://github.com/LiliizZ/CPEE) | No license stated by the authors; not redistributed | Plea and case identifiers, real outcomes, numeric record features, coded model decisions |
| Divorce property | China Judgments Online decisions distributed through the third-party `cncases/cases` archive (https://github.com/cncases/cases) | Published court judgments containing personal information; not redistributed | Claim and case identifiers, real outcomes, numeric record features, coded model decisions |

Instructions for obtaining each source are in `data/download/`.

The code in this repository is released under the MIT License (`LICENSE`).
Derived files in `data/derived/`, `decisions/`, and `posttraining/` are provided for
verifying the paper's results and remain subject to the terms of the underlying sources.
