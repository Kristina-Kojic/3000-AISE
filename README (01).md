# DS3000 Project — Group 1

**Predicting cardiovascular disease from routine health check-up data.**

| | |
|---|---|
| **Team** | Kristina, Romy, Rakan, Brandon |
| **Dataset** | #19 Cardiovascular Disease — OpenML ID 45547 (https://www.openml.org/d/45547) |
| **Task** | Binary classification (target: `cardio`, 1 = disease, 0 = no disease) |
| **Notebook** | `DS3000_Project_Group01.ipynb` (all project code is in this notebook) |

## Repository structure
```
DS3000_Project_Group01.ipynb   main project notebook
requirements.txt               Python libraries needed to run the notebook
data/
  cardio.arff                  dataset (about 3 MB)
  README.md                    data source and description
PLAN.md, TRACKER.md, ...       team planning documents
```

## How to run
**Option 1 — Google Colab (no installation)**
1. Open https://colab.research.google.com → **File → Open notebook → GitHub**.
2. Paste the repository link and open `DS3000_Project_Group01.ipynb`.
3. **Runtime → Run all.** The notebook loads the data directly from this repository.

**Option 2 — Local Jupyter**
1. Clone this repository.
2. Install the libraries: `pip install -r requirements.txt`
3. Open Jupyter **from the repository folder** (so the relative path `data/cardio.arff` works).
4. Open the notebook and choose **Kernel → Restart & Run All**.

**Run order:** run all cells from top to bottom. **Expected runtime:** about 30–60 seconds.
**Random seed:** `SEED = 42` is used for every random operation, so results are reproducible.

## Project stages
| Stage | Date | Commit |
|---|---|---|
| Stage 1 demo (EDA, split, baseline models) | Lab 5 | *(add commit hash before the demo)* |
| Stage 2 demo | Lab 8 | |
| Stage 3 demo | Lab 11 | |

## Planning documents
- **[PLAN.md](PLAN.md)** — checklist and deadlines
- **[NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)** — step-by-step guide to each notebook section
- **[TRACKER.md](TRACKER.md)** — who does what, and when
- **[GLOSSARY.md](GLOSSARY.md)** — key terms explained
