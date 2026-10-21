# Notebook Guide — Step by Step, With the WHY

> [!NOTE]
> This was our early plan. The notebook is the final version.

This guide walks through our notebook `DS3000_Project_GroupNN.ipynb` **in order**, one step at a time.

- 📅 Who does each step, and by when → **[TRACKER.md](TRACKER.md)**
- ✅ Short checklist version → **[PLAN.md](PLAN.md)**
- 📖 Don't know a word? → **[GLOSSARY.md](GLOSSARY.md)**

---

## Read this first: the 3 rules behind almost every "why"

Almost every decision in this project comes back to one of these three rules. If the TA asks "why did you do X?", the answer is usually one of them.

| Rule | In plain words | What goes wrong if we break it |
|---|---|---|
| **Rule 1 — Test data is a sealed exam** | The model must be graded on patients it has **never seen**. We don't look at the test set until the very end. | Our score looks better than it really is, like seeing the exam before writing it. |
| **Rule 2 — Only learn from the training set** | Anything that "learns" from data (scaler, model, tuning) only gets to see the **training** set. | Info from validation/test leaks in, and scores are too optimistic ("data leakage"). |
| **Rule 3 — Treat every model the same** | Same data, same split, same metrics, same timing for every model. | The comparison is unfair, so we can't honestly say which model is best. |

---

## How each step is written

Every step looks like this:

> **Goal** — what we're doing, in one line
> **Why** — the reason (usually tied to a Rule)
> **Code** — a starting outline. Fill in the `TODO`s.
> **✅ Done when** — how we know the step is finished
> **🗣️ If the TA asks** — a likely demo question with a short answer

Each step = one **Markdown heading cell** in the notebook (e.g. `## Step 4 — Clean the data`) followed by its code cells.

---

## The whole thing on one page

```
PART A → Demo 1 (Oct 22)          PART B → Demo 2 (Nov 12)         PART C → Demo 3 (Dec 3)
─────────────────────────         ─────────────────────────        ─────────────────────────
Step 0  Title + team              Step 10 Tree models              Step 14 PCA + clustering
Step 1  Setup                     Step 11 Neural network (MLP)     Step 15 Final test (ONCE)
Step 2  Load data                 Step 12 Tuning                   Step 16 Where it fails
Step 3  Explore (EDA)             Step 13 Pick + save winner       Step 17 Smoke test
Step 4  Clean
Step 5  New features
Step 6  Split + FREEZE
Step 7  Scale
Step 8  Helper functions
Step 9  Baseline models
```

---

# PART A — Data + simple models (Demo 1, Oct 22)

## Step 0 — Title, team, contribution log

**Goal:** A Markdown cell at the top (title, dataset, group #, names) and a contribution log at the very bottom.

**Why:** Marks are individual. The TA checks who did what using this log plus our GitHub commits.

**✅ Done when:** names and group number are at the top, and the log table exists at the bottom.

---

## Step 1 — Setup

**Goal:** All imports in one cell, plus one random seed used everywhere.

**Why:**
- A fixed seed means **everyone gets the exact same numbers** every time. Otherwise the report and the notebook won't match.
- Keeping imports in one place means the notebook runs top to bottom without errors.

**Code:**
```python
import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42                                  # use this EVERYWHERE something is random
os.makedirs("artifacts", exist_ok=True)    # saved split + tables
os.makedirs("models", exist_ok=True)       # saved final model
```

**✅ Done when:** the cell runs with no errors on everyone's laptop.

**🗣️ If the TA asks** "Why a seed?" → "So anyone who runs our notebook gets exactly our numbers."

---

## Step 2 — Load the data

**Goal:** Get dataset 45547 into a pandas table.

**Why:** `fetch_openml` is built into scikit-learn. Nobody has to pass files around, and it caches after the first download.

**Code:**
```python
from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=45547, as_frame=True)
df = data.frame

print(df.shape)     # (rows, columns)
df.head()
df.dtypes           # TODO: check whether 'cardio' loaded as text/category
```

**✅ Done when:** we know the number of rows and columns, and we've confirmed the column names match what we expected.

**🗣️ If the TA asks** "What's your target?" → "`cardio`: 1 = has heart disease, 0 = doesn't. It's a yes/no answer, so this is classification."

---

## Step 3 — Explore the data (EDA)

**Goal:** Look at the data **before** changing anything.

**Why:**
- You can't clean what you haven't looked at.
- **Class balance decides our metrics.** If it's ~50/50, accuracy is fair. If it's 90/10, accuracy would be misleading.
- It finds problems (missing values, duplicates, impossible numbers) and hints at which features matter.

**Code:**
```python
df.describe()                   # min/max expose impossible values fast
df.isna().sum()                 # missing values per column
df["cardio"].value_counts()     # class balance

# Fig 1 — class balance
sns.countplot(x="cardio", data=df); plt.title("Fig 1. Class balance"); plt.show()

# Fig 2 — one feature split by the target
sns.histplot(data=df, x="ap_hi", hue="cardio", bins=50); plt.title("Fig 2. Systolic BP by class"); plt.show()

# Fig 3 — correlations (like the Lecture 2 heatmap)
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f"); plt.title("Fig 3. Correlations"); plt.show()

# Which gender code is which? Men are usually taller on average.
df.groupby("gender")["height"].mean()
```

**Label every figure** (Fig 1, Fig 2, ...). The report must use the same labels.

**✅ Done when:** we have 2+ labelled plots and a short Markdown cell listing 3–5 things we found.

**🗣️ If the TA asks** "What did EDA tell you?" → Give real findings, e.g. "The classes are roughly balanced, blood pressure has impossible values, and cholesterol looks linked to disease."

---

## Step 4 — Clean the data

**Goal:** Fix or remove bad data using **fixed rules we choose ourselves**.

**Why, line by line:**
| Action | Why |
|---|---|
| Drop `id` | It's just a row number, not health info. A model could learn nonsense from it. |
| Age: days → years | So humans can read and explain it. |
| Drop duplicate rows | **Rule 1.** If the same patient ends up in both train and test, the model has already "seen" that answer. |
| Remove impossible values | A blood pressure of 11000 is a typo, and it confuses the model. |

**Is it OK to clean before splitting?** Yes, **as long as the rules are fixed numbers** we chose (like "BP between 70 and 250"). They don't learn anything from the data. Anything that uses the data's own averages or percentiles has to happen **after** the split (Rule 2).

**Code:**
```python
df = df.drop(columns=["id"], errors="ignore")
df["cardio"] = df["cardio"].astype(int)              # make sure the target is 0/1 numbers

df["age_years"] = (df["age"] / 365.25).round(1)
df = df.drop(columns=["age"])

df["gender"] = df["gender"].map({1: 0, 2: 1})        # TODO: confirm coding from Step 3

before = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", before - len(df))

# TODO: agree on cutoffs as a group, explain them in a Markdown cell
ok = (
    df["ap_hi"].between(70, 250) &
    df["ap_lo"].between(40, 150) &
    (df["ap_lo"] < df["ap_hi"]) &                   # bottom BP number must be lower than top
    df["height"].between(120, 220) &
    df["weight"].between(30, 200)
)
print("Impossible rows removed:", (~ok).sum())
df = df[ok].copy()
```

Leave `cholesterol` and `gluc` as 1/2/3. They're **ordered** (normal < high < very high), so the numbers already mean something.

**✅ Done when:** there are no impossible values left, and a Markdown cell says how many rows we removed and why.

**🗣️ If the TA asks** "Did removing rows hurt?" → Give the %. If it's small, the data is still representative. If it's large, we'd revisit the cutoffs.

---

## Step 5 — New features (optional)

**Goal:** Add BMI and pulse pressure.

**Why:** Doctors look at **BMI**, not height and weight separately. A more meaningful number can help simple models like logistic regression.

**Code:**
```python
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
```

**✅ Done when:** we've compared validation scores with and without these (after Step 9).

---

## Step 6 — Split the data, then FREEZE it

**Goal:** 60% train / 20% validation / 20% test, saved so it never changes.

**Why:**
| Part | Job | Rule |
|---|---|---|
| **Train (60%)** | The model learns from this | Rule 2 |
| **Validation (20%)** | We compare models and pick settings with this | Rule 1 |
| **Test (20%)** | Final exam, used **once** at the end | Rule 1 |

- `stratify=y` keeps the same sick/healthy mix in every part, so no part is accidentally easier (Rule 3).
- Saving the split proves it's frozen.
- We drop `cardio` **by name**, so the answer can't sneak into the inputs (Lecture 2, "bug #2").

**Code:**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["cardio"])
y = df["cardio"]

# 60% train, 40% leftover
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED)

# leftover → half validation, half test (20% / 20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print(len(X_train), len(X_val), len(X_test))        # → Table 1. Split sizes

split = pd.concat([
    pd.DataFrame({"row": X_train.index, "part": "train"}),
    pd.DataFrame({"row": X_val.index,   "part": "val"}),
    pd.DataFrame({"row": X_test.index,  "part": "test"}),
])
split.to_csv("artifacts/split_indices.csv", index=False)
```

**✅ Done when:** `artifacts/split_indices.csv` is saved and pushed. **Nobody changes this cell after that.**

**🗣️ If the TA asks** "Why three sets, not two?" → "If we tune on test, we're fitting to the exam. Validation absorbs the tuning, so test stays a fair final grade."

---

## Step 7 — Scale the data

**Goal:** Put all features on the same scale (mean 0, standard deviation 1).

**Why:**
- Blood pressure is around 120 and `smoke` is 0 or 1. **KNN** measures distance, so blood pressure would drown out everything else. Logistic regression, SVM and MLP also train better on scaled data (Lecture 2, slide 21).
- **Fit on train only (Rule 2).** The scaler learns a mean and std. If it learns them from test, test info has leaked in (Lecture 2, "bug #1").

**Code:**
```python
from sklearn.preprocessing import StandardScaler

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)    # LEARN mean/std from train
X_val_s   = scaler.transform(X_val)          # only APPLY
X_test_s  = scaler.transform(X_test)         # only APPLY — don't touch until Step 15
```

**✅ Done when:** `fit_transform` appears **only** on `X_train`.

**🗣️ If the TA asks** "Why not scale everything together?" → "Then the scaler has seen the test data, and the test score isn't truly on unseen data."

---

## Step 8 — Helper functions

**Goal:** Two small functions that **every** model goes through.

**Why:** **Rule 3.** One shared function means every model gets exactly the same data, metrics and timing. It also means less copy-paste, so fewer bugs.

**Our metrics:**
| Metric | Plain meaning | Why we use it |
|---|---|---|
| Accuracy | % we got right | Easy to read. Fair because classes are ~50/50 |
| **Recall** | Of the sick people, how many did we catch? | **Our main focus.** Missing a sick patient is the worst mistake for a screening tool |
| Precision | Of the people we flagged, how many were really sick? | Stops us "cheating" recall by flagging everyone |
| F1 | Balance of recall and precision | The single number we use to pick the winner |

**Code:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_scores(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }

def run_model(name, model, X_tr, y_tr, X_ev, y_ev):
    """Train on (X_tr, y_tr), score on (X_ev, y_ev), and time both."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_ev)
    predict_s = time.perf_counter() - t0

    return {"model": name, **get_scores(y_ev, pred),
            "train_s": round(train_s, 3), "predict_s": round(predict_s, 3)}

results = []    # every model adds one row here → our comparison table
trained = {}    # every trained model is kept here → used again in Step 15
```

**✅ Done when:** both functions run, and we can explain every metric in one sentence.

**🗣️ If the TA asks** "Why recall?" → "A missed sick patient goes home thinking they're fine. A false alarm just means an extra check-up."

---

## Step 9 — Baseline models

**Goal:** One "dumb" model plus 3 simple models, all scored on **validation**.

**Why each one:**
| Model | What it does | Why it's here |
|---|---|---|
| Dummy | Always guesses the most common answer | Our "zero point". A real model has to beat this or it learned nothing |
| Logistic Regression | Draws a straight-line boundary; uses maximum likelihood (Lecture 3) | The standard simple classifier |
| KNN | Looks at the most similar patients and copies their answer | Simple, no training math |
| SVM | Finds the widest gap between the two classes | Strong classic model. **Slow on big data**, so we train it on a sample |

**Code:**
```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

baselines = {
    "Dummy":  DummyClassifier(strategy="most_frequent"),
    "LogReg": LogisticRegression(max_iter=1000, random_state=SEED),
    "KNN":    KNeighborsClassifier(n_neighbors=15),     # TODO: try a few k values
}
for name, model in baselines.items():
    results.append(run_model(name, model, X_train_s, y_train, X_val_s, y_val))
    trained[name] = model

# SVM on a 10k-row sample (the full set is too slow) — say this in Markdown
idx = np.random.default_rng(SEED).choice(len(X_train_s), size=10000, replace=False)
svm = SVC(kernel="rbf", random_state=SEED)
results.append(run_model("SVM (10k sample)", svm,
                         X_train_s[idx], y_train.iloc[idx], X_val_s, y_val))
trained["SVM"] = svm

pd.DataFrame(results)       # → Table 2. Baseline comparison (validation)
```

**Parameter counts (required):** LogReg = `model.coef_.size + 1`. KNN = none (it just stores the training data). SVM = `svm.support_vectors_.shape[0]` support vectors.

**✅ Done when:** Table 2 shows all models with scores, times and parameter counts, plus 1 conclusion and 1 open risk written underneath.

**🗣️ If the TA asks** "Why a dummy model?" → "A score only means something compared to doing nothing smart. The dummy shows us that bar."

### 🎤 Demo 1 checklist (Oct 22)
- [ ] Kernel → Restart & Run All, no errors
- [ ] `X_test` is not used anywhere yet
- [ ] Pushed to GitHub, commit ID written in TRACKER.md
- [ ] Everyone can explain Steps 1–9 using the **Why** parts

---

# PART B — Better models + tuning (Demo 2, Nov 12)

## Step 10 — Tree models

**Goal:** Add a Decision Tree, Random Forest and XGBoost.

**Why:**
| Model | Plain idea | Why it's here |
|---|---|---|
| Decision Tree | A chain of yes/no questions ("ap_hi > 140?") | Very easy to explain, but overfits easily |
| Random Forest | Many trees vote | More stable than one tree |
| XGBoost | Trees built one after another, each fixing the last one's mistakes | Often the strongest on table data |

Trees can catch patterns that aren't straight lines (e.g. risk jumping after a certain age), which logistic regression can't.

**Code:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

trees = {
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "XGBoost":      XGBClassifier(n_estimators=300, max_depth=4,
                                  learning_rate=0.1, random_state=SEED),
}
for name, model in trees.items():
    results.append(run_model(name, model, X_train_s, y_train, X_val_s, y_val))
    trained[name] = model
```

**✅ Done when:** all 3 are in the same `results` table as the baselines.

**🗣️ If the TA asks** "Why keep the baselines?" → "To prove the fancy models are worth it. If XGBoost barely beats LogReg, the simpler model might be the smarter choice."

---

## Step 11 — Neural network (MLP)

**Goal:** Train a small neural network and plot its **learning curve** (train loss vs validation loss per epoch).

**Why:**
- It's required, and it's our deep learning model. (No CNN, since our data isn't images. Write that down.)
- **The learning curve shows overfitting.** Train loss going down while validation loss goes up means it's memorizing, not learning.

**Code:**
```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=SEED)
train_loss, val_loss = [], []

for epoch in range(50):                                   # TODO: pick # of epochs
    mlp.partial_fit(X_train_s, y_train, classes=[0, 1])   # one pass over the data
    train_loss.append(log_loss(y_train, mlp.predict_proba(X_train_s)))
    val_loss.append(log_loss(y_val,   mlp.predict_proba(X_val_s)))

plt.plot(train_loss, label="train"); plt.plot(val_loss, label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.title("Fig X. MLP learning curve"); plt.show()

n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
results.append({"model": "MLP", **get_scores(y_val, mlp.predict(X_val_s))})  # TODO: add times
trained["MLP"] = mlp
```

**✅ Done when:** there's a learning curve plot and one sentence saying whether it overfits.

**🗣️ If the TA asks** "Is it overfitting?" → Point at the plot. "Validation loss stops improving around epoch X while train keeps dropping, so after that it starts memorizing."

---

## Step 12 — Tuning (hyperparameter search)

**Goal:** Try different settings for our strongest models and keep the best.

**Why:**
- Default settings are rarely the best for our data.
- **Tune on validation only** (Rule 1). Tuning on test means picking whatever happens to suit the exam.
- **Write the plan BEFORE running it.** The project sheet requires this so we can't cherry-pick after seeing results.

**Step 12a — Markdown cell (write this first):**
> *Plan: tune Random Forest and XGBoost. Objective = validation F1. Budget = max ~20 combos per model. Stop when all combos are done.*

**Step 12b — Code:**
```python
from sklearn.model_selection import ParameterGrid

grid = {"n_estimators": [100, 300], "max_depth": [5, 10, None]}   # TODO: our choices

trials = []
for params in ParameterGrid(grid):
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    row = run_model(f"RF {params}", model, X_train_s, y_train, X_val_s, y_val)
    row.update(params)
    trials.append(row)

trials_df = pd.DataFrame(trials).sort_values("f1", ascending=False)
trials_df                                                # → Table X. Tuning trials
# TODO: repeat for XGBoost (and MLP if time)
```

**✅ Done when:** there's a tuning table for each tuned model, and the plan cell sits **above** the results.

**🗣️ If the TA asks** "Why not tune with the test set?" → "Then the test score would reflect choices made using the test set. It wouldn't be unseen anymore."

---

## Step 13 — Pick + save the winner

**Goal:** Pick one final model using **validation F1** (training time as a tiebreaker), then save it together with the scaler.

**Why:**
- Pick by **numbers**, not by which model sounds cool. The project sheet says exactly this.
- **Save the scaler with it.** New patients have to be scaled exactly like the training data, or the model gets garbage inputs.
- Saving means Demo 3 just loads the model. No retraining live.

**Code:**
```python
import joblib

best_model = trained["XGBoost"]   # TODO: the real winner — explain why in Markdown

joblib.dump({"model": best_model, "scaler": scaler, "features": list(X_train.columns)},
            "models/final_model.joblib")
```
⚠️ Check the file size. GitHub rejects anything over 100 MB, and big random forests can get that large.

**✅ Done when:** the model file is saved and there's one Markdown sentence saying why it won.

**🗣️ If the TA asks** "Why this model?" → "Highest validation F1 (X), good recall (Y), and reasonable training time (Z s)."

### 🎤 Demo 2 checklist (Nov 12)
- [ ] Restart & Run All, no errors
- [ ] One table with **all** models (baselines + trees + MLP), including params and times
- [ ] Tuning plan cell sits above the tuning results
- [ ] `X_test` still unused
- [ ] Pushed, commit ID in TRACKER.md
- [ ] Everyone can explain Steps 10–13

---

# PART C — Final results + analysis (Demo 3, Dec 3)

## Step 14 — PCA + clustering

**Goal:**
- **PCA**: squeeze our features into fewer "combo" features and see how much info each one keeps.
- **K-means**: group patients by similarity **without** looking at the answer.

**Why:**
- PCA shows whether our features overlap a lot (height, weight and BMI say similar things). Few components holding most of the info = lots of overlap.
- Clustering checks whether natural patient groups exist, and whether some groups have much more disease.
- Both are fit on **train only** (Rule 2).

**Code:**
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(random_state=SEED).fit(X_train_s)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("number of components"); plt.ylabel("share of info kept")
plt.title("Fig X. PCA"); plt.show()

km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_train_s)   # TODO: try k = 2..6
pd.Series(y_train.values).groupby(km.labels_).mean()    # disease rate per cluster
```

**✅ Done when:** there's one sentence each on what PCA and clustering showed.

---

## Step 15 — Final test: ONCE

**Goal:** Load the saved model, run **every** model on the test set **one time**, and make one final table.

**Why:** This is the honest answer to "how well does it work on new patients?" (Rule 1). We run it once so we can't tweak things afterwards to look better. The project sheet explicitly bans changing anything after seeing test results.

**Code:**
```python
saved = joblib.load("models/final_model.joblib")
X_test_final = saved["scaler"].transform(X_test[saved["features"]])

final_rows = []
for name, model in trained.items():                    # same test set for every model (Rule 3)
    final_rows.append({"model": name, **get_scores(y_test, model.predict(X_test_s))})

final_table = pd.DataFrame(final_rows)                  # → Table X. Final test results
final_table
```

**✅ Done when:** the final table exists and **nothing above it gets changed afterwards**.

**🗣️ If the TA asks** "Is test close to validation?" → If yes: "Our validation was a reliable guide." If test is much worse: "We may have over-tuned to validation."

---

## Step 16 — Where the model fails

**Goal:** Split the test results by group (age, gender, cholesterol...) and find where the model is weakest.

**Why:**
- One overall score hides problems. 75% recall overall could be 50% for young patients.
- **At least one failure case is required**, and showing we understand the limits earns marks.
- Fairness check (optional): does it work equally well for men and women?

**Code:**
```python
from sklearn.metrics import recall_score

res = X_test.copy()
res["true"] = y_test.values
res["pred"] = saved["model"].predict(X_test_final)
res["age_group"] = pd.cut(res["age_years"], bins=[0, 40, 50, 60, 100])

res.groupby("age_group", observed=True).apply(
    lambda g: recall_score(g["true"], g["pred"]), include_groups=False)
# TODO: repeat by gender, cholesterol level, BMI group
```

**Robustness idea:** add small random noise to the test inputs (like a slightly wrong BP reading) and see how much the scores drop.

**✅ Done when:** we can name at least 1 group where the model is weak, and give 2–3 real-world limitations (e.g. smoking and alcohol are self-reported).

**🗣️ If the TA asks** "When shouldn't someone trust this model?" → Name the weak group plus the data limitations.

---

## Step 17 — Smoke test

**Goal:** A tiny cell at the end that proves everything works, in seconds.

**Why:** It's required. The TA can check the pipeline without waiting for any training.

**Code:**
```python
saved = joblib.load("models/final_model.joblib")
assert set(X_train.index).isdisjoint(X_test.index), "train/test overlap!"
assert "cardio" not in saved["features"],            "target leaked into features!"
print(saved["model"].predict(saved["scaler"].transform(X_test[saved["features"]].head(5))))
print("Smoke test passed ✅")
```

**✅ Done when:** it prints "Smoke test passed ✅".

### 🎤 Demo 3 checklist (Dec 3)
- [ ] Restart & Run All, no errors, outputs visible
- [ ] Every figure/table labelled, matching the report
- [ ] Test set used **only** in Steps 15–17
- [ ] Smoke test passes
- [ ] README says how to run it, and `requirements.txt` exists
- [ ] Pushed, commit ID in TRACKER.md
- [ ] **Everyone** can explain **every** step (the TA can ask anyone anything)
