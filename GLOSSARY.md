# 📖 What Does That Mean? — Plain-English Explainer

Every term used in our docs, explained simply, with examples from **our** heart disease data.
If a word in PLAN, TRACKER or NOTEBOOK_GUIDE confuses you, look it up here.

**Jump to:** [The data](#1-the-data) · [Cleaning](#2-cleaning-the-data) · [Splitting](#3-splitting-the-data) · [Scaling](#4-scaling) · [Models](#5-models) · [Scores](#6-scores-metrics) · [Tuning](#7-tuning) · [Extra analysis](#8-extra-analysis) · [GitHub + Jupyter](#9-github--jupyter)

---

## 1. The data

**Dataset**
The whole table of data. Ours has one row per patient (~70,000 patients).

**Row**
One patient.

**Column / Feature**
One piece of info about each patient, like `age`, `weight`, or `cholesterol`. The model uses features to make its guess.
> In code, the features are called **`X`**.

**Target (label)**
The thing we're trying to predict. Ours is **`cardio`**: 1 = has heart disease, 0 = doesn't.
> In code, the target is called **`y`**.

**Classification**
Predicting a **category** (yes/no, cat/dog). Ours is classification because the answer is 0 or 1.
(The other kind is *regression*, which predicts a number like a price. That's what Lectures 1–2 covered.)

**EDA (Exploratory Data Analysis)**
Looking at the data with tables and plots **before** doing anything else, to understand it and spot problems.
> **What we do:** Step 3. Make plots, check balance, check for missing and weird values.

**Class balance**
How many 0s vs 1s there are. Ours is roughly 50/50, which is "balanced".
> **Why it matters:** if it were 95% healthy, a model that always says "healthy" would score 95% accuracy and be useless. Balanced data means accuracy is a fair score.

**Missing values**
Empty cells in the table. Usually shown as `NaN`.
> **What we do:** count them with `df.isna().sum()`. If there are none, we just say so.

---

## 2. Cleaning the data

**Cleaning**
Fixing or removing bad data so it doesn't confuse the model.

**Duplicate rows**
Two rows that are exactly the same.
> **Why remove them:** if the same patient ends up in both training and test, the model has basically already seen the answer, and our score looks better than it really is.

**Outlier**
A value that's very far from the rest. It can be **real but rare** (a 2.1 m tall person) or **impossible** (a typo).

### ✂️ Cleaning cutoff — the one everyone asks about

**What it is:**
A rule we set, like *"blood pressure must be between 70 and 250"*. Any row outside that range gets **removed**, because the value can't be real. It's almost certainly a typo.

**Why we need it:**
This dataset has typos in it. A top blood pressure (`ap_hi`) of **11000** or **−150** is impossible, since a normal reading is around **120**. If we leave these in, the model tries to learn from garbage.

**Real example:**
| Patient | `ap_hi` (top BP) | `ap_lo` (bottom BP) | Keep? | Why |
|---|---|---|---|---|
| 1 | 120 | 80 | ✅ Keep | Normal reading |
| 2 | 160 | 100 | ✅ Keep | High BP, but real (this is exactly what we want the model to learn from!) |
| 3 | 11000 | 90 | ❌ Remove | Impossible, probably meant 110 |
| 4 | 120 | 800 | ❌ Remove | Impossible |
| 5 | 80 | 120 | ❌ Remove | Bottom number can't be higher than top, so they were probably swapped |

**How we decide the cutoffs (the call on Sat Oct 3):**
1. Run `df.describe()` to see the min and max of each column. That shows the crazy values.
2. Pick limits that are **generous**. Only cut things that are clearly impossible, not just unusual. Sick patients have unusual values, and we need them!
3. Count how many rows get removed. If it's a small %, we're fine. If it's a lot, loosen the limits.
4. Write the cutoffs and the reasons in a Markdown cell in the notebook.

**Suggested starting cutoffs (discuss and change if needed):**
| Column | Keep if between | Why |
|---|---|---|
| `ap_hi` (top BP) | 70 – 250 | Outside this range isn't a realistic reading |
| `ap_lo` (bottom BP) | 40 – 150 | Same idea |
| `ap_lo` vs `ap_hi` | `ap_lo` must be less than `ap_hi` | The bottom number is always lower |
| `height` (cm) | 120 – 220 | Adults outside this range are almost certainly typos |
| `weight` (kg) | 30 – 200 | Same idea |

**Why it's OK to do this before splitting:** these are fixed numbers **we chose**, not something calculated from the data. (The opposite would be "remove the top 1% of values". That uses the data itself, so it would have to happen after splitting, on the training set only.)

**Feature engineering**
Making new, more useful columns from existing ones.
> **Ours:**
> - **BMI** = weight ÷ height² (in metres). Doctors use it for body size.
> - **Pulse pressure** = top BP − bottom BP.

---

## 3. Splitting the data

**Train / validation / test split**
Cutting the data into 3 parts:
| Part | Size | Analogy | Used for |
|---|---|---|---|
| **Train** | 60% | Textbook + homework | The model learns from this |
| **Validation** | 20% | Practice exams | Comparing models, choosing settings |
| **Test** | 20% | The final exam | Grading the final model **once**, at the very end |

> **Why:** if you study using the final exam questions, your grade doesn't show what you actually learned. Same for the model.

**Freezing the split**
Saving exactly which rows went into which part, and never changing it after. That way every model is judged on the same patients.

**Stratify**
Making sure each part has the same mix of sick and healthy patients (e.g. all three are ~50/50).
> **Why:** otherwise one part could end up accidentally easier or harder.

**Random seed (`SEED = 42`)**
A fixed number that makes "random" things (like shuffling for the split) come out the same every time.
> **Why:** so all four of us, and the TA, get identical results. 42 isn't special, it just has to stay the same.

**Data leakage**
When information from the validation or test data sneaks into training, so scores look better than they really are. It's the #1 thing the course warns about.
> **Examples:** fitting the scaler on all the data, accidentally leaving `cardio` inside `X`, or tuning using the test set.

---

## 4. Scaling

**Scaling / standardizing**
Converting every feature to the same scale, so each column has an average of 0 and a spread of 1.
> **Why:** blood pressure is in the 100s and `smoke` is 0 or 1. Some models (KNN, SVM, logistic regression, MLP) would think blood pressure matters 100× more just because the numbers are bigger.

**StandardScaler**
The scikit-learn tool that does the scaling.

**`fit` vs `transform`**
- **fit** = *learn* something from data (e.g. the scaler learns the average and spread)
- **transform** = *apply* what was learned
- **fit_transform** = both at once

> **Our rule:** `fit` only on **train**. Then only `transform` validation and test.

---

## 5. Models

**Model**
The thing that learns a pattern from the training data and then makes guesses on new patients.

**Training (fitting) a model**
Letting the model learn from the training data: `model.fit(X_train, y_train)`.

**Predicting**
Asking the trained model for its guesses: `model.predict(X_val)`.

**Baseline**
A simple model we use as a starting point. Fancier models must beat it to be worth using.

**Dummy model**
The dumbest possible baseline. It always guesses the most common answer. If a real model can't beat it, the real model learned nothing.

**The models we use, in one line each:**
| Model | Plain idea |
|---|---|
| Logistic Regression | Draws a straight line between sick and healthy |
| KNN (K-Nearest Neighbours) | "What did the 15 most similar patients have?" |
| SVM (Support Vector Machine) | Finds the widest gap between the two groups |
| Decision Tree | A flowchart of yes/no questions |
| Random Forest | Lots of decision trees voting |
| XGBoost | Trees built one after another, each fixing the last one's mistakes |
| MLP (neural network) | Layers of simple math units that learn together (our "deep learning" model) |

**Parameters**
The numbers a model **learns by itself** during training (e.g. the weights in logistic regression). We have to report how many each model has.

**Overfitting**
The model **memorizes** the training data instead of learning general patterns. It scores great on train but badly on validation, like memorizing answers instead of understanding the material.

**Underfitting**
The model is too simple to catch the pattern, so it's bad on both train and validation.

---

## 6. Scores (metrics)

First, the 4 possible outcomes for each patient:
| | Model says **sick** | Model says **healthy** |
|---|---|---|
| **Actually sick** | ✅ True positive | ❌ **False negative** (missed them, the worst one!) |
| **Actually healthy** | ❌ False positive (false alarm) | ✅ True negative |

This table is called a **confusion matrix**.

| Metric | Question it answers | Example |
|---|---|---|
| **Accuracy** | What % did we get right overall? | 73 out of 100 correct = 73% |
| **Recall** | Of the people who ARE sick, how many did we catch? | 80 of 100 sick people flagged = 80% |
| **Precision** | Of the people we SAID were sick, how many really were? | 70 of 100 flagged were really sick = 70% |
| **F1** | One score that balances recall and precision | Used to pick our winner |

> **Why recall is our focus:** for a screening tool, sending a sick person home (false negative) is much worse than a false alarm.

**Training time / inference time**
How long a model takes to learn, and how long it takes to make predictions. We have to report both.

---

## 7. Tuning

**Hyperparameter**
A setting **we choose** before training (the model doesn't learn it). Examples: number of trees in a random forest, `k` in KNN, number of layers in the MLP.
> **Parameter vs hyperparameter:** parameters are learned by the model. Hyperparameters are picked by us.

**Tuning / HPO (Hyperparameter Optimization)**
Trying different hyperparameter values to find the best combination, **judged on the validation set**.

**Grid search**
Trying **every combination** from a list. For example, trees = [100, 300] × depth = [5, 10] gives 4 combos to try.

**Tuning plan**
Writing down **before** we start what we'll try, how many combos, and what score we're trying to improve.
> **Why:** so we can't cheat by changing the goal after seeing results. It's required.

**Epoch**
One full pass through the training data (used for the MLP).

**Loss**
A number showing how wrong the model is. Lower = better. The model tries to shrink it while training.

**Learning curve**
A plot of loss over epochs for train AND validation.
> **How to read it:** if the train line keeps going down but the validation line starts going **up**, the model is overfitting.

**Saving / checkpointing a model**
Saving the trained model to a file (with `joblib`) so we can load it later without retraining.

---

## 8. Extra analysis

**PCA (Principal Component Analysis)**
Squeezes many features into a few "summary" features and shows how much info is kept.
> **What it tells us:** if 3 summary features keep 90% of the info, a lot of our columns are saying similar things (e.g. height, weight and BMI).

**Clustering / K-means**
Groups patients by how similar they are, **without** looking at the answer. Then we check whether some groups have way more heart disease.

**Unsupervised learning**
Learning without the answer (like clustering). Our main models are *supervised*, since they learn from the answer.

**CNN**
A neural network for images. **We skip it**, because our data is a table, not images.

**Failure analysis**
Finding **where** the model does badly, e.g. "it misses a lot of sick patients under 40".

**Fairness check**
Checking whether the model works equally well for different groups (e.g. men vs women). Optional for us.

**Robustness test**
Slightly messing with the inputs (adding small noise) to see if the model still works. A robust model doesn't fall apart from small changes.

**Smoke test**
A tiny, fast cell at the end that checks everything loads and runs. It's the TA's quick "does it work?" check.

---

## 9. GitHub + Jupyter

**Repository (repo)**
Our project folder on GitHub.

**Pull**
Download the latest changes from GitHub. **Always do this before you start working.**

**Commit**
Save a snapshot of your changes with a message, e.g. `add age histogram`.

**Push**
Upload your commits to GitHub so the team can see them.

**Commit ID (hash)**
A code like `a3f9c21` that identifies one exact snapshot. We write it down before each demo so the TA knows which version we're presenting.

**Merge conflict**
When two people edited the same part of a file and Git doesn't know which version to keep. Avoid it by **not editing the notebook at the same time**.

**Kernel**
The "engine" running your notebook's code.

**Restart & Run All**
Wipes everything and runs the notebook top to bottom. If it works with no errors, the notebook is reproducible. **Do this before every demo.**

**`requirements.txt`**
A list of the libraries (and versions) our notebook needs, so anyone can install them.

**Reproducible**
Anyone can run our notebook and get the same results. That's why we use a seed, save the split, and list requirements.
