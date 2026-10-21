# DS3000 Project Plan — Group NN

**Read this first. It's our whole project on one page.**

📘 Step-by-step "how + why" with code → **[NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)**
📅 Who does what + deadlines → **[TRACKER.md](TRACKER.md)**
📖 What does a word mean? → **[GLOSSARY.md](GLOSSARY.md)**

---

## The basics

- **What we're doing:** Predict if a patient has heart disease (yes / no) from their checkup info.
- **Dataset:** #19 Cardiovascular Disease → https://www.openml.org/d/45547
- **Language:** Python, in ONE Jupyter notebook: `DS3000_Project_GroupNN.ipynb`
- **Type of problem:** Classification (the answer is 0 or 1)
- **Worth:** 45% of our grade

---

## Key dates

| Date | What |
|---|---|
| **Oct 22** (Lab 5) | **Demo 1** — data + simple models (5%) |
| **Nov 12** (Lab 8) | **Demo 2** — better models + tuning (10%) |
| **Dec 3** (Lab 11) | **Demo 3** — full final presentation (15%) |
| **Dec 9, 11:59 PM** | Submit GitHub repo + 4-page report |

No labs during reading week (Oct 10–18).

---

## 5 rules we never break

1. **Never look at the test set until the final demo.** Use the validation set for all decisions.
2. **Fit the scaler on the training set only.** Then just apply it to validation and test.
3. **Don't put the answer (`cardio`) in the inputs.** Drop it by name.
4. **Everyone commits their own work to GitHub.** They check who did what.
5. **Everyone must be able to explain everything.** The TA can question any of us on any part.

---

## Step 1 — Look at the data
**When:** ASAP (this week)

- [ ] Load the data into the notebook
- [ ] Count rows and columns
- [ ] Check the target `cardio`: how many 0s vs 1s?
- [ ] Check for missing values
- [ ] Check for duplicate rows
- [ ] Look for weird values (negative blood pressure, 50 cm tall people, etc.)
- [ ] Make at least 2 plots (e.g. bar chart of 0s vs 1s, histogram of age)

**Done when:** we can explain what each column means and what's wrong with the data.

<details>
<summary>What we expect to find (click to open)</summary>

| Column | Meaning |
|---|---|
| `age` | age (probably in **days**, convert to years) |
| `gender` | 1 or 2 |
| `height` / `weight` | cm / kg |
| `ap_hi` / `ap_lo` | blood pressure top / bottom number (has bad values) |
| `cholesterol` / `gluc` | 1 = normal, 2 = high, 3 = very high |
| `smoke` / `alco` / `active` | 0 or 1 |
| `cardio` | **the answer** (0 = healthy, 1 = heart disease) |

If there's an `id` column, delete it.
</details>

---

## Step 2 — Clean the data
**When:** ASAP (this week)

- [ ] Delete the `id` column
- [ ] Change age from days to years
- [ ] Remove duplicate rows
- [ ] Remove impossible rows (agree on cutoffs as a group, e.g. blood pressure too high/low, bottom number bigger than top number)
- [ ] Write down how many rows we removed and why

**Done when:** the data has no impossible values and we wrote down every cleaning choice.

---

## Step 3 — Split the data (then never change it)
**When:** ASAP (this week)

- [ ] Split into **60% train / 20% validation / 20% test**
- [ ] Keep the same 0/1 ratio in each part (use `stratify`)
- [ ] Use a fixed seed (`random_state=42`)
- [ ] Save the split so it never changes
- [ ] Make a small table showing the size of each part

**Done when:** the split is saved and frozen.

> **Train** = the model learns from this · **Validation** = we compare and tune with this · **Test** = used ONCE at the very end

---

## Step 4 — First simple models
**When:** Lab 3–4 (Oct 1 & Oct 8)

- [ ] Scale the data (StandardScaler, fit on train only)
- [ ] Train a "dumb" model that always guesses the most common answer (our starting point)
- [ ] Train **Logistic Regression**
- [ ] Train **KNN**
- [ ] Train **SVM** (if too slow, use fewer rows and say so)
- [ ] Score each one on the **validation** set

**Metrics we'll use:** accuracy, recall, precision, F1.
**Why recall matters:** missing a sick person is worse than a false alarm.

**Done when:** we have a table comparing all models (scores + training time).

---

## Step 5 — Prep Demo 1 (Oct 22)
- [ ] Notebook runs top to bottom with no errors
- [ ] Show: data, plots, cleaning, split, models, comparison table
- [ ] Write 1 thing we learned + 1 risk we still need to check
- [ ] Push to GitHub and write down the commit ID
- [ ] Practice: everyone explains every part once

---

## Step 6 — Better models + tuning
**When:** Labs 6–7 (Oct 29 & Nov 5)

- [ ] Add **Decision Tree**, **Random Forest**, **XGBoost**
- [ ] Add a **neural network (MLP)** and plot its learning curve
- [ ] **Before tuning**, write down: which settings we'll try, and what score we're trying to improve
- [ ] Tune the models using the **validation** set
- [ ] Make a table of all tuning results
- [ ] Pick the best model based on the numbers
- [ ] Save the best model + the scaler

**Done when:** one updated table with every model, and one clear winner.

---

## Step 7 — Prep Demo 2 (Nov 12)
- [ ] Same checklist as Demo 1 (runs clean, pushed, commit ID, everyone practiced)
- [ ] Be ready to explain why the winner won

---

## Step 8 — Extra analysis
**When:** Labs 9–10 (Nov 19 & Nov 26)

- [ ] **PCA** — what does it tell us about our features?
- [ ] **Clustering** (K-means) — do patients form natural groups? (optional)
- [ ] ~~CNN~~ — skip, our data isn't images
- [ ] Find at least one case where the model does badly (e.g. certain ages?)
- [ ] (Optional) Does it work equally well for men and women?

---

## Step 9 — Final test + Demo 3 (Dec 3)
- [ ] Load the saved best model
- [ ] Run it on the **test set ONE time**
- [ ] One final table comparing all models
- [ ] Add a small "quick check" cell that proves the notebook works
- [ ] Present everything through the notebook (**no slides**)
- [ ] Everyone speaks

---

## Step 10 — Submit (Dec 9, 11:59 PM)
- [ ] Restart notebook, run everything, keep outputs, delete junk cells
- [ ] Update `README.md` (how to set up and run)
- [ ] Add `requirements.txt`
- [ ] Write the **4-page report** (IEEE format) — every figure must come from the notebook
- [ ] List who did what
- [ ] Submit repo link + final commit ID

---

## Questions to ask the TA

1. Lab 3 says "regression model" but ours is classification. Do we just use Logistic Regression?
2. Some labs say "test performance" but the rules say don't touch test until the end. Should we use validation?
3. Can we remove impossible values before splitting?
4. Is it OK to train SVM on fewer rows if it's too slow?
5. Is the report worth 10% or 14%? (The sheet says both.)

---

## Who's doing what

See **[TRACKER.md](TRACKER.md)** for owners, buddies, and due dates.

---

## How we use GitHub

1. **Pull** before you start working
2. Make small changes
3. **Commit** with a clear message (e.g. `add age histogram`)
4. **Push** when done
5. Tell the group chat you pushed

Only one person edits the main notebook at a time (to avoid conflicts).
