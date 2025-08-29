
# Probability & Statistics for Machine Learning

> A deeply practical, interview-ready refresher that expands the corresponding chapter in your PDF and maps cleanly to your repo’s structure. Examples use NumPy/SciPy/StatsModels/scikit-learn; copy/paste them into notebooks as needed.

---

## Table of Contents

1. Foundations of Probability for ML  
2. Random Variables & Core Distributions  
3. Expectations, Variance, Inequalities  
4. Sampling & Asymptotics (LLN, CLT)  
5. Likelihood, Cross-Entropy & KL (and why classification uses log-loss)  
6. Estimation: CIs for Means & Proportions (Wilson, Clopper–Pearson)  
7. Resampling: Bootstrap CIs & Permutation Tests  
8. Hypothesis Testing in A/B Experiments (errors, power, multiple testing)  
9. Bayesian Thinking in Practice (Beta–Binomial AB testing)  
10. Evaluating Classifiers: ROC/PR, Calibration & Brier Score  
11. Detecting Dataset Shift (KS test, PSI)  
12. Common Interview Traps & Best Practices  
13. Appendix: Expanded Code Patterns

---

## 1) Foundations of Probability for ML

**Conditional probability & Law of Total Probability (LTP).** We often decompose complex prediction tasks into simpler conditional pieces. LTP connects marginals to conditionals:
\[
P(A)=\sum_i P(A\mid B_i)\,P(B_i)
\]
for a partition \(\{B_i\}\). This is how we combine per-segment conversion rates into an overall rate, or marginalize latent variables. citeturn1search2

**Bayes’ rule.** In classification and filtering, we invert conditionals (“likelihoods”) into posteriors:
\[
P(H\!\mid D) \propto P(D\!\mid H)\,P(H)
\]
This underpins Naive Bayes, spam filtering, and medical diagnosis—any time you update beliefs with new evidence. citeturn1search9

**(Conditional) independence.** Conditional independence assumptions let us factor high‑dimensional likelihoods; e.g., Naive Bayes assumes features are conditionally independent given the class. Understanding when that’s approximately true is crucial for model bias/variance trade‑offs. citeturn1search4turn1search10

---

## 2) Random Variables & Core Distributions

- **Bernoulli/Binomial**: binary outcomes and counts of successes in \(n\) IID trials with success prob \(p\).
- **Poisson**: counts of rare events in a fixed interval; mean equals variance \(\lambda\). citeturn3search2  
- **Exponential**: inter‑arrival times with the *memoryless* property; the only continuous memoryless distribution. citeturn3search1

**Law of rare events (sketch).** A Binomial(\(n,p\)) with \(n\to\infty,\,p\to 0\), \(np=\lambda\) tends to Poisson(\(\lambda\)), explaining why Poisson fits rare‑event phenomena. citeturn3search2

---

## 3) Expectations, Variance, Inequalities (engineer’s view)

- **Expectation** \(\mathbb E[X]\) is linear; **variance** scales quadratically with constants.  
- **Chebyshev’s inequality**: for any \(X\) with finite variance, \(\Pr(|X-\mu|\ge k\sigma)\le 1/k^2\). Use when you want distribution‑free tail bounds. citeturn4search11

---

## 4) Sampling & Asymptotics (Why “averaging works”)

- **Law of Large Numbers (LLN):** sample averages converge to the true mean as \(n\) grows—why empirical rates stabilize with more data. citeturn4search0  
- **Central Limit Theorem (CLT):** the standardized mean is approximately Normal (under mild conditions), enabling z/t intervals and tests even when the population isn’t Normal. citeturn1search13

> LLN says *where* the estimator goes; CLT says *how it fluctuates* for finite \(n\).

---

## 5) Likelihood, Cross‑Entropy & KL (why log‑loss rules)

For a classifier with predicted distribution \(q_\theta(y\mid x)\), supervised training often minimizes **cross‑entropy** between empirical labels \(p\) and predictions \(q\):
\[
H(p,q)= -\mathbb E_{p}\,[\log q]
\]
Since \(H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)\), minimizing cross‑entropy with fixed \(p\) is equivalent to minimizing **KL divergence** from truth to model. In one‑hot classification, cross‑entropy reduces to negative log‑likelihood (MLE). citeturn6search0turn6search2

**Engineering note (stability).** Compute softmax/log‑loss with the **log‑sum‑exp** trick (subtract max‑logit before exponentiating) to avoid overflow/underflow. citeturn6search8

---

## 6) Estimation: Confidence Intervals (CIs)

### 6.1 Mean difference (Welch’s t)

When groups have unequal variances/sizes, Welch’s t‑procedures are robust:
```python
import numpy as np
from scipy import stats

x, y = np.array([...]), np.array([...])
t, p = stats.ttest_ind(x, y, equal_var=False)  # Welch test
# 95% CI for mean difference via Welch SE + DOF
nx, ny = len(x), len(y)
vx, vy = x.var(ddof=1), y.var(ddof=1)
se = np.sqrt(vx/nx + vy/ny)
df = (vx/nx + vy/ny)**2 / ((vx**2/((nx**2)*(nx-1))) + (vy**2/((ny**2)*(ny-1))))
delta = stats.t.ppf(0.975, df) * se
ci = (x.mean()-y.mean()-delta, x.mean()-y.mean()+delta)
```
SciPy documents `ttest_ind` and the Welch option. citeturn5search0

### 6.2 Proportions (robust choices > Wald)

The classic Wald CI \( \hat p \pm z\sqrt{\hat p(1-\hat p)/n} \) is fragile for small \(n\) or extreme \(\hat p\). Prefer **Wilson** (score), **Clopper–Pearson** (exact), or **Jeffreys** intervals—all available in StatsModels:
```python
from statsmodels.stats.proportion import proportion_confint
count, n = 37, 200
ci_wilson = proportion_confint(count, n, method="wilson")
ci_exact  = proportion_confint(count, n, method="beta")      # Clopper–Pearson
ci_jeff   = proportion_confint(count, n, method="jeffreys")
```
The API explains coverage properties and trade‑offs. citeturn0search1

---

## 7) Resampling: Bootstrap CIs & Permutation Tests

When analytic variance is messy (e.g., medians or complex metrics), use **bootstrap** CIs:
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
sample = rng.normal(loc=0.0, scale=1.0, size=500)
res = stats.bootstrap((sample,), np.mean, n_resamples=10_000, method="BCa")
res.confidence_interval  # low/high
```
SciPy’s `bootstrap` implements percentile and BCa intervals. citeturn0search0

**Permutation tests** give exact or simulation‑based p‑values with minimal assumptions—especially useful in A/B tests with skew/heavy tails:
```python
from scipy import stats

def diff_mean(a, b, axis):
    return a.mean(axis=axis) - b.mean(axis=axis)

perm = stats.permutation_test((x, y), diff_mean,
                              n_resamples=50_000,
                              alternative="two-sided",
                              random_state=0)
perm.pvalue
```
citeturn0search4

---

## 8) Hypothesis Testing in A/B Experiments

**Errors & Power.** \(\alpha\) is false‑positive rate; power \(=1-\beta\) is the chance to detect a true effect. Use StatsModels to plan sample size for a target effect and desired power:
```python
from statsmodels.stats.power import TTestIndPower, NormalIndPower

# t-test (means) and z-test (proportions)
t_calc = TTestIndPower().solve_power(effect_size=0.3, power=0.8, alpha=0.05)
z_calc = NormalIndPower().solve_power(effect_size=0.2, power=0.8, alpha=0.05)
```
citeturn5search1turn5search7

**Multiple testing.** If you run many experiments/segments/metrics, control **false discovery rate** (FDR). StatsModels and SciPy implement **Benjamini–Hochberg** (BH) and **Benjamini–Yekutieli** (BY):
```python
import numpy as np
from statsmodels.stats.multitest import fdrcorrection

pvals = np.array([0.03, 0.20, 0.002, 0.08, 0.049])
reject, pvals_adj = fdrcorrection(pvals, alpha=0.05, method="indep")  # BH
```
citeturn2search1turn2search13

---

## 9) Bayesian Thinking in Practice (fast & useful)

**Beta–Binomial A/B.** With a Beta prior \(\text{Beta}(\alpha,\beta)\) for rate \(p\) and data \(k/n\), the posterior is \(\text{Beta}(\alpha+k,\beta+n-k)\). This yields credible intervals and easy answers like \(P(p_B>p_A)\):
```python
import numpy as np
from scipy.stats import beta

# Prior: Beta(1,1) ~ Uniform; Data: A: 37/200, B: 52/200
postA = beta(1+37, 1+(200-37))
postB = beta(1+52, 1+(200-52))

ciA = postA.interval(0.95)
ciB = postB.interval(0.95)

rng = np.random.default_rng(0)
pa = postA.rvs(200_000, random_state=rng)
pb = postB.rvs(200_000, random_state=rng)
prob_b_beats_a = (pb > pa).mean()
prob_b_beats_a
```
(Conceptual grounding follows from Bayes + LTP above.)
 
---

## 10) Evaluating Classifiers: ROC/PR, Calibration & Brier

**ROC AUC** summarizes discrimination across thresholds; great for rank‑based evaluation. **PR curves** are often more informative under class imbalance.
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

auc = roc_auc_score(y_true, y_scores)
precision, recall, thr = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)  # PR AUC proxy
```
citeturn0search3turn5search2

**Calibration & reliability diagrams.** A model can rank well yet be poorly calibrated. Plot predicted vs. empirical probabilities using `calibration_curve`; fix with isotonic or Platt scaling.
```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
```
citeturn0search2

**Brier Score** is a strictly proper scoring rule for probabilistic predictions—mean squared error of probabilities (lower is better).  
```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_true, y_prob)
```
citeturn2search0

---

## 11) Detecting Dataset Shift (before your model drifts)

**Two‑sample KS test** compares continuous distributions \(F\) vs \(G\) without parametric assumptions—handy for spotting covariate drift between train and serve:
```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(feature_train, feature_live)
```
Small \(p\) suggests shift. citeturn2search2

**Population Stability Index (PSI)** quantifies covariate shift via bucketed divergence; common monitoring thresholds: \(\approx 0.1\) moderate, \(\approx 0.2\) severe (rules of thumb vary by org). citeturn2search15

```python
import numpy as np

def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0,1,bins+1))
    ex, _ = np.histogram(expected, bins=qs)
    ac, _ = np.histogram(actual,   bins=qs)
    ex = np.where(ex==0, 1, ex); ac = np.where(ac==0, 1, ac)
    ex_pct, ac_pct = ex/ex.sum(), ac/ac.sum()
    return np.sum((ac_pct - ex_pct) * np.log(ac_pct / ex_pct))

psi_val = psi(train_feature, live_feature)
```

---

## 12) Common Interview Traps & Best Practices

1. **Don’t trust Wald CIs for proportions.** Prefer Wilson/Clopper–Pearson in small‑n/extreme‑p regimes. citeturn0search1  
2. **ROC vs PR.** On imbalanced data, PR is often more informative than ROC for selection decisions. citeturn5search8  
3. **Welch over Student’s t** with unequal variances; when in doubt, permutation test your chosen statistic. citeturn5search0turn0search4  
4. **Power before you test.** Compute minimal detectable effect (MDE) & \(n\) up‑front. citeturn5search7  
5. **Monitor drift.** Add KS and PSI checks to your pipeline; alert on sustained deviations. citeturn2search2turn2search15  
6. **Calibrate if you need probabilities.** Add reliability diagrams & Brier score to evaluation. citeturn0search2turn2search0

---

## 13) Appendix: Expanded Code Patterns

### A) Wilson vs. Wald CI comparison (simulation)
```python
import numpy as np
from statsmodels.stats.proportion import proportion_confint

def coverage(n=50, p=0.05, reps=20000, alpha=0.05):
    rng = np.random.default_rng(0)
    wald_hits = wilson_hits = 0
    z = 1.959963984540054
    for _ in range(reps):
        x = rng.binomial(n, p)
        phat = x/n
        # Wald
        se = (phat*(1-phat)/n)**0.5
        lo_wald, hi_wald = phat - z*se, phat + z*se
        wald_hits += (lo_wald <= p <= hi_wald)
        # Wilson
        lo_wil, hi_wil = proportion_confint(x, n, alpha=alpha, method='wilson')
        wilson_hits += (lo_wil <= p <= hi_wil)
    return wald_hits/reps, wilson_hits/reps

coverage()  # Wald under‑covers when p is small; Wilson closer to 95%
```
citeturn0search1

### B) Calibrating a classifier & plotting reliability
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=5000, n_features=20, weights=[0.85,0.15], random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
p_raw = clf.predict_proba(Xte)[:,1]

cal = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv=3).fit(Xtr, ytr)
p_cal = cal.predict_proba(Xte)[:,1]

for p, lbl in [(p_raw,"raw"),(p_cal,"isotonic")]:
    prob_true, prob_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    plt.plot(prob_pred, prob_true, marker='o', label=f"{lbl} (Brier={brier_score_loss(yte,p):.3f})")

plt.plot([0,1],[0,1],'--',alpha=.5,label='perfect')
plt.xlabel("Predicted probability"); plt.ylabel("Empirical frequency"); plt.legend(); plt.tight_layout()
```
citeturn0search2turn2search0

### C) Monitoring shift with KS & PSI
```python
import numpy as np
from scipy.stats import ks_2samp

def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0,1,bins+1))
    ex, _ = np.histogram(expected, bins=qs)
    ac, _ = np.histogram(actual,   bins=qs)
    ex = np.where(ex==0, 1, ex); ac = np.where(ac==0, 1, ac)
    ex_pct, ac_pct = ex/ex.sum(), ac/ac.sum()
    return np.sum((ac_pct - ex_pct) * np.log(ac_pct / ex_pct))

rng = np.random.default_rng(0)
train = rng.normal(0,1, 4000)
live  = rng.normal(0.2,1.1, 4000)

ks_stat, ks_p = ks_2samp(train, live)
psi_val = psi(train, live)
ks_stat, ks_p, psi_val
```
citeturn2search2

---

### Notes on numerical stability & reproducibility

- Prefer `solve`/`lstsq`/`pinv` over explicit inverses.
- Use log‑sum‑exp for softmax/log‑loss.
- Set seeds for reproducibility when simulating or bootstrapping.

---

**Suggested filename:** `probability_statistics_for_ml.md`  
**Suggested repo location:** `samples/Week4/` (to mirror your current layout, without using calendar terms in the title).
