# Deep Learning Systems Notes (Unified Shapes + DRAM / Row-Major Intuition)

Goal: unify math-style derivations (often column-vector conventions) with engineering implementations (C/C++/NumPy/PyTorch, typically “samples-as-rows”); and connect loss functions to noise assumptions, then derive Softmax-CE gradients in a shape-consistent way.

---

## 0. Notation and Shapes (Lock the conventions first)

I use the engineering (batch-first) convention:

- Single-sample feature: $x^{(i)} \in \mathbb{R}^{n}$ (in code usually a length-$n$ vector)
- Data matrix: $X \in \mathbb{R}^{m \times n}$, **each row is one sample** ($m$ samples, $n$ features)
- Multiclass parameters: $\theta \in \mathbb{R}^{n \times k}$ ($k$ classes)
- Logits:
  - Single sample: $h(x^{(i)}) \in \mathbb{R}^{k}$
  - Batch: $H = X\theta \in \mathbb{R}^{m \times k}$

### Mapping the “column-vector” tradition to the “samples-as-rows” convention

In many math derivations, a single sample is a column vector ($n \times 1$), and samples are stacked as

$\tilde X = [x^{(1)}, \dots, x^{(m)}] \in \mathbb{R}^{n \times m}$,

so logits are written as

$\theta^\top \tilde X \in \mathbb{R}^{k \times m}$.

In engineering, I typically use $X = \tilde X^\top$ (samples as rows), so:

$$
X\theta = (\tilde X^\top)\theta
\quad\Longleftrightarrow\quad
(\theta^\top \tilde X)^\top
$$

Takeaway: **these are the same up to transposes. The only real pitfall is mixing the two conventions in one derivation without tracking shapes.**

---

## 1. Why “Samples as Rows” Feels Natural in Practice (DRAM / Cache / Row-Major)

I like the cleanliness of column-vector derivations, but in practice (C/C++/NumPy/PyTorch) I often organize data as rows. The hardware/memory hierarchy intuition helps.

### 1.1 DRAM Reads an Entire Row at Once (Row Buffer)

- DRAM is organized by rows: a **wordline** connects a full row of storage cells.
- Each cell is typically a capacitor (charge encodes 0/1) + a transistor (switch).
- On read, the controller raises the wordline voltage, opening transistors; the capacitors’ signals are sensed/amplified, and **an entire row (roughly 8KB–16KB) is brought into the row buffer**.
- Subsequent reads pull smaller chunks from the row buffer up the hierarchy.

### 1.2 CPU Cache Lines (64B) + Spatial Locality

- CPU caches move data in cache-line granularity (commonly 64B).
- With row-major layout, **contiguous access along a row** tends to:
  - reduce cache misses (better 64B locality),
  - and reuse the same activated DRAM row buffer contents (8–16KB locality).
- This aligns well with the common batch-first GEMM pattern: $X\theta$.

---

## 2. Loss Functions as Maximum Likelihood Under Noise Assumptions

Model observations as:

$$
y = f(x;\theta) + \epsilon
$$

Different assumptions about the noise distribution $\epsilon$ induce different negative log-likelihoods (losses).

---

## 3. L2 / MSE: Gaussian Noise $\rightarrow$ Least Squares; Best Constant Predictor is the Mean

### 3.1 Gaussian noise $\rightarrow$ L2

If $\epsilon \sim \mathcal{N}(0,\sigma^2)$, then maximizing likelihood is equivalent to minimizing squared error:

$$
\max_\theta \log p(y|x,\theta)
\;\Longleftrightarrow\;
\min_\theta \sum_i (y_i - f(x_i;\theta))^2
$$

(up to constants/scales that do not change the optimum).

### 3.2 Predicting a constant $a$

$$
J(a) = \mathbb{E}[(Y-a)^2]
$$

Differentiate:

$$
\frac{\partial J}{\partial a} = -2\mathbb{E}[Y] + 2a = 0
\;\Rightarrow\;
a = \mathbb{E}[Y]
$$

Bias–variance decomposition (handy to remember):

$$
\mathbb{E}[(Y-a)^2]
= \underbrace{\mathbb{E}\left[(Y-\mathbb{E}[Y])^2\right]}_{\mathrm{Var}(Y)}
+ \underbrace{(\mathbb{E}[Y]-a)^2}_{\text{Bias}^2}
$$

Takeaway: under L2, the best constant predictor is the **mean**.

---

## 4. L1: Laplace Noise $\rightarrow$ Absolute Deviation; Best Constant Predictor is the Median

If $\epsilon \sim \mathrm{Laplace}(0,b)$ with density

$$
p(\epsilon) = \frac{1}{2b}\exp(-|\epsilon|/b),
$$

then the negative log-likelihood corresponds to

$$
\min_a \mathbb{E}[|Y-a|].
$$

The optimum satisfies

$$
P(Y>a) = P(Y<a),
$$

so the best constant predictor is the **median**.

---

## 5. Robust Losses: Huber and Student-t Intuition

### 5.1 Huber loss (interpolating L2 and L1)

- small residuals: quadratic (L2-like, smooth gradients)
- large residuals: linear (L1-like, less sensitive to outliers)

### 5.2 Student-t (heavy tails) as robustness

- Student-t is heavier-tailed than Gaussian, hence more robust to outliers.
- Common intuition: it can be viewed as a Gaussian with randomized variance/precision (a scale-mixture), producing heavy tails.

---

## 6. Cross Entropy: Bernoulli $\rightarrow$ Categorical / Multinomial

### 6.1 Binary classification: Bernoulli

Let $p=f_\theta(x)$. Then

$$
P(y|x,\theta) = p^y(1-p)^{1-y}
$$

and

$$
\log P(y|x,\theta) = y\log p + (1-y)\log(1-p).
$$

Negating this gives binary cross entropy.

### 6.2 Multiclass: Categorical cross entropy (one-hot)

Probability vector $p=[p_1,\dots,p_k]$ with $\sum_j p_j=1$ and one-hot label $y$:

$$
\log P(y|x,\theta) = \sum_j y_j \log p_j.
$$

Negating gives CCE.

### 6.3 Multinomial (counts over $n$ trials)

$$
P(n_1,\dots,n_k) = \frac{n!}{\prod_i n_i!}\prod_i p_i^{n_i}.
$$

When $n=1$, this reduces to the categorical case (single-sample).

---

## 7. Softmax + Cross Entropy: Key Identities and Gradients (Single Sample $\rightarrow$ Batch)

### 7.1 Softmax

$$
z_i = \frac{e^{h_i}}{\sum_j e^{h_j}},
\qquad
\sum_i z_i = 1.
$$

### 7.2 Single-sample cross entropy (label $y$)

$$
\ell(h,y) = -\log z_y = -h_y + \log\sum_j e^{h_j}.
$$

Gradient w.r.t. logits:

$$
\frac{\partial \ell}{\partial h_i} = z_i - \mathbf{1}[i=y].
$$

Equivalently:

$$
\delta = z - \mathrm{onehot}(y).
$$

### 7.3 Batch-form final gradient (the main result)

Let:

- $H = X\theta \in \mathbb{R}^{m\times k}$
- $Z = \mathrm{softmax}(H) \in \mathbb{R}^{m\times k}$ (row-wise softmax)
- $Y \in \mathbb{R}^{m\times k}$ (one-hot labels)
- $\Delta = Z - Y$

Then:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = X^\top (Z-Y) = X^\top \Delta.
$$

Implementation-wise: compute $\Delta$ as “predicted probs minus one-hot”, then multiply by $X^\top$.

---

## 8. Matrix Differentials / Trace Trick Cheat Sheet (for mechanical derivations)

Three core identities:

1. Differential and trace form:

$$
df = \mathrm{tr}\!\left(\left(\frac{\partial f}{\partial X}\right)^\top dX\right)
$$

2. Product rule:

$$
d(AB) = dA\cdot B + A\cdot dB
$$

3. Trace cyclic permutation:

$$
\mathrm{tr}(ABC)=\mathrm{tr}(BCA)=\mathrm{tr}(CAB)
$$

Typical template (derive $\partial \ell / \partial \theta$):

- Start with

$$
d\ell = \mathrm{tr}(\Delta^\top dH)
$$

- Use

$$
dH = d(X\theta) = X\, d\theta
$$

- Combine and cycle the trace:

$$
\begin{aligned}
d\ell
&= \mathrm{tr}(\Delta^\top X\, d\theta) \\
&= \mathrm{tr}\!\left((X^\top \Delta)^\top d\theta\right)
\end{aligned}
$$

Therefore:

$$
\frac{\partial \ell}{\partial \theta} = X^\top \Delta
$$

---

## 9. One-page summary

- Column-vector derivations vs samples-as-rows implementations are not contradictory: they differ mainly by transposes, as long as shapes are tracked consistently.
- Batch-first “samples as rows” matches row-major spatial locality and aligns with cache-line and DRAM row-buffer behavior.
- Loss functions correspond to noise assumptions: Gaussian $\rightarrow$ L2 (mean), Laplace $\rightarrow$ L1 (median), Huber / Student-t $\rightarrow$ robust behavior.
- For Softmax + CE: the key intermediate is $\Delta = Z - Y$, and the parameter gradient is $X^\top\Delta$.
