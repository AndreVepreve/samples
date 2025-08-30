
# Mathematics for Machine Learning — Comprehensive Notes

> A deep-dive, exam-ready reference that unfolds the “Mathematics for Machine Learning” chapter into detailed explanations, derivations, examples, NumPy snippets, and practical tips. Focus areas are **Linear Algebra Essentials** and **Norms, Inner Products, and Inequalities**, followed by concise but complete definitions of **Eigenvalues/Eigenvectors**, **SVD**, **Pseudoinverse**, **Jacobian**, **Hessian**, and related shapes you’ll use every day as an AI engineer.

---

## Table of Contents

1. Linear Algebra Essentials  
   1.1 Vectors, matrices, tensors, and shapes  
   1.2 Matrix operations (add, scalar, matmul)  
   1.3 Special matrices (identity, diagonal, symmetric, orthogonal)  
   1.4 Rank, range, null space, linear independence  
   1.5 Determinant and invertibility  
   1.6 Numerical issues: conditioning, scaling
2. Norms, Inner Products, and Inequalities  
   2.1 Vector norms (ℓ₀, ℓ₁, ℓ₂, ℓ∞) and when to use them  
   2.2 Matrix norms (Frobenius, spectral)  
   2.3 Inner products, cosine similarity  
   2.4 Cauchy–Schwarz, triangle inequality, Hölder & Minkowski
3. Eigenvalues & Eigenvectors — definition, intuition, uses
4. Singular Value Decomposition (SVD) — definition, properties, low-rank
5. Moore–Penrose Pseudoinverse — definition, least squares, SVD link
6. Matrix Calculus Cheat Sheet — Jacobian, gradient, Hessian (with shapes)
7. Worked Example — Linear regression: closed form, gradient descent, ridge
8. Quick Reference — “what to use when” for interviews & projects

---

## 1) Linear Algebra Essentials

### 1.1 Vectors, matrices, tensors, and shapes

- **Vector**: ordered list of numbers, an element of $\mathbb{R}^n$. Shape: `(n,)` or `(n,1)`.
- **Matrix**: rectangular array in $\mathbb{R}^{m\times n}$. Acts as a linear map $A: \mathbb{R}^n \to \mathbb{R}^m$.  
  ![](./assets/f_linear_map.png)
- **Tensor**: multi-dimensional array; in ML, usually order-3+ arrays (e.g., images as `H×W×C`).

**NumPy toy example**
```python
import numpy as np
x = np.array([1., 2., 3.])         # shape (3,)
A = np.array([[1., 0., -1.],
              [2., 1.,  0.]])      # shape (2,3)
y = A @ x                          # shape (2,)
```

---

### 1.2 Matrix operations (add, scalar, matmul)

- **Addition / Scalar**: Element-wise, same shape requirement.  
- **Matrix multiplication**: `(m×n)·(n×p) -> (m×p)`. Not commutative but associative.  
  Element formula:  
  ![](./assets/f_matmul_element.png)

**Dot product (vector)**: $\mathbf{a}^\top \mathbf{b}$ equals the Euclidean inner product; $\|\mathbf{a}\|_2^2 = \mathbf{a}^\top\mathbf{a}$.

---

### 1.3 Special matrices

- **Identity $I$**: $I\mathbf{x}=\mathbf{x}$.
- **Diagonal $D$**: $D_{ij}=0$ for $i\neq j$; easy to invert if diagonal entries nonzero.
- **Symmetric $A=A^\top$**: real eigenvalues; orthogonal eigenvectors.
- **Orthogonal $Q^\top Q=I$**: length-preserving transforms (rotations/reflections).  
  ![](./assets/f_orthogonal_preserve.png)

**Why you care**: Orthogonal/Unitary transforms (Fourier, PCA bases) preserve energy; helpful for stable optimization and conditioning.

---

### 1.4 Rank, range, null space, linear independence

- **Rank**: dimension of the column space (number of independent columns).  
- **Range (column space)**: all vectors $A\mathbf{x}$ you can reach.  
- **Null space**: all vectors $\mathbf{x}$ with $A\mathbf{x}=\mathbf{0}$.  
- **Rank–nullity** (for $A\in\mathbb{R}^{m\times n}$):  
  ![](./assets/f_rank_nullity.png)

**ML motivation**: Rank tells you if features are redundant; low rank ⇒ compressible data ⇒ PCA/low-rank models.

---

### 1.5 Determinant and invertibility

- **Determinant** measures volume scaling for square $A$. $\det(A)=0$ ⇒ singular (not invertible).  
- **Invertibility**: $A^{-1}$ satisfies $AA^{-1}=I$. Equivalent condition:  
  ![](./assets/f_invertible_det.png)

**Geometric picture**: In $\mathbb{R}^2$, $|\det(A)|$ is the area scale factor of a transformed unit square.

---

### 1.6 Numerical issues: conditioning & scaling

- **Condition number** gauges sensitivity. Large $\kappa$ ⇒ tiny data noise causes large solution errors.  
  ![](./assets/f_condition_number.png)
- **Best practice**: center/standardize features; avoid explicitly computing matrix inverses; prefer factorizations (QR/SVD/Cholesky).

---

## 2) Norms, Inner Products, and Inequalities

### 2.1 Vector norms (how we measure size)

A **norm** $\|\cdot\|$ maps vectors to nonnegative scalars and satisfies positivity, homogeneity, and triangle inequality.

Common vector norms:

- **$\ell_2$ (Euclidean)**: $\|\mathbf{x}\|_2=\sqrt{\sum_i x_i^2}$. Smooth; rotationally invariant.  
- **$\ell_1$**: $\|\mathbf{x}\|_1=\sum_i |x_i|$. Promotes sparsity (Lasso).  
- **$\ell_\infty$**: $\|\mathbf{x}\|_\infty=\max_i |x_i|$. Useful for worst-case bounds.  
- **$\ell_0$** (not a true norm): counts nonzeros; used conceptually in sparsity.

General $p$-norm:  
![](./assets/f_p_norm.png)

**Choosing norms in ML**  
- Use $\ell_2$ in smooth optimization & Gaussian noise models.  
- Use $\ell_1$ when you want feature selection (sparsity).  
- Use $\ell_\infty$ for robust, adversarial, or max-error constraints.

---

### 2.2 Matrix norms

- **Frobenius**: element-wise energy.  
- **Spectral / operator (induced by $\ell_2$)**: largest singular value.  
  ![](./assets/f_matrix_norms.png)

**Tips**: Use $\|\cdot\|_F$ for easy, element-wise energy; use $\|\cdot\|_2$ when operator amplification matters (stability).

---

### 2.3 Inner products and cosine similarity

Angle-based similarity used in embeddings:  
![](./assets/f_cosine_sim.png)

---

### 2.4 Inequalities you’ll actually use

- **Cauchy–Schwarz**:  
  
```
|\mathbf{a}^\top\mathbf{b}|\le \|\mathbf{a}\|_2\|\mathbf{b}\|_2
```

- **Triangle inequality**: $\|\mathbf{a}+\mathbf{b}\|\le \|\mathbf{a}\|+\|\mathbf{b}\|$.  
- **Hölder**: $\sum_i |a_i b_i| \le \|\mathbf{a}\|_p\|\mathbf{b}\|_q$ with $1/p+1/q=1$.  
- **Minkowski**: $\|\mathbf{a}+\mathbf{b}\|_p \le \|\mathbf{a}\|_p+\|\mathbf{b}\|_p$.

---

## 3) Eigenvalues & Eigenvectors (definition, purpose, intuition)

Definition:  
![](./assets/f_eig.png)

Why useful: spectral theorem (symmetric matrices), PCA, graph Laplacians, power iteration.

---

## 4) Singular Value Decomposition (SVD)

Works for any $m\times n$ matrix:  
![](./assets/f_svd.png) with descending singular values.

Low-rank approximation (Eckart–Young):  
![](./assets/f_eckart_young.png)

PCA via SVD: principal directions in $V$, variances in $\Sigma^2/(n-1)$.

---

## 5) Moore–Penrose Pseudoinverse

Via SVD:  
![](./assets/f_pinv.png)

Least squares: $\hat{\mathbf{x}}=A^+\mathbf{b}$ for over/under-determined systems.

---

## 6) Matrix Calculus Cheat Sheet — Jacobian, Gradient, Hessian (with shapes)

- Jacobian:  
  ![](./assets/f_jacobian.png)
- Hessian:  
  ![](./assets/f_hessian.png)

Chain rule:  
![](./assets/f_chain_rule.png)

Least-squares gradient (and $H=A^\top A$):  
![](./assets/f_ls_grad.png)

---

## 7) Worked Example — Linear Regression (closed form, GD, ridge)

Objective, gradients, and ridge closed form:  
![](./assets/f_ridge_closed.png)

**NumPy template**
```python
import numpy as np

n, d = 100, 5
X = np.random.randn(n, d)
true_w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
y = X @ true_w + 0.1*np.random.randn(n)

# Closed form via solve (more stable than explicit inverse)
w_hat = np.linalg.solve(X.T @ X, X.T @ y)

# Ridge
lam = 1e-2
w_ridge = np.linalg.solve(X.T @ X + n*lam*np.eye(d), X.T @ y)

# Gradient Descent
w = np.zeros(d)
eta = 1e-2
for _ in range(2000):
    grad = (X.T @ (X @ w - y)) / n
    w -= eta * grad
```

---

## 8) Quick definitions & interview-ready notes

## 8) Quick Definitions — Expanded & Interview‑Ready

### Eigenvalues & Eigenvectors
![](./assets/f_eig.png)

**What they are.** For a square matrix A, nonzero vector v and scalar λ satisfying A v = λ v are an **eigenvector/eigenvalue** pair. Eigenvectors are directions that are only **scaled** (not rotated) by the linear map; λ is the scale (possibly negative).

**Why they matter in ML.**
- **PCA**: eigenvectors of the covariance matrix give principal directions; eigenvalues give explained variance per direction.
- **Stability & dynamics**: the sign/magnitude of eigenvalues of Jacobians/Hessians or update operators hints at stability/divergence.
- **Graph learning**: eigenpairs of graph Laplacians encode cuts, diffusion, and community structure.
- **Optimization geometry**: eigenvalues of the Hessian quantify curvature; small eigenvalues → flat directions, large → steep directions.

**Key properties.**
- Symmetric (real) matrices have **real eigenvalues** and **orthonormal eigenvectors** (spectral theorem).
- The **trace** equals the sum of eigenvalues; **determinant** equals the product of eigenvalues.
- The **dominant eigenvector** can be approximated efficiently via **power iteration**.

**Pitfalls.** For non‑symmetric matrices, eigenvectors may be non‑orthogonal or complex; use SVD for robust geometry.

---

### Singular Value Decomposition (SVD)
![](./assets/f_svd.png)   (Best rank‑k approx: ![](./assets/f_eckart_young.png))

**What it is.** Any A ∈ ℝ^{m×n} factors as A = U Σ V^T with orthonormal U,V and nonnegative singular values in Σ. It is the most **numerically stable** way to study a linear map’s action.

**Why it’s ubiquitous.**
- **Compression / low‑rank**: truncate to k singular values to get the best rank‑k approximation in Frobenius norm.
- **PCA via SVD**: for zero‑mean data matrix X, right singular vectors (rows of V^T) are principal axes; Σ^2/(n−1) gives variances.
- **Conditioning**: κ₂(A) = σ_max/σ_min (see ![](./assets/f_condition_number.png)).

**When to use it.** Ill‑conditioned least squares, dimensionality reduction, denoising, initialization, and diagnostics.

**Cost.** Dense SVD is O(m n min{m,n}); use randomized/iterative SVD for large sparse matrices.

---

### Moore–Penrose Pseudoinverse
![](./assets/f_pinv.png)

**What it is.** For any matrix A, the **pseudoinverse** A^+ provides the minimum‑norm solution to least squares, even when A is rectangular or rank‑deficient.

**Core use (least squares).** Solve min_x ||A x − b||₂² via x* = A^+ b. This equals the limit of ridge solutions as λ → 0⁺, and is given stably by the SVD (invert nonzero singular values).

**Why not invert?** Explicit (A^T A)^{-1} A^T is numerically fragile; A^+ via SVD handles rank deficiency gracefully.

**Interview tips.** State the four Penrose conditions if asked for formality; in practice, say “compute via SVD, zero‑out tiny σ_i.”

---

### Jacobian (for vector outputs) and Hessian (for scalar objectives)
![](./assets/f_jacobian.png)   ![](./assets/f_hessian.png)   Chain rule: ![](./assets/f_chain_rule.png)

**Jacobian.** For f: ℝⁿ→ℝᵐ, the **Jacobian** J_f ∈ ℝ^{m×n} collects first derivatives and is the best **linear approximation** of f near a point. In deep learning, backprop is repeated Jacobian‑vector (and vector‑Jacobian) products.

**Hessian.** For a scalar g: ℝⁿ→ℝ, the **Hessian** H_g ∈ ℝ^{n×n} contains second derivatives (curvature). In optimization:
- **Newton/Quasi‑Newton** methods use H^{-1} ∇g (or approximations) for fast convergence.
- **Convexity**: H ⪰ 0 locally implies convex behavior; eigenvalues of H quantify anisotropy/ill‑conditioning.

**Shapes & practice.**
- If f(x)=A x, then J_f = A; for least squares ½||A x − b||₂², ∇ = A^T(Ax − b) and H = A^T A (see ![](./assets/f_ls_grad.png)).
- Exact Hessians are expensive in high‑D; use **Hessian‑vector products** (automatic differentiation) for CG/Trust‑Region methods.

---

### Sources & Further Reading
- Gilbert Strang — *Linear Algebra and Learning from Data*  
- Boyd & Vandenberghe — *Convex Optimization* (free PDF)  
- Goodfellow, Bengio, Courville — *Deep Learning* (MIT Press, free online)  
- Murphy — *Probabilistic Machine Learning*  
- Petersen & Pedersen — *The Matrix Cookbook* (free PDF)

---

## Practical Examples (Hands-on)

### Matrix multiplication & associativity
Let
```
A = [[1, 2],
     [0, 1]]
B = [[ 2, 0],
     [-1, 3]]
C = [[1, 1],
     [4, 0]]
```
Then
```
A @ (B @ C) = [[24, 0], [11, -1]]
(A @ B) @ C = [[24, 0], [11, -1]]
```
They are equal, illustrating associativity.

### Orthogonal matrix (rotation)
45° rotation matrix \(Q\) satisfies \(Q^\top Q \approx I\):
```
Q^T Q ≈
[[1.0, 0.0], [0.0, 1.0]]
```

### Rank & null space
For
```
M = [[1, 2, 3],
     [2, 4, 6]]
```
```
rank(M) = 1
nullspace basis ≈ [[-2, 1, 0], [-3, 0, 1]]
```
(check: each basis vector v satisfies M @ v = 0).

### Determinant & area scaling
For
```
A = [[2, 0],
     [1, 3]]
det(A) = 6.0
```
A unit square’s area scales by |det(A)| = 6.0.

### Ill-conditioning demo
Let
```
A = [[1, 1],
     [1, 1+ε]],  ε = 0.0001
cond_2(A) ≈ 4.0e+04
```
Two nearby right-hand sides give very different solutions:
```
x(b=[2, 2+ε]) = [1.0, 1.0]
x(b=[2, 2-ε]) = [3.0, -1.0]
Δx = [2.0, -2.0]
```

### Vector norms
For v = [3.0, -4.0, 0.0, 12.0]:
```
||v||_1 = 19.0
||v||_2 = 13.0000
||v||_∞ = 12.0
```

### Cosine similarity
For a = [1.0, 2.0, 3.0], b = [-1.0, 0.0, 1.0]:
```
cosθ = 0.377964
Cauchy–Schwarz: |a·b| = 2.000000 ≤ ||a||·||b|| = 5.291503
Triangle: ||a+b|| = 4.472136 ≤ ||a|| + ||b|| = 5.155871
```

### Eigenvalues & eigenvectors (symmetric)
For
```
A = [[2, 1],
     [1, 2]]
```
```
eigenvalues = [3.0, 1.0]
eigenvectors (columns) ≈
[[0.707107, -0.707107], [0.707107, 0.707107]]
```

### SVD (tiny 2×2)
For
```
T = [[3, 1],
     [1, 3]]
```
```
U ≈ [[-0.707107, -0.707107], [-0.707107, 0.707107]]
S ≈ [4.0, 2.0]
V^T ≈ [[-0.707107, -0.707107], [-0.707107, 0.707107]]
```

### Pseudoinverse for least squares
Solve min_x ||Ax - b||_2 with
```
A = [[1,0],
     [1,1],
     [1,2]],  b = [1, 2, 2]
x* = [1.166667, 0.5]
MSE = 0.055556
```

### Jacobian & Hessian at (x,y)=(1,2)
For f(x,y) = [x^2, x y] and g(x,y) = x^2 + x y + y^2:
```
J_f(1,2) =
[[2.0, 0.0], [2.0, 1.0]]
∇g(1,2) = [4.0, 5.0]
H_g = 
[[2.0, 1.0], [1.0, 2.0]]
```

---

## Sources & Links (Official/Authoritative)

- Boyd & Vandenberghe — *Convex Optimization* (free PDF, Stanford) — https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
- Convex Optimization homepage (slides, extras) — https://stanford.edu/~boyd/cvxbook/
- Petersen & Pedersen — *The Matrix Cookbook* (DTU) — https://www2.compute.dtu.dk/pubdb/pubs/3274-full.html
- The Matrix Cookbook (Waterloo mirror PDF) — https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
- Goodfellow, Bengio, Courville — *Deep Learning* (free online) — https://www.deeplearningbook.org/
- MIT OCW 18.06 Linear Algebra (Strang) — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- MIT OCW 18.06SC (self-paced) — https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
- Murphy — *Probabilistic Machine Learning* (series site) — https://probml.github.io/pml-book/
- Murphy — *Probabilistic Machine Learning: An Introduction* (MIT Press) — https://mitpress.mit.edu/9780262046824/probabilistic-machine-learning/

---

## Glossary — Terms and Abbreviations in the Quick Notes (Fully Explained)

### Axes (Principal Axes)
**Definition.** Coordinate directions used to describe data or a linear map. In PCA or spectral analyses, **principal axes** are orthonormal directions along which data variance is extremal.  
**Purpose.** Provide an interpretable basis to project, visualize, compress, and denoise data.  
**Details.** The principal axes are the eigenvectors of the covariance matrix (or right singular vectors from SVD of the centered data matrix). Projections onto the first few axes retain most variance.

### Scaling (in Eigenanalysis)
**Definition.** The multiplicative change of a vector in a specific direction under a linear map. If \(A\mathbf{v}=\lambda \mathbf{v}\), the factor \(\lambda\) is the **scaling** along eigenvector \(\mathbf{v}\).  
**Purpose.** Explains how transformations stretch or contract space; used in stability, conditioning, and mode analysis.  
**Details.** Negative \(\lambda\) flips direction; \(|\lambda|>1\) expands, \(|\lambda|<1\) contracts.

### PCA (Principal Component Analysis)
**Definition.** Orthogonal linear transform that rotates data to a basis of maximal variance directions.  
**Purpose.** **Dimensionality reduction**, **compression**, **denoising**, and **visualization**.  
**Details.** For centered data \(X\), SVD \(X=U\Sigma V^\top\) yields principal components (columns of \(V\)); variances are \(\Sigma^2/(n-1)\). The first \(k\) components explain the largest portion of variance.

### Stability (Optimization / Dynamics / Linear Systems)
**Definition.** A property describing whether iterates or trajectories remain bounded or converge (vs. diverge).  
**Purpose.** Guarantees predictable training and robust systems.  
**Details.** In gradient descent on a quadratic with Hessian \(H\), step size \(\eta\) must satisfy \(0<\eta<2/\lambda_{\max}(H)\) to ensure stability. In discrete linear systems \(x_{t+1}=Ax_t\), stability requires spectral radius \(\rho(A)<1\).

### Universal (re: SVD as a universal factorization)
**Definition.** SVD exists for **any** real (or complex) matrix; no symmetry, squareness, or rank assumptions needed.  
**Purpose.** A single, robust tool to analyze geometry (directions and gains) of any linear map.  
**Details.** \(A=U\Sigma V^\top\) with orthonormal \(U,V\) and nonnegative singular values in \(\Sigma\). Right singular vectors (columns of \(V\)) form an orthonormal basis of input space; left singular vectors (\(U\)) of output space.

### (Numerically) Stable
**Definition.** An algorithm is numerically stable if small input/rounding errors do not blow up in the output.  
**Purpose.** Reliability in floating‑point computations.  
**Details.** SVD/QR‑based solvers are stable; explicitly computing \((X^\top X)^{-1}\) is **unstable** when \(X\) is ill‑conditioned (use SVD/QR instead).

### Compression (Low‑Rank Compression)
**Definition.** Approximating a matrix (or dataset) by a lower‑rank representation retaining most of its “energy/variance.”  
**Purpose.** Reduce storage and compute; denoise.  
**Details.** Truncated SVD keeps the top \(k\) singular triplets: \(A_k=\sum_{i=1}^k \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top\). This is optimal in Frobenius norm (Eckart–Young).

### Pseudoinverse (Moore–Penrose)
**Definition.** The unique matrix \(A^+\) satisfying the Penrose conditions; computes minimum‑norm least‑squares solutions even when \(A\) is rectangular or rank‑deficient.  
**Purpose.** Solve \(\min_x\|Ax-b\|_2^2\) robustly; handle under/overdetermined systems.  
**Details.** Via SVD \(A=U\Sigma V^\top\), set \(A^+=V\Sigma^+U^\top\) where \(\Sigma^+\) inverts nonzero singular values and leaves zeros for near‑zero ones (regularization effect).

### Least Squares (LS)
**Definition.** Optimization problem minimizing squared residuals: \(\min_x \|Ax-b\|_2^2\).  
**Purpose.** Core of **linear regression**, system identification, and many estimators under Gaussian noise.  
**Details.** Normal equations \(A^\top A\,x=A^\top b\) (avoid explicit inverse); use QR/SVD for stability. See gradient PNG in the doc (LS gradient / Hessian).

### Rectangular Matrix
**Definition.** A non‑square matrix \(A\in\mathbb{R}^{m\times n}\) with \(m\ne n\).  
**Purpose.** Models mappings between spaces of different dimensions (e.g., more samples than features or vice versa).  
**Details.** In LS: **overdetermined** (\(m>n\)) — many equations; **underdetermined** (\(m<n\)) — many solutions, pick minimum‑norm via \(A^+\).

### Rank‑Deficient
**Definition.** A matrix whose rank is less than \(\min(m,n)\).  
**Purpose.** Signals redundancy or collinearity in features; impacts identifiability and conditioning.  
**Details.** LS problems become ill‑posed; \(A^+\) (via SVD) yields the minimum‑norm solution; regularization (ridge) improves generalization and stability.

### Linearization
**Definition.** Approximating a nonlinear function near a point by its first‑order Taylor expansion: \(f(x)\approx f(x_0)+J_f(x_0)(x-x_0)\).  
**Purpose.** Analyze/optimize complex models locally; basis of Gauss–Newton, EKF, and backprop’s local derivatives.  
**Details.** Valid in a neighborhood where higher‑order terms are small; accuracy depends on curvature (Hessian).

### Backprop (Backpropagation)
**Definition.** Efficient algorithm to compute gradients of scalar losses through composite functions (neural nets) using the chain rule in reverse.  
**Purpose.** Enables training deep networks by gradient‑based optimization.  
**Details.** Implements repeated **vector‑Jacobian** products; memory‑efficient variants (checkpointing) trade compute for memory.

### Curvature
**Definition.** Second‑order behavior captured by the **Hessian**; tells how gradients change with position.  
**Purpose.** Drives step‑size choice, trust‑region radii, and convergence speed.  
**Details.** Large eigenvalues ⇒ steep directions; tiny eigenvalues ⇒ flat/ill‑conditioned directions; negative eigenvalues indicate saddle/non‑convex regions.

### Newton (Newton’s Method / Newton–Raphson)
**Definition.** Second‑order optimization method updating \(x_{k+1}=x_k - H^{-1}\nabla f(x_k)\).  
**Purpose.** Achieve **quadratic** local convergence near a well‑behaved optimum.  
**Details.** Exact Hessians are expensive; **quasi‑Newton** (BFGS/L‑BFGS) build low‑rank Hessian approximations using gradients only; **Hessian‑vector products** enable CG‑Newton without forming \(H\).
