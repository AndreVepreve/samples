
# Mathematics for Machine Learning — A Practical Refresher

> A dense, exam‑ready chapter you can print or paste into your repo. All equations use `$$ … $$` block math so they render cleanly on GitHub and VS Code.

---

## Table of Contents

1. Linear Algebra Essentials  
2. Norms, Inner Products, and Inequalities  
3. Matrix Identities You’ll Use Every Day  
4. Eigenvalues, Spectral Theorem & SVD  
5. Pseudoinverse & Least‑Squares Geometry  
6. Calculus & Matrix Calculus (with common gradients)  
7. Optimization Basics for ML (convexity, GD/SGD)  
8. Numerical Stability & Conditioning  
9. Practical Patterns (code)  
10. Quick Exercises (with answers sketched)

---

## 1) Linear Algebra Essentials

**What this section gives you.** The minimum set of objects and identities you’ll actually use in ML code and derivations.

**Objects & notation**
- **Scalars** \(a\in\mathbb{R}\), **vectors** \(\mathbf{x}\in\mathbb{R}^n\), **matrices** \(A\in\mathbb{R}^{m\times n}\), **tensors** (higher‑order arrays).
- **Standard basis** \(e_i\): all zeros except a 1 in position \(i\); any \(\mathbf{x}\) can be written \(\sum_i x_i e_i\).
- **Identity** \(I\), **zero** \(0\); **transpose** \(A^\top\); **symmetric** \(A=A^\top\); **SPD/PSD** \(A\succ 0/A\succeq 0\).

**Linear combinations, span, independence**
- A **linear combination** is \(\sum_i \alpha_i \mathbf{v}_i\).  
- The **span** of \(\{\mathbf{v}_i\}\) is all their linear combinations.  
- Vectors are **linearly independent** if only the trivial combination gives zero.  
- **Rank** of \(A\): dimension of its column space; by **rank–nullity**, for \(A\in\mathbb{R}^{m\times n}\): \(\dim \mathcal{N}(A)=n-\operatorname{rank}(A)\).

**Linear maps as matrices**
A matrix \(A\) is a linear map. Shapes tell you legality and cost:
- Matrix–vector \(A\mathbf{x}\in\mathbb{R}^m\), cost \(\mathcal{O}(mn)\).  
- Matrix–matrix \(AB\) defined when inner dims agree.  
- Block matrices let you express concatenation/partitioning cleanly.

**Orthogonality & projections**
Two vectors are **orthogonal** if \(\mathbf{x}^\top\mathbf{y}=0\). The **orthogonal projector** onto the column space of full‑column‑rank \(A\) is
```math
P = A(A^\top A)^{-1}A^\top,
```
so \(P\mathbf{b}\) is the closest point in \(\mathcal{R}(A)\) to \(\mathbf{b}\).

**Quadratic forms**
A symmetric \(Q\) defines \(q(\mathbf{x})=\mathbf{x}^\top Q \mathbf{x}\). If \(Q\succeq 0\) then \(q(\mathbf{x})\ge 0\). In ML this shows up in losses (least squares), regularizers (ridge), and curvature (Hessians).

**Why ML cares (purpose)**
- Shapes/rank explain *what you can compute* (e.g., when normal equations are solvable).  
- Projections = least squares; orthogonality underlies PCA and gradient steps.  
- Quadratic forms capture energy/variance and connect to curvature.

## 2) Norms, Inner Products, and Inequalities

**Why norms/inner products?** They formalize *size*, *distance*, and *angles*—the core of optimization, regularization, and geometry in ML.

**Vector norms**
- \( \ell_2 \): \( \|\mathbf{x}\|_2 = (\sum_i x_i^2)^{1/2} \) (rotation‑invariant).  
- \( \ell_1 \): \( \|\mathbf{x}\|_1 = \sum_i |x_i| \) (promotes sparsity; used in lasso).  
- \( \ell_\infty \): \( \|\mathbf{x}\|_\infty = \max_i |x_i| \) (adversarial \(\epsilon\)-balls).  
All norms on finite‑dimensional spaces are equivalent, but constants matter for optimization.

**Matrix norms**
- **Frobenius**: \( \|A\|_F = \sqrt{\sum_{ij} a_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)} \).  
- **Spectral/Operator 2‑norm**: \( \|A\|_2 = \sigma_{\max}(A) \) (largest singular value).  
- **Induced p‑norms**: \( \|A\|_p = \max_{\|\mathbf{x}\|_p=1}\|A\mathbf{x}\|_p \).

**Inner products & geometry**
```math
\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y}, \qquad
|\langle \mathbf{x}, \mathbf{y} \rangle| \le \|\mathbf{x}\|_2 \,\|\mathbf{y}\|_2 \quad\text{(Cauchy–Schwarz)}.
```
Angles via \( \cos\theta = \frac{\langle \mathbf{x},\mathbf{y}\rangle}{\|\mathbf{x}\|\,\|\mathbf{y}\|} \).

**Inequalities you actually use**
- **Triangle**: \( \|\mathbf{x}+\mathbf{y}\| \le \|\mathbf{x}\|+\|\mathbf{y}\| \).  
- **H\"older/Minkowski**: generalize Cauchy–Schwarz/triangle to \(p\)-norms.  
- **Submultiplicativity**: \( \|AB\| \le \|A\|\,\|B\| \).

**Purpose in ML**
- Norms define **regularizers** (L1/L2), **constraints**, **early stopping** targets.  
- Inner products drive **similarity** (cosine) and **projections**; inequalities bound errors and prove convergence.  
- Operator norms give **Lipschitz constants** and stability estimates.

## 3) Matrix Identities You’ll Use Every Day

**Trace tricks.**  
- \\(\\mathrm{tr}(AB)=\\mathrm{tr}(BA)\\) (when products are defined).  
- \\(\\mathrm{tr}(A)=\\sum_i \\lambda_i(A)\\) for square \\(A\\).  
- \\(\\|A\\|_F^2=\\mathrm{tr}(A^T A)\\).

**Determinant facts.**  
- \\(\\det(AB)=\\det(A)\\,\\det(B)\\).  
- SPD \\(A\\Rightarrow \\det(A)>0\\).

**Rank inequalities.**

```math
\\mathrm{rank}(AB)\\le \\min\\{\\mathrm{rank}(A),\\mathrm{rank}(B)\\}.
```

---

## 4) Eigenvalues, Spectral Theorem & SVD

#### Definitions & Purpose (Eigen / SVD)
- **Eigenvalue/Eigenvector — Definition.** Non‑zero \(x\) with \(Ax=\lambda x\); \(\lambda\) is the eigenvalue.
- **Eigenvalue/Eigenvector — Purpose.** Reveals invariant directions and stretch factors of a linear map; key in stability, PCA of covariance, and power iterations.
- **SVD — Definition.** Factorization \(A = U\Sigma V^\top\) with orthonormal \(U,V\) and non‑negative singular values on \(\Sigma\).
- **SVD — Purpose.** Universal for any rectangular matrix; gives low‑rank approximation (Eckart–Young), conditioning insight, and numerically stable bases for least squares.

### 4.1 Diagonalizability, Eigenspaces, and Multiplicities

Let \(A\in\mathbb{R}^{n\times n}\). The **eigenspace** for an eigenvalue \(\lambda\) is
\(\mathcal{E}_\lambda(A)=\{\,x:\,Ax=\lambda x\,\}\) whose dimension is the **geometric multiplicity** \(g_\lambda=\dim\mathcal{E}_\lambda(A)\).
The **algebraic multiplicity** \(a_\lambda\) is the multiplicity of \(\lambda\) as a root of the characteristic polynomial \(\chi_A(t)=\det(tI-A)\).

- Always \(1 \le g_\lambda \le a_\lambda\).
- \(A\) is **diagonalizable** iff the eigenvectors span \(\mathbb{R}^n\), equivalently iff \(\sum_{\lambda} g_\lambda = n\), which in turn is equivalent to \(g_\lambda=a_\lambda\) for every eigenvalue.
- A **defective** matrix is not diagonalizable; for it, some eigenvalue has \(g_\lambda < a_\lambda\).
- **Sufficient condition:** if \(A\) has \(n\) **distinct** eigenvalues then it is diagonalizable because each eigenspace is 1‑dimensional and eigenvectors from distinct eigenvalues are linearly independent.
- **Orthogonal diagonalization:** every real **symmetric** matrix \(A=A^\top\) is diagonalizable by an orthogonal matrix \(Q\): \(A=Q\Lambda Q^\top\) with \(\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)\) and \(Q^\top Q=I\) (spectral theorem).

**Jordan form (intuition).** When \(A\) is not diagonalizable, there exists a basis where
\(A\) assumes **Jordan normal form** (upper triangular blocks with \(\lambda\) on the diagonal and 1 on the superdiagonal).
The number of Jordan blocks for \(\lambda\) equals \(g_\lambda\), while their sizes add to \(a_\lambda\).
Diagonalizability is the special case where **all** blocks are size \(1\).

**Practical takeaways.**
- Symmetric/Hermitian matrices are the nicest case: real eigenvalues, orthonormal eigenvectors.
- Repeated eigenvalues require checking \(g_\lambda\); distinct eigenvalues guarantee diagonalizability.
- Numerical software rarely forms Jordan form; it is ill‑conditioned. Prefer Schur or SVD in practice.


**Eigen‑decomposition (symmetric).** If \\(A=A^T\\), then there exists an orthonormal basis of eigenvectors \\(U=[\\mathbf{u}_1\\cdots \\mathbf{u}_n]\\) and real eigenvalues \\(\\lambda_i\\) such that

```math
A = U\\,\\mathrm{diag}(\\lambda_1,\\ldots,\\lambda_n)\\,U^T.
```

This is the **spectral theorem**. Quadratic forms decouple:

```math
\\mathbf{x}^T A \\mathbf{x} = \\sum_i \\lambda_i\\,(\\mathbf{u}_i^T\\mathbf{x})^2.
```

**Singular Value Decomposition (SVD).** For any \\(A\\in\\mathbb{R}^{m\\times n}\\),

```math
A = U\\,\\Sigma\\,V^T,\\qquad U\\in\\mathbb{R}^{m\\times m},\\; V\\in\\mathbb{R}^{n\\times n},\\; \\Sigma=\\mathrm{diag}(\\sigma_1\\!\\ge\\!\\cdots\\!\\ge\\!\\sigma_r\\!\\ge\\!0).
```

Columns of \\(U\\) and \\(V\\) are orthonormal left/right singular vectors; \\(\\sigma_i\\) are singular values.  
**Best rank‑k approximation (Eckart–Young).** Truncate to the top \\(k\\) singular values: \\(A_k=U_{:k}\\Sigma_{:k,:k}V_{:k}^T\\) minimizes \\(\\|A-A_k\\|_F\\).

**Condition number.** \\(\\kappa_2(A)=\\sigma_{\\max}(A)/\\sigma_{\\min}(A)\\) (for full‑rank); large \\(\\kappa\\) \\(\\Rightarrow\\) numerical sensitivity.

---

## 5) Pseudoinverse & Least‑Squares Geometry

#### Definitions & Purpose (Pseudoinverse)
- **Moore–Penrose pseudoinverse — Definition.** \(A^+\) is the unique matrix satisfying the four Penrose conditions; via SVD, \(A^+=V\Sigma^+ U^\top\).
- **Moore–Penrose pseudoinverse — Purpose.** Produces least‑squares and minimum‑norm solutions for inconsistent/underdetermined systems and enables stable regression.

**Moore–Penrose pseudoinverse.** Defined for any \\(A\\): the unique matrix \\(A^+\\) satisfying the four Penrose conditions. If \\(A=U\\Sigma V^T\\), then

```math
A^+=V\\,\\Sigma^+\\,U^T,\\quad \\Sigma^+_{ii}=\\begin{cases}1/\\sigma_i,&\\sigma_i>0\\\\0,&\\sigma_i=0.\\end{cases}
```

**Least squares.** For overdetermined \\(A\\mathbf{x}\\approx \\mathbf{b}\\), the solution minimizing \\(\\|A\\mathbf{x}-\\mathbf{b}\\|_2\\) satisfies the **normal equations**

```math
A^T A\\,\\hat{\\mathbf{x}} = A^T\\mathbf{b},
```

or equivalently \\(\\hat{\\mathbf{x}}=A^+\\mathbf{b}\\). Geometrically, \\(A\\hat{\\mathbf{x}}\\) is the **orthogonal projection** of \\(\\mathbf{b}\\) onto \\(\\mathcal{R}(A)\\).

**Stable solvers.** Prefer QR/SVD over explicit normal equations when \\(A^T A\\) is ill‑conditioned.

---

## 6) Calculus & Matrix Calculus (with common gradients)

#### Definitions & Purpose (Jacobian / Hessian)
- **Jacobian — Definition.** Matrix of first partial derivatives \(J_{ij}=\partial g_i/\partial x_j\) for \(g:\mathbb{R}^n\to\mathbb{R}^m\).
- **Jacobian — Purpose.** Measures local sensitivity; underlies backpropagation and change‑of‑variables.
- **Hessian — Definition.** Matrix of second partials \(H_{ij}=\partial^2 f/\partial x_i\partial x_j\) for \(f:\mathbb{R}^n\to\mathbb{R}\).
- **Hessian — Purpose.** Encodes curvature; PSD Hessian characterizes convexity; used by Newton/Quasi‑Newton and to diagnose sharp/flat minima.

### 6.1 Jacobian & Hessian Shapes in Neural Networks

Consider an affine layer \(z=W x + b\) with \(W\in\mathbb{R}^{m\times n}\), \(b\in\mathbb{R}^m\).

- **Jacobian of affine:** \(\displaystyle J_{z,x} \equiv \frac{\partial z}{\partial x}=W\in\mathbb{R}^{m\times n}\).
- **ReLU:** \(y=\mathrm{ReLU}(z)\) has Jacobian \(J_{y,z}=\mathrm{diag}(\mathbb{1}_{z>0})\in\mathbb{R}^{m\times m}\).
- **Chain rule across layers:** for \(x\!\xrightarrow[]{W_1,b_1}\!z_1\!\xrightarrow[]{\phi_1}\!a_1\!\xrightarrow[]{W_2,b_2}\!\cdots\!\xrightarrow[]{\phi_L}\!a_L\), the Jacobian is the product of per‑layer Jacobians in reverse (backprop): \(J_{a_L,x} = \prod_{\ell=L}^{1} J_{\phi_\ell} J_{z_\ell,x_{\ell-1}}\).

**Softmax.** For \(s=\mathrm{softmax}(z)\in\mathbb{R}^K\) with \(s_i=\exp(z_i)/\sum_j \exp(z_j)\),
the Jacobian is the \(K\times K\) matrix

```math
J_{s,z}=\mathrm{diag}(s) - s\,s^\top.
```

With one‑hot \(y\), the cross‑entropy \(\ell(z,y)=-\sum_i y_i\log s_i\) has gradient
\(\nabla_z \ell = s - y\) and (per‑example) Hessian \(H_z=\mathrm{diag}(s) - s\,s^\top\) (PSD).

**Logistic regression (binary) Hessian.**
Stack examples into design matrix \(X\in\mathbb{R}^{N\times d}\) and let \(p=\sigma(X\theta)\), \(S=\mathrm{diag}(p\odot(1-p))\).
For negative log‑likelihood \(L(\theta)\), the Hessian is

```math
\nabla^2_\theta L(\theta) = X^\top S X \succeq 0,
```

which shows convexity (and strict convexity when \(X\) has full column rank and \(0<p_i<1\)).

**Convolutional layers as linear maps.**
A discrete convolution can be written as a matrix–vector product \(y = W_\mathrm{conv}\,x\) where \(W_\mathrm{conv}\)
is (block‑)Toeplitz with Toeplitz blocks. The Jacobian w.r.t. the input is exactly this large sparse matrix.
In practice, autodiff uses **vector–Jacobian products** and never materializes \(W_\mathrm{conv}\).

**Shapes cheat sheet.**
- Affine \(W\in\mathbb{R}^{m\times n}\): \(J_{z,x}\in\mathbb{R}^{m\times n}\), Hessian \(0\).
- Elementwise \(\phi:\mathbb{R}^m\!\to\!\mathbb{R}^m\): \(J_{\phi}=\mathrm{diag}(\phi'(z))\).
- Softmax \(s\in\mathbb{R}^K\): \(J_{s,z}\in\mathbb{R}^{K\times K}\) with \(J_{s,z} = \mathrm{diag}(s)-s s^\top\).
- Logistic loss: \(H_\theta = X^\top S X\in\mathbb{R}^{d\times d}\).


**Gradients, Jacobians, Hessians.** For \\(f:\\mathbb{R}^n\\to\\mathbb{R}\\), the gradient is \\(\\nabla f\\in\\mathbb{R}^n\\) and the Hessian is \\(\\nabla^2 f\\in\\mathbb{R}^{n\\times n}\\). For \\(g:\\mathbb{R}^n\\to\\mathbb{R}^m\\), the Jacobian \\(J\\in\\mathbb{R}^{m\\times n}\\) has entries \\(J_{ij}=\\partial g_i/\\partial x_j\\).

**Chain rule (vector form).**

```math
\\nabla_x f(g(x)) = J_g(x)^T\\,\\nabla f\\big( g(x) \\big).
```

**Common identities (assume compatible shapes; \\(A\\) constant):**

```math
\\nabla_{\\mathbf{x}}\\,(\\mathbf{a}^T\\!\\mathbf{x})=\\mathbf{a},\\quad
\\nabla_{\\mathbf{x}}\\,\\tfrac12\\|\\mathbf{x}\\|_2^2=\\mathbf{x},\\quad
\\nabla_{\\mathbf{x}}\\,\\tfrac12\\|A\\mathbf{x}-\\mathbf{b}\\|_2^2 = A^T(A\\mathbf{x}-\\mathbf{b}).
```

**Quadratic form.** For symmetric \\(Q\\),  

```math
\\nabla_{\\mathbf{x}}\\,(\\mathbf{x}^T Q\\mathbf{x}) = (Q+Q^T)\\mathbf{x}=2Q\\mathbf{x}. 
```

**Softmax & cross‑entropy (multi‑class).** Let \\(z\\in\\mathbb{R}^K\\), \\(\\mathrm{softmax}(z)_k = \\exp(z_k)/\\sum_j\\exp(z_j)\\). With one‑hot label \\(y\\), cross‑entropy \\(\\ell(z,y)=-\\sum_k y_k\\log\\mathrm{softmax}(z)_k\\) has gradient

```math
\\nabla_z\\,\\ell(z,y) = \\mathrm{softmax}(z) - y.
```

---

## 7) Optimization Basics for ML (convexity, GD/SGD)

**Convex sets & functions.** A set \\(C\\) is convex if \\(\\theta x+(1-\\theta)y\\in C\\) for any \\(x,y\\in C\\), \\(\\theta\\in[0,1]\\). A function \\(f\\) is convex if its domain is convex and

```math
f(\\theta x+(1-\\theta)y)\\le \\theta f(x)+(1-\\theta)f(y).
```

Important property: **every local minimum of a convex function is global**.

**Gradient descent (fixed step).** For differentiable convex \\(f\\) with L‑Lipschitz gradient, GD with step \\(\\eta\\le 1/L\\) satisfies

```math
f(x_k)-f(x^*)\\le \\frac{\\|x_0-x^*\\|_2^2}{2\\eta k},
```

i.e., \\(\\mathcal{O}(1/k)\\) sublinear convergence.

**Stochastic gradient descent.** Replace full gradients with unbiased minibatch estimates; use decaying steps or adaptive methods (Adam) and iterate averaging; monitor validation loss to avoid overfitting.

---

## 8) Numerical Stability & Conditioning

**Conditioning.** Relative error amplification scales with condition number \\(\\kappa(A)\\). Prefer algorithms with better stability (QR/SVD) and regularize ill‑posed problems (ridge: add \\(\\lambda I\\)).

**Don’t invert explicitly.** Solve \\(A\\mathbf{x}=\\mathbf{b}\\) with factorizations (`solve`) or least‑squares (`lstsq`), not `inv(A)@b`.

**Log‑sum‑exp trick.** For stability in softmax/log‑likelihood, compute

```math
\\log\\sum_i e^{z_i} = z_{\\max} + \\log\\sum_i e^{z_i-z_{\\max}}.
```

---

## 9) Practical Patterns (code)

```python
# Shapes & projections
import numpy as np

A = np.array([[1.,2.],[2.,5.],[3.,4.]])   # m x n (3x2), full column rank
b = np.array([1., 0., 2.])

# Least squares via stable solvers
x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)  # SVD-based
proj = A @ x_hat                               # projection of b onto R(A)

# Pseudoinverse & projection matrix
A_pinv = np.linalg.pinv(A)
proj2 = A @ A_pinv @ b
assert np.allclose(proj, proj2)

# SVD and best rank-k approximation
M = np.random.randn(6, 4)
U, S, Vt = np.linalg.svd(M, full_matrices=False)
k = 2
Mk = (U[:, :k] * S[:k]) @ Vt[:k, :]  # top-k
```

```python
# Matrix calculus patterns (verify numerically)
import numpy as np

rng = np.random.default_rng(0)
Q = rng.standard_normal((5,5))
Q = 0.5*(Q+Q.T)           # symmetric
x = rng.standard_normal(5)

# f(x) = x^T Q x, grad = 2 Q x
def f(x): return x @ Q @ x
def grad(x): return 2*Q @ x

# Finite-difference check
eps = 1e-6
v = rng.standard_normal(5)
lhs = (f(x + eps*v) - f(x - eps*v)) / (2*eps)
rhs = grad(x) @ v
assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-6)
```

---

## 10) Quick Exercises (answers sketched)

1. **Projection & normal equations.** Prove that the least‑squares residual \\(\\mathbf{r}=\\mathbf{b}-A\\hat{\\mathbf{x}}\\) is orthogonal to every column of \\(A\\).  
   *Sketch:* differentiate \\(\\tfrac12\\|A\\mathbf{x}-\\mathbf{b}\\|^2\\), set gradient to zero to get \\(A^T(A\\hat{\\mathbf{x}}-\\mathbf{b})=0\\Rightarrow A^T\\mathbf{r}=0\\).

2. **Spectral theorem application.** For SPD \\(Q\\), show \\(\\min_{\\|\\mathbf{x}\\|=1} \\mathbf{x}^T Q\\mathbf{x}=\\lambda_{\\min}(Q)\\) and the maximizer corresponds to \\(\\lambda_{\\max}(Q)\\).  
   *Sketch:* express \\(\\mathbf{x}^TQ\\mathbf{x}\\) in the eigenbasis.

3. **SVD & low‑rank denoising.** Given \\(Y=X+E\\) with \\(X\\) rank‑\\(k\\) and small noise \\(E\\), argue why the top‑\\(k\\) SVD truncation recovers most of \\(X\\)’s energy.

4. **Matrix calculus.** Derive \\(\\nabla_{\\mathbf{x}}\\,\\tfrac12\\|W\\mathbf{x}-\\mathbf{b}\\|_2^2=W^T(W\\mathbf{x}-\\mathbf{b})\\) using index notation or trace.

5. **GD rate.** For an \\(L\\)-smooth convex function, verify the \\(\\mathcal{O}(1/k)\\) rate by telescoping the standard descent lemma with \\(\\eta\\le 1/L\\).

---

### Suggested filename
`mathematics_for_machine_learning.md` (place under `samples/Week3/` in your repo).
