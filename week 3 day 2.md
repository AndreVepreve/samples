# Day 2: Advanced Linear Algebra Concepts

## Topics and Concepts

### 1. Determinants
- **Definition**: The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important information about the matrix, such as whether it is invertible. 
- **Interpretation**: A determinant of zero indicates that the matrix does not have an inverse.

### 2. Matrix Inverses
- **Definition**: The inverse of a matrix \( A \) is another matrix \( A^{-1} \) such that:
  \[
  A \cdot A^{-1} = I
  \]
  where \( I \) is the identity matrix.
- **Calculation**: For a \( 2 \times 2 \) matrix \( \begin{bmatrix} a & b \\ c & d \end{bmatrix} \),
  the inverse is given by:
  \[
  \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
  \]
  provided \( ad - bc 
eq 0 \).

### 3. Eigenvalues and Eigenvectors
- **Eigenvalues**: A scalar \( \lambda \) that indicates how much the eigenvector is stretched or compressed during the transformation represented by a matrix.
- **Eigenvectors**: A non-zero vector \( \vec{v} \) that changes at most by a scalar factor during the matrix transformation, defined by:
  \[
  A \vec{v} = \lambda \vec{v}
  \]

### 4. Matrix Decomposition
- **Definition**: The process of breaking down a matrix into a product of matrices to simplify operations. Common types include:
  - **LU Decompostion**: Splitting a matrix into a lower triangular matrix and an upper triangular matrix.
  - **QR Decomposition**: Decomposing a matrix into an orthogonal matrix and an upper triangular matrix.

## Glossary

- **Determinant**: A scalar value derived from a square matrix representing its properties like invertibility.
- **Matrix Inverse**: A matrix that, when multiplied by the original matrix, results in the identity matrix.
- **Eigenvalue**: A scalar that describes how a linear transformation affects a corresponding eigenvector.
- **Eigenvector**: A vector that remains in the same direction after the transformation represented by the matrix.
- **Matrix Decomposition**: The process of breaking down a matrix into simpler, constituent matrices.
- **Identity Matrix**: A square matrix with ones on the diagonal and zeros elsewhere, denoted by \( I \).