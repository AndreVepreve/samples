# Day 1: Linear Algebra Fundamentals

## Topics and Concepts

### 1. Vectors
- **Definition**: A vector is a one-dimensional array that represents quantities having both magnitude and direction. 
- **Notation**: Often represented as \( \vec{v} \) or in list format like \( [v_1, v_2, \ldots, v_n] \).

### 2. Matrices
- **Definition**: A matrix is a two-dimensional array of numbers arranged in rows and columns.
- **Dimensions**: Denoted as \( m \times n \) where \( m \) is the number of rows and \( n \) is the number of columns. 
- **Example**: 
    \[
    \begin{bmatrix}
    2 & -3 & 1 \\
    2 & 0 & -1 \\
    1 & 4 & 5
    \end{bmatrix}
    \]

### 3. Properties of Matrices
- A vector is a special case of a matrix, either a single row or column. For example, a row vector with elements \( [2, 3, 4] \) is a 1x3 matrix.

### 4. Matrix Operations
- **Addition and Subtraction**: Can only be performed on matrices of the same dimensions element-wise.
- **Scalar Multiplication**: Multiplying every element of a matrix by a scalar value.
- **Example**: If \( A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \) and the scalar is \( 2 \), then \( 2A = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix} \).
  
- **Matrix Multiplication**: Requires that the number of columns in the first matrix equals the number of rows in the second matrix. The elements of the resulting matrix are computed using the dot product of rows and columns.

## Glossary

- **Vector**: A one-dimensional array representing quantities with both magnitude and direction.
- **Matrix**: A rectangular array of numbers arranged in rows and columns.
- **Dimensions**: Refers to the size of a matrix, denoted as \( m \times n \) where \( m \) is the number of rows and \( n \) is the number of columns.
- **Element-wise Operations**: Operations that are performed on corresponding elements of matrices of the same dimensions.
- **Scalar**: A single numerical value, often used to multiply matrices.
- **Dot Product**: A form of multiplication for vectors that produces a scalar, used in matrix multiplication.