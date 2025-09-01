# Day 3: Calculus for Machine Learning (Derivatives)

## Topics and Concepts

### 1. Introduction to Derivatives

- **Definition**: A derivative measures the rate at which a function changes with respect to its input. For a function \( f(x) \), the derivative denotes how \( f(x) \) changes as \( x \) changes.
- **Notation**: The derivative of \( f(x) \) is denoted as \( f'(x) \) or \( \frac{df}{dx} \).

### 2. Role of Derivatives in Optimization

- **Optimization Purpose**: Derivatives are crucial for optimizing loss functions in machine learning. They help find the minimum or maximum of a function by determining the slope of the tangent line at a point.
- **Example**: In linear regression, the loss function measures the error of predictions. By analyzing the derivative, we can determine how to adjust parameters to minimize this error.

### 3. Common Derivatives

- **Power Rule**: For \( f(x) = x^n \), the derivative is \( f'(x) = n \cdot x^{n-1} \).
- **Trigonometric Functions**: For \( f(x) = \sin(x) \), the derivative is \( f'(x) = \cos(x) \).

### 4. Python Implementation

- **SymPy Library**: A Python library used for symbolic mathematics, making it easier to compute derivatives.
  ```python
  import sympy as sp
  x = sp.symbols('x')
  f = x**3 - 5*x + 7
  derivative = sp.diff(f, x)
  print(derivative)
  ```

### 5. Computing Gradients

- **Gradient Definition**: The gradient is a vector of partial derivatives, representing the direction of steepest ascent in multivariable functions.
- **Gradient Descent**: An iterative optimization algorithm that adjusts parameters in the direction of the negative gradient to minimize the loss function.

## Glossary

- **Derivative**: The measure of how a function changes as its input changes.
- **Optimization**: The process of finding the best solution, often minimizing a loss function in machine learning.
- **Gradient**: A vector that contains all the partial derivatives of a multivariable function, indicating the direction of the steepest increase.
- **SymPy**: A Python library for symbolic mathematics, facilitating operations like differentiation.
- **Loss Function**: A function that measures the difference between predicted and actual outcomes in a model.
