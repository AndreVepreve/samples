# Day 4: Calculus for Machine Learning (Integrals and Optimization)

## Topics and Concepts

### 1. Integrals
- **Definition**: An integral computes the area under a curve, representing the cumulative total of a function's values.
- **Types**:
  - **Definite Integral**: Represents the area under the curve between two limits \( a \) and \( b \).
    \[
    \int_a^b f(x) \, dx
    \]
  - **Indefinite Integral**: Represents a family of functions and contains a constant of integration.

### 2. Applications of Integrals
- **Probability Distributions**: Integrals are used to calculate probabilities from Probability Density Functions (PDFs).
- **Cost Functions**: Regularization in cost functions often requires integration, such as integrating over weights in Bayesian models.

### 3. Optimization Concepts
- **Local vs. Global Minima**:
  - **Local Minimum**: A point where function values are lower than neighboring points.
  - **Global Minimum**: The point with the lowest value of the function across the entire domain.
  
- **Convex Functions**: A function where any local minimum is also a global minimum, making optimization simpler.
  
- **Non-convex Functions**: Many neural network loss functions are non-convex, presenting challenges in optimization.

### 4. Stochastic Gradient Descent (SGD)
- **Definition**: An optimization algorithm that uses random subsets of data (mini-batches) to update parameters, facilitating faster convergence.
- **Variants**:
  - **Mini-batch SGD**: Involves updating the model using small batches of data.
  - **Momentum**: Uses the past gradients to accelerate convergence.
  - **Adam Optimizer**: Combines momentum with adaptive learning rates to improve performance.

## Glossary

- **Integral**: A mathematical operation that calculates the area under a curve defined by a function.
- **Definite Integral**: Integral evaluated between two specific limits, giving a numerical value.
- **Indefinite Integral**: Integral without specified limits, resulting in a general form of antiderivatives.
- **Probability Density Function (PDF)**: Function that describes the likelihood of a continuous random variable.
- **Cost Function**: A function used to measure the error of a model's predictions.
- **Local Minimum**: A point where the function value is lower than surrounding points.
- **Global Minimum**: The overall lowest point of a function across its domain.
- **Convex Function**: A function where any line segment between two points on the graph lies above the graph.
- **Non-convex Function**: A function that may have multiple local minima, complicating optimization.
- **Stochastic Gradient Descent (SGD)**: An optimization algorithm that updates model parameters using random subsets of data.
- **Momentum**: Technique that helps accelerate SGD by considering past gradients.
- **Adam Optimizer**: An optimization algorithm that adapts learning rates based on moment estimates.