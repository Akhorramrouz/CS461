# Logistic Regression & Gradient Descent Tutorial
*A Step-by-Step Guide for Undergraduate Students*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Linear vs Logistic Regression](#linear-vs-logistic-regression)
3. [The Sigmoid Function](#the-sigmoid-function)
4. [Logistic Regression Mathematics](#logistic-regression-mathematics)
5. [Gradient Descent Fundamentals](#gradient-descent-fundamentals)
6. [Real-World Example: Email Spam Detection](#real-world-example-email-spam-detection)
7. [Step-by-Step Implementation](#step-by-step-implementation)
8. [Complete Code Examples](#complete-code-examples)
9. [Practice Exercises](#practice-exercises)

---

## 1. Introduction

### What is Logistic Regression?
Logistic Regression is a statistical method used for **binary classification** problems - predicting whether something belongs to one of two categories (Yes/No, Spam/Not Spam, Pass/Fail).

### Real-World Applications:
- **Medical Diagnosis**: Will a patient have a disease? (Yes/No)
- **Marketing**: Will a customer buy a product? (Buy/Don't Buy)
- **Finance**: Will a loan default? (Default/Safe)
- **Technology**: Is an email spam? (Spam/Ham)

---

## 2. Linear vs Logistic Regression

### Linear Regression Problem
Linear regression predicts continuous values but can output any number (-∞ to +∞). For classification, we need probabilities (0 to 1).

**Example**: Predicting house prices vs Predicting if someone will buy a house

### Why Not Linear Regression for Classification?
```
Linear Regression Output: -2, 0.5, 1.2, 3.7
❌ These aren't valid probabilities!

Logistic Regression Output: 0.1, 0.6, 0.8, 0.9
✅ These are valid probabilities (0-1 range)
```

---

## 3. The Sigmoid Function

### The Mathematical Foundation
The **sigmoid function** transforms any real number into a value between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ` (linear combination)

### Key Properties:
- **Input**: Any real number (-∞ to +∞)
- **Output**: Probability between 0 and 1
- **Shape**: S-shaped curve
- **Interpretation**: 
  - σ(z) > 0.5 → Predict Class 1
  - σ(z) < 0.5 → Predict Class 0

### Visual Understanding:
```
z = -∞  →  σ(z) = 0.0   (Definitely Class 0)
z = 0    →  σ(z) = 0.5   (Uncertain)
z = +∞  →  σ(z) = 1.0   (Definitely Class 1)
```

---

## 4. Logistic Regression Mathematics

### The Model
For a simple case with one feature:
```
P(y=1|x) = σ(β₀ + β₁x)
P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x)))
```

### Cost Function: Log-Likelihood
We can't use Mean Squared Error (like in linear regression). Instead, we use **Log-Likelihood**:

```
Cost(β) = -1/m * Σ[yᵢ * log(σ(zᵢ)) + (1-yᵢ) * log(1-σ(zᵢ))]
```

**Why this cost function?**
- When y=1: Cost = -log(σ(z)) → Penalizes low probabilities for positive examples
- When y=0: Cost = -log(1-σ(z)) → Penalizes high probabilities for negative examples

---

## 5. Gradient Descent Fundamentals

### The Optimization Problem
We need to find the best parameters (β₀, β₁, β₂, ...) that minimize our cost function.

### Gradient Descent Algorithm
1. **Start** with random parameter values
2. **Calculate** the cost using current parameters
3. **Compute** gradients (derivatives) of cost with respect to each parameter
4. **Update** parameters in the opposite direction of gradients
5. **Repeat** until convergence

### Mathematical Formula
For each parameter βⱼ:
```
βⱼ = βⱼ - α * (∂Cost/∂βⱼ)
```

Where:
- **α (alpha)**: Learning rate (how big steps we take)
- **∂Cost/∂βⱼ**: Partial derivative (gradient) of cost with respect to βⱼ

### Gradient Calculation for Logistic Regression
```
∂Cost/∂βⱼ = 1/m * Σ[(σ(zᵢ) - yᵢ) * xᵢⱼ]
```

### Key Concepts:
- **Learning Rate (α)**:
  - Too large: May overshoot the minimum
  - Too small: Very slow convergence
  - Typical values: 0.01, 0.1, 0.3
- **Convergence**: When cost stops decreasing significantly
- **Iterations**: Number of times we update parameters

---

## 6. Real-World Example: Email Spam Detection

### Problem Setup
**Objective**: Predict if an email is spam based on word frequencies

**Features**:
- x₁: Frequency of word "free"
- x₂: Frequency of word "money"  
- x₃: Number of exclamation marks
- y: 1 if spam, 0 if not spam

### Sample Data
```
Email 1: "free money!!!" → [3, 1, 3] → Spam (1)
Email 2: "meeting tomorrow" → [0, 0, 0] → Not Spam (0)
Email 3: "free trial available" → [1, 0, 0] → Not Spam (0)
```

### Model
```
P(Spam) = σ(β₀ + β₁*freq_free + β₂*freq_money + β₃*exclamation_count)
```

---

## 7. Step-by-Step Implementation

### Step 1: Data Preparation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, n_clusters_per_class=1, random_state=42)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Step 2: Sigmoid Function
```python
def sigmoid(z):
    """
    Sigmoid activation function
    Args:
        z: linear combination (β₀ + β₁x₁ + β₂x₂ + ...)
    Returns:
        Probability between 0 and 1
    """
    # Clip z to prevent overflow
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))
```

### Step 3: Cost Function
```python
def compute_cost(X, y, weights):
    """
    Compute logistic regression cost
    """
    m = X.shape[0]
    z = X.dot(weights)
    predictions = sigmoid(z)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-7
    cost = -1/m * (y.dot(np.log(predictions + epsilon)) + 
                   (1-y).dot(np.log(1-predictions + epsilon)))
    return cost
```

### Step 4: Gradient Computation
```python
def compute_gradients(X, y, weights):
    """
    Compute gradients for logistic regression
    """
    m = X.shape[0]
    z = X.dot(weights)
    predictions = sigmoid(z)
    
    # Gradient formula: (1/m) * X^T * (predictions - y)
    gradients = 1/m * X.T.dot(predictions - y)
    return gradients
```

### Step 5: Gradient Descent
```python
def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent to optimize weights
    """
    # Add bias term (intercept)
    X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize weights randomly
    weights = np.random.normal(0, 0.01, X_with_bias.shape[1])
    
    costs = []
    
    for i in range(num_iterations):
        # Forward pass
        cost = compute_cost(X_with_bias, y, weights)
        costs.append(cost)
        
        # Compute gradients
        gradients = compute_gradients(X_with_bias, y, weights)
        
        # Update weights
        weights = weights - learning_rate * gradients
        
        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.4f}")
    
    return weights, costs
```

### Step 6: Training and Prediction
```python
# Train the model
final_weights, cost_history = gradient_descent(X, y, learning_rate=0.1, num_iterations=1000)

def predict(X, weights):
    """Make predictions using trained weights"""
    X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
    probabilities = sigmoid(X_with_bias.dot(weights))
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

# Make predictions
predictions, probabilities = predict(X, final_weights)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 8. Complete Code Examples

### Example 1: Simple Binary Classification
```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialize weights
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        # Gradient descent
        for i in range(self.num_iterations):
            z = X.dot(self.weights)
            predictions = self._sigmoid(z)
            
            cost = self._compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            gradients = 1/X.shape[0] * X.T.dot(predictions - y)
            self.weights -= self.learning_rate * gradients
    
    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X.dot(self.weights))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y, predictions):
        epsilon = 1e-7
        return -np.mean(y*np.log(predictions + epsilon) + 
                       (1-y)*np.log(1-predictions + epsilon))

# Example usage
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=42)

# Train model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print(f"Accuracy: {np.mean(predictions == y):.2%}")
```

### Example 2: Spam Detection with Text Features
```python
import numpy as np
import pandas as pd

# Simulated email data
def create_spam_dataset():
    np.random.seed(42)
    
    # Create features: word frequencies
    data = []
    labels = []
    
    # Generate spam emails
    for _ in range(200):
        free_count = np.random.poisson(3)  # Spam emails have more "free"
        money_count = np.random.poisson(2)  # More "money" 
        exclamation_count = np.random.poisson(4)  # More exclamation marks
        data.append([free_count, money_count, exclamation_count])
        labels.append(1)  # Spam
    
    # Generate ham (non-spam) emails
    for _ in range(300):
        free_count = np.random.poisson(0.5)  # Ham emails have fewer trigger words
        money_count = np.random.poisson(0.3)
        exclamation_count = np.random.poisson(0.8)
        data.append([free_count, money_count, exclamation_count])
        labels.append(0)  # Ham
    
    return np.array(data), np.array(labels)

# Create dataset
X, y = create_spam_dataset()
feature_names = ['freq_free', 'freq_money', 'exclamation_count']

# Train spam detector
spam_detector = LogisticRegression(learning_rate=0.1, num_iterations=1000)
spam_detector.fit(X, y)

# Test new emails
test_emails = [
    [5, 3, 8],  # High spam indicators
    [0, 0, 1],  # Low spam indicators
    [2, 1, 3]   # Medium spam indicators
]

for i, email in enumerate(test_emails):
    prob = spam_detector.predict_proba([email])[0]
    prediction = "SPAM" if prob > 0.5 else "HAM"
    print(f"Email {i+1}: Features {email} → {prediction} (probability: {prob:.3f})")
```

### Example 3: Visualization
```python
def plot_results(X, y, model):
    # Create a mesh to plot the decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probability of Class 1')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    
    # Plot 2: Cost function over iterations
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function During Training')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Generate and plot results
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=42)
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X, y)
plot_results(X, y, model)
```

---

## 9. Practice Exercises

### Exercise 1: Medical Diagnosis
Create a logistic regression model to predict if a patient has diabetes based on:
- Age
- BMI (Body Mass Index)
- Blood pressure

**Dataset Simulation**:
```python
def create_diabetes_dataset():
    np.random.seed(42)
    n_samples = 400
    
    # Generate features
    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)
    
    # Create labels based on risk factors
    risk_score = 0.02*age + 0.1*bmi + 0.01*blood_pressure - 4
    probabilities = 1 / (1 + np.exp(-risk_score))
    labels = np.random.binomial(1, probabilities)
    
    X = np.column_stack([age, bmi, blood_pressure])
    return X, labels

# Your task: Implement and train the model
```

### Exercise 2: Customer Purchase Prediction
Predict if a customer will make a purchase based on:
- Time spent on website (minutes)
- Number of pages viewed
- Previous purchases

### Exercise 3: Gradient Descent Variations
Experiment with different learning rates:
- Try α = 0.001, 0.01, 0.1, 1.0
- Plot cost functions for each
- Observe convergence behavior

---

## Key Takeaways

### Logistic Regression
1. **Purpose**: Binary classification (0 or 1 outcomes)
2. **Core**: Uses sigmoid function to convert linear outputs to probabilities
3. **Output**: Probability between 0 and 1
4. **Decision**: Threshold at 0.5 (typically)

### Gradient Descent
1. **Purpose**: Optimization algorithm to find best parameters
2. **Process**: Iteratively update parameters using gradients
3. **Key Parameter**: Learning rate (α) controls step size
4. **Goal**: Minimize cost function

### Practical Tips
1. **Feature Scaling**: Normalize features for better convergence
2. **Learning Rate**: Start with 0.01-0.1, adjust based on convergence
3. **Iterations**: Monitor cost function to determine when to stop
4. **Regularization**: Add L1/L2 regularization to prevent overfitting

---

## Further Reading
- Andrew Ng's Machine Learning Course
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- Scikit-learn documentation for comparison with professional implementations