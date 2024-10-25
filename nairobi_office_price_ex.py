import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV file
data = pd.read_csv('Nairobi Office Price Ex.csv')

# Extract the 'SIZE' and 'PRICE' columns as numpy arrays
size = data['SIZE'].values
price = data['PRICE'].values

# Function to compute Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(x)
    y_pred = m * x + c
    # Calculating gradients
    dm = (-2 / N) * np.sum(x * (y - y_pred))  # Gradient w.r.t m (slope)
    dc = (-2 / N) * np.sum(y - y_pred)        # Gradient w.r.t c (y-intercept)
    # Updating weights
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

# Linear Regression using Gradient Descent for 50 epochs (higher number of epochs)
def linear_regression(x, y, epochs=50, learning_rate=0.0001):
    # Random initial values for m (slope) and c (y-intercept)
    m, c = np.random.rand(), np.random.rand()
    
    for epoch in range(epochs):
        y_pred = m * x + c
        mse = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch + 1}, MSE: {mse}")
        
        # Update m and c using gradient descent
        m, c = gradient_descent(x, y, m, c, learning_rate)
    
    # After training, plot the final regression line
    plt.scatter(x, y, color='blue')  # Plot data points
    plt.plot(x, m * x + c, color='red')  # Line of best fit
    plt.xlabel('Office Size')
    plt.ylabel('Office Price')
    
    # Fix the axes so that they match the data range
    plt.xlim(min(x) - 5, max(x) + 5)  # Set x-axis limits
    plt.ylim(min(y) - 10, max(y) + 10)  # Set y-axis limits
    
    plt.title('Line of Best Fit after Final Epoch')
    plt.savefig('final_fit.png', dpi=300)
    return m, c

# Train the model
m, c = linear_regression(size, price, epochs=10, learning_rate=0.0001)


size_to_predict = 100
predicted_price = m * size_to_predict + c
print(f"Predicted price for office size {size_to_predict} sq. ft: {predicted_price:.2f}")