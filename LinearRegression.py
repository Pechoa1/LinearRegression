# goal is to recreate a simple linear regression with gradiant descent using MSE

# Algorithm goes as follows:
# 1. initialize some random m and b
# 2. forward prop - calculate predictions where y{hat} = mx + b
# 3. compute cost J(m,b)
# 4. calculate the gradients of the cost function with respect to m and b
# 5. update m and b using the gradiant * learn rate respectively
# 6. loop 2-5 until cost converges
# 7. print final parameters for m and c

import numpy as np
import pandas as pd


class LinearRegressionGD:
    def __init__(self, learning_rate, iterations):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = np.random.randn()  # Initial slope
        self.b = np.random.randn()  # Initial intercept

    def fit(self, X, y):
        """
        Fit the model to the data using gradient descent.
        """
        m_history = []  # To track the history of the slope
        b_history = []  # To track the history of the intercept
        cost_history = []  # To track the history of the cost function
        prev_cost = float("inf")  # Store inf for prev_cost on first run

        for i in range(self.iterations):
            # obtain y-hat: y=mx+b
            y_pred = self.m * X + self.b

            # calc cost using Mean Squared Error
            cost = np.mean((y_pred - y) ** 2)

            # used to stop regression early if insignificant progress is being made
            if abs(prev_cost - cost) < 1e-6:
                print(f"Converged at iteration {i}")
                break

            # save new cost for next iter
            prev_cost = cost

            # Calculate gradients
            m_gradient = -2 * np.mean(X * (y - y_pred))  # Derivative with respect to m
            b_gradient = -2 * np.mean(y - y_pred)  # Derivative with respect to b

            # Update parameters
            self.m -= self.learning_rate * m_gradient
            self.b -= self.learning_rate * b_gradient

            # Record history
            m_history.append(self.m)
            b_history.append(self.b)
            cost_history.append(cost)

            # print ever 100 iters
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost}, m = {self.m}, b = {self.b}")

        return m_history, b_history, cost_history

# could have a function here to make production with final m and b

if __name__ == "__main__":
    # using a sample data set from geeksforgeeks for this
    url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
    data = pd.read_csv(url)

    # Drop missing values
    data = data.dropna()

    # Prepare the training data
    train_input = np.array(data.x[0:500]).reshape(500, 1)  # Independent variable
    train_output = np.array(data.y[0:500]).reshape(500, 1)  # Dependent variable

    # Initialize the Linear Regression model
    model = LinearRegressionGD(learning_rate=0.00001, iterations=1000)

    # Fit the model to the data
    m_history, b_history, cost_history = model.fit(train_input, train_output)

    # Print the final parameters
    print(f"Final learned parameters: m = {model.m}, b = {model.b}")
