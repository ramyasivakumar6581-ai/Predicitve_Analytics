import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Input (Month) and Output (Sales)
X = data[['Month']]
y = data['Sales']
print(data)
print(data.isnull().sum())
# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict future sales for months 11, 12, and 13
future_months = pd.DataFrame({'Month': [11, 12, 13]})
predictions = model.predict(future_months)

# Display predictions
print("Predicted Sales:")
for month, sale in zip(future_months['Month'], predictions):
    print(f"Month {month}: {sale:.2f}")

# Model evaluation
r2 = r2_score(y, model.predict(X))
mse = mean_squared_error(y, model.predict(X))
print(f"\nModel Performance:")
print(f"R² Score: {r2:.2f}")
print(f"MSE: {mse:.2f}")

# Plot the graph
plt.figure(figsize=(8,6))
plt.scatter(data['Month'], data['Sales'], color="blue", label="Actual Sales")
plt.plot(data['Month'], model.predict(X), color="green", linewidth=2, label="Regression Line")
plt.scatter(future_months['Month'], predictions, color="red", marker="x", s=100, label="Predicted Sales")

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Prediction using Linear Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
