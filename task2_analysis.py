
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("cleaned_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sales Trend
df.groupby('Date')['Sales'].sum().plot(figsize=(10,5), title="Sales Trend Over Time")
plt.show()

# Profit vs Discount
sns.scatterplot(x='Discount', y='Profit', data=df, alpha=0.6)
plt.title("Profit vs Discount")
plt.show()

# Sales by Region
sns.barplot(x='Region', y='Sales', data=df, estimator=sum)
plt.title("Sales by Region")
plt.show()

# Sales by Category
sns.barplot(x='Category', y='Sales', data=df, estimator=sum)
plt.title("Sales by Category")
plt.show()

# Linear Regression Model
X = df[['Profit', 'Discount']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))


# -------- PROJECT 1: GENERAL EDA --------

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Histograms
df[['Sales','Profit','Discount']].hist(bins=20, figsize=(12,6))
plt.suptitle("Histograms of Sales, Profit, and Discount")
plt.show()

# Boxplots
sns.boxplot(data=df[['Sales','Profit','Discount']])
plt.title("Boxplots for Outlier Detection")
plt.show()

# Heatmap
sns.heatmap(df[['Sales','Profit','Discount']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
