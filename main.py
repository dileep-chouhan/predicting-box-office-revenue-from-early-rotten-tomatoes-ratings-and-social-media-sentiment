import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'RottenTomatoesScore': np.random.randint(0, 101, num_movies),
    'SocialMediaSentiment': np.random.normal(0, 1, num_movies), # Scale of -1 to +1
    'BoxOfficeRevenue': 100000 + 50000 * np.random.rand(num_movies) + 10000 * np.random.normal(0,1,num_movies) + 20000*np.random.randint(0,101,num_movies)/100 #Adding noise and correlation
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# No significant cleaning needed for synthetic data, but real-world data would require handling missing values and outliers.
# --- 3. Feature Engineering ---
# Consider adding interaction terms or polynomial features for better model fit if needed. This is omitted for simplicity in synthetic data.
# --- 4. Model Building ---
X = df[['RottenTomatoesScore', 'SocialMediaSentiment']]
y = df['BoxOfficeRevenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 6. Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Box Office Revenue")
plt.ylabel("Predicted Box Office Revenue")
plt.title("Actual vs. Predicted Box Office Revenue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') #Line of perfect prediction
plt.tight_layout()
output_filename = 'actual_vs_predicted.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10,6))
sns.regplot(x='RottenTomatoesScore',y='BoxOfficeRevenue',data=df)
plt.title('Rotten Tomatoes Score vs Box Office Revenue')
plt.tight_layout()
output_filename2 = 'rotten_tomatoes_vs_revenue.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
#Further analysis and model improvement could involve exploring other features, different regression models, or hyperparameter tuning.  This example provides a basic framework.