import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('fuel_consumption_vs_speed.csv')
X = data['speed_kmh'].values.reshape(-1, 1)
y = data['fuel_consumption_l_per_100km'].values


linear_model = LinearRegression()
poly_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())

models = {'Лінійна': linear_model, 'Поліном': poly_model}
results = []

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    results.append({
        'Модель': name,
        'MSE': mean_squared_error(y, y_pred),
        'MAE': mean_absolute_error(y, y_pred)
    })

print("Результати порівняння:")
print(pd.DataFrame(results))


speeds = np.array([35, 95, 140]).reshape(-1, 1)
preds = poly_model.predict(speeds)

print("\nПрогнози:")
for s, p in zip([35, 95, 140], preds):
    print(f"{s} km/h ---- {p:.2f} l/100km")




X_plot = np.linspace(X.min() - 5, X.max() + 30, 100).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Дані')
plt.plot(X_plot, linear_model.predict(X_plot), '--', label='Лінійна')
plt.plot(X_plot, poly_model.predict(X_plot), label='Поліном', linewidth=2)
plt.legend()
plt.grid(True)
plt.show()
