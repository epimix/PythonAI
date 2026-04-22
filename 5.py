import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv("./assets/fuel_consumption_vs_speed.csv")

X = df[['speed_kmh']]
y = df['fuel_consumption_l_per_100km']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

best_degree = 1
best_mse = 999999

for d in range(1, 6):
    poly = PolynomialFeatures(d)
    model = LinearRegression()

    model.fit(poly.fit_transform(X_train), y_train)
    pred = model.predict(poly.transform(X_test))

    mse = mean_squared_error(y_test, pred)
    print(f"Degree {d} -> MSE = {mse:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_degree = d

print("\nBest degree:", best_degree)

poly = PolynomialFeatures(best_degree)
model = LinearRegression()

model.fit(poly.fit_transform(X), y)

speeds = pd.DataFrame({'speed_kmh':[35,95,140]})
predictions = model.predict(poly.transform(speeds))

print("\nPredictions:")
for s, p in zip(speeds['speed_kmh'], predictions):
    print(f"{s} km/h -> {p:.2f} l/100km")

plt.scatter(X, y)
x_line = np.linspace(20, 150, 100).reshape(-1,1)
plt.plot(x_line, model.predict(poly.transform(x_line)))
plt.xlabel("Speed")
plt.ylabel("Fuel consumption")
plt.show()



model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

model.summary()

X = df[['speed_kmh']].values
y = df['fuel_consumption_l_per_100km'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

model.fit(
    X_scaled,
    y_scaled,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

example_speeds = np.array([[35], [95], [140]])
example_scaled = scaler_X.transform(example_speeds)

pred_scaled = model.predict(example_scaled)
predictions = scaler_y.inverse_transform(pred_scaled)

for s, p in zip(example_speeds, predictions):
    print(f"{s[0]} km/h -> {p[0]:.2f} l/100km")