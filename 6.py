from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import pandas as pd

wine = load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,   
    epochs=200,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)


print("\n--- Results ---")
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names
))

model.save("wine_model.h5")
print("\nModel saved!")



samples = 20   
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_new = np.random.normal(mean, std, (samples, X.shape[1]))

print("\nGenerated dataset:")
print(X_new[:5])

loaded_model = tf.keras.models.load_model("wine_model.h5")

pred_logits = loaded_model.predict(X_new)
pred_classes = np.argmax(pred_logits, axis=1)

print("\nPredictions for generated dataset:")

for i in range(samples):
    print(
        f"Sample {i+1}: {wine.target_names[pred_classes[i]]}"
    )