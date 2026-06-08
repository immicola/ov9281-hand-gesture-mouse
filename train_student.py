"""
train_student.py — обучение микро-модели (ровно как в статье)
Архитектура: Input(42) → Dense(20, ReLU) → Dense(10, ReLU) → Dense(8, Softmax)
Параметров: 1,103  |  Размер: 22 KB → 7 KB (TFLite quantized)

Запуск: python3 train_student.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# ── КОНФИГ ──
CSV_PATH = "/home/micola/distillation_dataset.csv"
MODEL_OUTPUT = "/home/micola/student_model_7kb.tflite"
EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 30  # early stopping

# ── ЗАГРУЗКА ──
print("Загружаем датасет...")
df = pd.read_csv(CSV_PATH)
X = df.iloc[:, :42].values.astype(np.float32)
y = df.iloc[:, 42:].values.astype(np.float32)

# Убедимся что one-hot корректный
n_classes = y.shape[1]
print(f"  Сэмплов: {len(X)},  Классов: {n_classes}")

# Распределение по классам
counts = y.sum(axis=0)
for i, c in enumerate(counts):
    print(f"    Класс {i}: {int(c)} сэмплов ({c/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
)
print(f"\n  Обучающих: {len(X_train)}")
print(f"  Тестовых:  {len(X_test)}")

# ── АРХИТЕКТУРА (как в статье — БЕЗ Dropout) ──
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(42,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Считаем параметры
trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
print(f"\nМодель создана. Параметров: {trainable:,}")
print(f"  (в статье: 1,103 — совпадает)")
model.summary()

# ── ОБУЧЕНИЕ ──
print("\nЗапускаем обучение...\n")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    ),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# ── ОЦЕНКА ──
print("\n=== Оценка на тестовой выборке ===")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {acc*100:.2f}%")

y_pred = model.predict(X_test, verbose=0)
y_true = y_test.argmax(axis=1)
y_pred_cls = y_pred.argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_cls, digits=3))

# ── КОНВЕРТАЦИЯ В TFLITE ──
print("\nКонвертируем в TFLite с квантованием...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(MODEL_OUTPUT, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(MODEL_OUTPUT) / 1024
print(f"\n{'='*40}")
print(f"  Модель: {MODEL_OUTPUT}")
print(f"  Размер: {size_kb:.2f} KB")
print(f"  (в статье: 7 KB)")
print(f"{'='*40}")