"""
SCRIPT 1 - TREINAMENTO DA CNN
===============================
Treina uma CNN com o dataset de lixo e exporta:
  - modelo_lixo.keras     (modelo completo)
  - modelo_lixo.tflite    (modelo leve para Raspberry Pi)
  - classes.txt           (nomes das classes)

MELHORIAS v2:
  - Augmentation agressivo para simular camera real
  - Fine-tuning das ultimas camadas do MobileNetV2
  - Regularizacao L2 para reduzir overfitting
  - Learning rate scheduler automatico

Como usar:
  python script1_treino.py

Dependencias:
  pip install tensorflow numpy matplotlib
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Sem GUI, salva direto em arquivo
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ── Configuracoes ──────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\Braga\Desktop\EcoScan\dataset_final"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_FASE1 = 15   # Fase 1: so treina o topo (rapido)
EPOCHS_FASE2 = 20   # Fase 2: fine-tuning (mais epocas, mais preciso)
SEED         = 42

print("[INFO] Carregando dataset...")
print(f"[INFO] Caminho: {DATASET_PATH}\n")

# ── Augmentation AGRESSIVO para simular camera real ───────────────────────────
# O dataset tem fotos em fundo branco/limpo
# Aqui simulamos condicoes reais: sombra, blur, angulos, etc.
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,           # Mais rotacao
    zoom_range=0.35,             # Mais zoom
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.2,       # Desloca horizontalmente
    height_shift_range=0.2,      # Desloca verticalmente
    shear_range=0.2,             # Distorcao
    brightness_range=[0.5, 1.5], # Variacao de brilho maior
    channel_shift_range=30.0,    # Muda as cores levemente
    fill_mode="nearest",
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_data = val_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
)

# Salva os nomes das classes
classes = list(train_data.class_indices.keys())
with open("classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(classes))

print(f"[OK] Classes ({len(classes)}): {classes}")
print(f"[OK] Treino: {train_data.samples} | Validacao: {val_data.samples}\n")

# ── Modelo MobileNetV2 ────────────────────────────────────────────────────────
print("[INFO] Construindo modelo MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Congela tudo na fase 1

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu",
                  kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu",
                  kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ── FASE 1: Treina so o topo ───────────────────────────────────────────────────
print("\n[FASE 1] Treinando o topo do modelo...\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_f1 = [
    EarlyStopping(monitor="val_accuracy", patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("modelo_lixo_melhor.keras", monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FASE1,
    callbacks=callbacks_f1,
)

# ── FASE 2: Fine-tuning das ultimas 50 camadas ────────────────────────────────
print("\n[FASE 2] Fine-tuning das ultimas camadas...\n")

base_model.trainable = True
# Descongela so as ultimas 50 camadas
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # LR menor
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_f2 = [
    EarlyStopping(monitor="val_accuracy", patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("modelo_lixo_melhor.keras", monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FASE2,
    callbacks=callbacks_f2,
)

# ── Avaliacao ──────────────────────────────────────────────────────────────────
_, val_acc = model.evaluate(val_data, verbose=0)
print(f"\n[OK] Acuracia final na validacao: {val_acc*100:.1f}%")

# ── Salva modelo completo ──────────────────────────────────────────────────────
model.save("modelo_lixo.keras")
print("[OK] Modelo salvo: modelo_lixo.keras")

# ── Exporta TFLite ────────────────────────────────────────────────────────────
print("\n[INFO] Exportando para TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("modelo_lixo.tflite", "wb") as f:
    f.write(tflite_model)

tamanho = os.path.getsize("modelo_lixo.tflite") / (1024 * 1024)
print(f"[OK] TFLite salvo: modelo_lixo.tflite ({tamanho:.1f} MB)")

# ── Grafico combinado das duas fases ──────────────────────────────────────────
acc  = history1.history["accuracy"]     + history2.history["accuracy"]
vacc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"]         + history2.history["loss"]
vloss= history1.history["val_loss"]     + history2.history["val_loss"]
fase2_inicio = len(history1.history["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(acc,  label="Treino",    color="#4FC3F7")
ax1.plot(vacc, label="Validacao", color="#FF8A65")
ax1.axvline(fase2_inicio, color="white", linestyle="--", alpha=0.5, label="Fine-tuning")
ax1.set_title("Acuracia", color="white")
ax1.set_xlabel("Epoca", color="white")
ax1.legend()
ax1.set_facecolor("#1e1e2e")
ax1.tick_params(colors="white")

ax2.plot(loss,  label="Treino",    color="#4FC3F7")
ax2.plot(vloss, label="Validacao", color="#FF8A65")
ax2.axvline(fase2_inicio, color="white", linestyle="--", alpha=0.5, label="Fine-tuning")
ax2.set_title("Loss", color="white")
ax2.set_xlabel("Epoca", color="white")
ax2.legend()
ax2.set_facecolor("#1e1e2e")
ax2.tick_params(colors="white")

fig.patch.set_facecolor("#13131f")
plt.tight_layout()
plt.savefig("grafico_treino.png", dpi=120, bbox_inches="tight")
print("[OK] Grafico salvo: grafico_treino.png")
print("\n[PRONTO] Rode o script2_app.py para testar em tempo real!")