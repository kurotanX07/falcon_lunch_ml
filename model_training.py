# model_training.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# データの読み込み
df = pd.read_csv('data/image_scores.csv')
image_dir = 'data/raw/'

# 画像の前処理関数
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 正規化
    return img_array

# データセットの準備
X = []
y = []

for idx, row in df.iterrows():
    img_path = os.path.join(image_dir, row['image_filename'])
    if os.path.exists(img_path):
        img_array = preprocess_image(img_path)
        X.append(img_array)
        y.append(row['score'])

X = np.array(X)
y = np.array(y)

# トレーニングとテストデータに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MobileNetV3Smallをベースにしたモデルの作成
base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# 転移学習のため、ベースモデルの層を凍結
base_model.trainable = False

# 新しいモデルを作成
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # 回帰なので活性化関数なし
])

# モデルのコンパイル
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# モデルの学習
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# モデルの保存
model.save('models/falcon_lunch_model.h5')

# 学習曲線の可視化
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('models/training_history.png')
plt.show()

# モデルの評価
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'テストデータでの平均絶対誤差: {test_mae:.2f}')