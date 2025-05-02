# test_model.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os

# モデルの読み込み
model = tf.keras.models.load_model('models/falcon_lunch_model_finetuned.h5')

# テスト画像のディレクトリ
test_dir = 'data/test_images/'

# 画像の前処理関数
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 正規化
    img_array = np.expand_dims(img_array, axis=0)  # バッチ次元の追加
    return img_array

# テスト画像のリスト
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 結果表示
plt.figure(figsize=(15, 10))
for i, image_name in enumerate(test_images[:9]):  # 最初の9枚を表示
    img_path = os.path.join(test_dir, image_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = preprocess_image(img_path)
    
    # 予測
    score = model.predict(img_array)[0][0]
    
    # 表示
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f'Score: {score:.1f}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('models/test_predictions.png')
plt.show()