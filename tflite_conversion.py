# tflite_conversion.py
import tensorflow as tf

# モデルの読み込み
model = tf.keras.models.load_model('models/falcon_lunch_model_finetuned.h5')

# TFLiteコンバーターの作成
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 変換
tflite_model = converter.convert()

# ファイルに保存
with open('models/falcon_lunch_model.tflite', 'wb') as f:
    f.write(tflite_model)