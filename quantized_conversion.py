# quantized_conversion.py
import tensorflow as tf
import numpy as np

# モデルの読み込み
model = tf.keras.models.load_model('models/falcon_lunch_model_finetuned.h5')

# キャリブレーションデータセットの作成（代表的な入力サンプル）
def representative_dataset():
    # キャリブレーション用の代表的な入力データ
    # 実際のデータセットから少量のサンプルを使用
    for img in X_train[:100]:
        img = np.expand_dims(img, axis=0)
        yield [img.astype(np.float32)]

# TFLiteコンバーターの作成
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# INT8量子化の設定
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

# 変換
tflite_model_quant = converter.convert()

# ファイルに保存
with open('models/falcon_lunch_model_quant.tflite', 'wb') as f:
    f.write(tflite_model_quant)

# モデルサイズの確認
import os
original_size = os.path.getsize('models/falcon_lunch_model.tflite') / (1024 * 1024)
quantized_size = os.path.getsize('models/falcon_lunch_model_quant.tflite') / (1024 * 1024)
print(f'オリジナルモデルサイズ: {original_size:.2f} MB')
print(f'量子化モデルサイズ: {quantized_size:.2f} MB')
print(f'サイズ削減率: {(1 - quantized_size/original_size) * 100:.2f}%')