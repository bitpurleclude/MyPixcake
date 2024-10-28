import os
import json
import argparse
from unittest import result

import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import grpc
from keras.src.export.export_lib import TFSMLayer
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

# 设置日志级别为 ERROR，以抑制警告和信息性日志
tf.get_logger().setLevel('ERROR')

TFS_HOST = 'localhost'
TFS_PORT = 8500

def load_image(img_file, target_size):
    """加载并调整图像大小。"""
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))

def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()

def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()

def get_image_quality_predictions(image_path, model_path):
    # Load and preprocess image
    image = load_image(image_path, target_size=(224, 224))  # 使用整合后的 load_image 函数
    image = keras.applications.mobilenet.preprocess_input(image)

    # Load the model using TFSMLayer with the correct endpoint
    model = tf.keras.Sequential([
        TFSMLayer(model_path, call_endpoint='image_quality')
    ])

    # Run the model
    prediction = model.predict(np.expand_dims(image, axis=0))

    # 访问质量预测
    quality_prediction = prediction['quality_prediction'][0]  # 获取数组中的第一个元素

    # 计算均值评分
    result = round(calc_mean_score(quality_prediction), 3)  # 将结果四舍五入到小数点后3位

    # 按要求输出结果格式
    output = {'mean_score_prediction': result}
    print(json.dumps(output, indent=2))  # 打印格式化的JSON字符串

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', help='Path to image file.', required=True)
    parser.add_argument('-mp', '--model-path', help='Path to model directory.', required=True)
    args = parser.parse_args()
    get_image_quality_predictions(args.image_path, args.model_path)
