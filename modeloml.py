import numpy as np
import os

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import question_answer
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.question_answer import DataLoader

train_data_path = tf.keras.utils.get_file(
    fname='triviaqa-web-train-8000.json',
    origin='https://github.com/ccasimiro88/TranslateAlignRetrieve/raw/master/SQuAD-es-v2.0/train-v2.0-es_small.json')
validation_data_path = tf.keras.utils.get_file(
    fname='triviaqa-verified-web-dev.json',
    origin='https://github.com/ccasimiro88/TranslateAlignRetrieve/raw/master/SQuAD-es-v2.0/dev-v2.0-es_small.json')
spec = model_spec.get('mobilebert_qa_squad')
train_data = DataLoader.from_squad(train_data_path, spec, is_training=True)
validation_data = DataLoader.from_squad(validation_data_path, spec, is_training=False)
model = question_answer.create(train_data, model_spec=spec)
model.summary()
model.evaluate(validation_data)
model.export(export_dir='.')
model.export(export_dir='.', export_format=ExportFormat.VOCAB)
model.evaluate_tflite('model.tflite', validation_data)
