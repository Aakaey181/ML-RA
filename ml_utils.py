import os, time, random, shutil, requests, numpy as np, tensorflow as tf
# from duckduckgo_search import DDGS
import tensorflow.keras.applications.mobilenet_v2 as mobilenet_v2

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = mobilenet_v2.preprocess_input(img)
    return img, label

def create_dataset(paths, labs):
    ds = tf.data.Dataset.from_tensor_slices((paths, labs))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
