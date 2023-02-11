import numpy as np
import pandas as pd
import tensorflow as tf


class MyModule(tf.Module):
    def __init__(self, value):
        self.weight = tf.Variable(value)

    @tf.function
    def multiply(self, x):
        return x * self.weight


mod = MyModule(3)
mod.multiply(tf.constant([1, 2, 3]))


save_path = "./saved_my_dir"
tf.saved_model.save(mod, save_path)
