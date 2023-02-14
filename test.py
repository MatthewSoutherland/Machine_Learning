import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

matplotlib.rcParams["figure.figsize"] = [9, 6]

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)


def f(x):
    y = x**2 + 2 * x - 5
    return y


y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), ".", label="Data")
plt.plot(x, f(x), label="Ground truth")
plt.legend()


class Model(tf.Module):
    def __init__(self):
        # Randomly generate weight and bias terms
        rand_init = tf.random.uniform(shape=[3], minval=0.0, maxval=5.0, seed=22)
        # Initialize model parameters
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])

    @tf.function
    def __call__(self, x):
        # Quadratic Model : quadratic_weight * x^2 + linear_weight * x + bias
        return self.w_q * (x**2) + self.w_l * x + self.b


quad_model = Model()


def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x, y, ".", label="Data")
    plt.plot(x, f(x), label="Ground truth")
    plt.plot(x, model(x), label="Predictions")
    plt.title(title)
    plt.legend()


plot_preds(x, y, f, quad_model, "Before training")
