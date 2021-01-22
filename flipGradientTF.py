# Courtesy of Michele Tonutti and Berton Earnshaw, updated by author
# https://github.com/michetonu/gradient_reversal_keras_tf
# As proposed in Ganin et al. "Domain-adversarial training of neural networks"
# @article{10.5555/2946645.2946704,
#   author = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and
#       Larochelle, Hugo and Laviolette, Fran\c{c}ois and Marchand, Mario and Lempitsky, Victor},
#   title = {Domain-Adversarial Training of Neural Networks},
#   year = {2016},
#   issue_date = {January 2016},
#   publisher = {JMLR.org},
#   volume = {17},
#   number = {1},
#   issn = {1532-4435},
#   journal = {J. Mach. Learn. Res.},
#   month = jan,
#   pages = {2096–2030},
#   numpages = {35},
# }

import tensorflow as tf
from tensorflow.keras.layers import Layer


class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda
        self.trainable = False

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -dy * self.hp_lambda

        return y, custom_grad

    def call(self, x, mask=None):
        return self.grad_reverse(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))