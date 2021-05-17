from tensorflow import keras, constant
from tensorflow.keras import layers
from tensorflow import reduce_mean, reduce_min, reduce_max, reshape
from transformers import TFBertModel, BertConfig, logging
from flipGradientTF import GradientReversal
from tensorflow_probability import distributions as tfd
import tensorflow_addons as tfa


class CheckpointCallback(keras.callbacks.ModelCheckpoint):
    def on_train_batch_end(self, batch, logs=None):
        logs['batch'] = batch
        super(CheckpointCallback, self).on_train_batch_end(batch, logs)


class ClassifierBuilder:
    def __init__(self, args):
        self.hidden = args.hidden_size
        self.embed_length = args.embed_length
        self.max_length = args.max_length
        self.cache = args.cache_dir
        self.dropout = args.dropout
        self.adversarial = args.adversarial
        self.hplambda = args.hplambda
        self.dp = args.dp
        self.epsilon = args.epsilon
        self.balance = args.balance
        self.learning_rate = args.learning_rate
        self.identifier = args.identifier
        self.labels = args.labels
        self.priv_labels = args.priv_labels

    def get_classifier(self, name: str):
        """
        Return an end-to-end classifier that takes a set of tokenized inputs and learns to predict labels.
        :param name: name to attach to the model
        :return: keras.Model instance
        """
        logging.set_verbosity_error()
        config = BertConfig(return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            use_cache=False,
                            hidden_size=self.embed_length)
        embed_model = TFBertModel.from_pretrained('bert-base-uncased',
                                                  config=config,
                                                  cache_dir=self.cache,
                                                  local_files_only=True)
        for layer in embed_model.layers:
            layer.trainable = False

        input_ids = layers.Input(shape=(self.max_length,), dtype='int32')
        attention_mask = layers.Input(shape=(self.max_length,), dtype='int32')
        sequence_embedding = embed_model(input_ids, attention_mask=attention_mask).last_hidden_state
        x = reduce_mean(sequence_embedding, axis=1, name='mean_embedding')

        if self.dp:
            embed_min = reduce_min(x, keepdims=True)
            embed_max = reduce_max(x, keepdims=True)
            x = (x - embed_min) / (embed_max - embed_min)
            print(x)
            noise = tfd.Laplace(constant([0.0]), constant([1.0 / self.epsilon]))
            noise_s = noise.sample(sample_shape=self.embed_length)
            x += reshape(noise_s, shape=(-1))

        features_i = layers.Dense(self.hidden, activation='relu')(x)
        if self.dropout:
            features_i = layers.Dropout(self.dropout)(features_i)

        features = layers.Dense(self.hidden, activation='relu')(features_i)

        # target task classifier
        linear = layers.Dense(self.hidden, activation='relu')(features)
        if self.dropout:
            linear = layers.Dropout(self.dropout)(linear)
        preds = layers.Dense(len(self.labels), activation='softmax', name="base")(linear)

        # adversary classifier
        if self.adversarial:
            reversal = GradientReversal(hp_lambda=self.hplambda)(features)
            a_linear = layers.Dense(self.hidden,
                                    activation='relu')(reversal)
        else:
            a_linear = layers.Dense(self.hidden,
                                    activation='relu')(features)
        if self.dropout:
            a_linear = layers.Dropout(self.dropout)(a_linear)
        if self.identifier == 'gender':
            a_preds = layers.Dense(1, activation='sigmoid', name="attacker")(a_linear)
        else:
            a_preds = layers.Dense(len(self.priv_labels), activation='softmax', name='attacker')(a_linear)

        model = keras.Model(inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                            outputs={"base": preds, "attacker": a_preds},
                            name=name)
        if self.identifier == 'gender':
            a_loss = keras.losses.BinaryCrossentropy()
            a_metrics = [
                keras.metrics.BinaryAccuracy(name='acc'),
                tfa.metrics.F1Score(name='f1', average="micro", num_classes=2, threshold=0.5)
            ]
        else:
            a_loss = keras.losses.CategoricalCrossentropy()
            a_metrics = [
                keras.metrics.CategoricalAccuracy(name='acc'),
                tfa.metrics.F1Score(name='f1', average='weighted', num_classes=len(self.priv_labels))
            ]
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss={
                          "base": keras.losses.CategoricalCrossentropy(),
                          "attacker": a_loss
                      },
                      metrics={
                          "base": [
                              keras.metrics.CategoricalAccuracy(name='acc'),
                              tfa.metrics.F1Score(name='f1', average='weighted', num_classes=len(self.labels))
                          ],
                          "attacker": a_metrics
                      })

        return model
