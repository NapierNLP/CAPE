from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import reduce_mean
from transformers import TFBertModel, BertConfig
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from flipGradientTF import GradientReversal


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
        self.dropout = args.dropout_rate
        self.adversarial = args.adversarial
        self.hplambda = args.hplambda

    def get_classifier(self, n_labels: int, n_priv_labels: int, name: str):
        """
        Return an end-to-end classifier that takes a set of tokenized inputs and learns to predict labels.
        :param n_labels: number of possible label values to predict
        :param n_priv_labels: number of possible values of private information label
        :param name: name to attach to the model
        :return: keras.Model instance
        """
        config = BertConfig(return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            use_cache=False,
                            hidden_size=self.embed_length)
        embed_model = TFBertModel.from_pretrained('bert-base-uncased', config=config, cache_dir=self.cache)
        for layer in embed_model.layers:
            layer.trainable = False

        input_ids = layers.Input(shape=(self.max_length,), dtype='int32')
        attention_mask = layers.Input(shape=(self.max_length,), dtype='int32')
        embedding = embed_model(input_ids, attention_mask=attention_mask).last_hidden_state
        x = reduce_mean(embedding, axis=1, name='mean_embedding')
        if self.dropout:
            x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(self.embed_length, activation='relu', name='feature_learn')(x)

        # target task classifier
        linear = layers.Dense(self.hidden, activation='relu')(x)
        preds = layers.Dense(n_labels, activation='softmax', name="base")(linear)

        # adversary classifier
        if self.adversarial:
            x = GradientReversal(self.hplambda)(x)
        a_linear = layers.Dense(self.hidden, activation='relu')(x)
        a_preds = layers.Dense(n_priv_labels, activation='softmax', name="attacker")(a_linear)

        model = keras.Model(inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                            outputs=[preds, a_preds],
                            name=name)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss={
                          "base": keras.losses.CategoricalCrossentropy(name='loss'),
                          "attacker": keras.losses.CategoricalCrossentropy(name='loss')
                      },
                      metrics={
                          "base": [CategoricalAccuracy(name='acc'),
                                   Precision(name='prec'),
                                   Recall(name='rec'),
                                   AUC(name='auc')],
                          "attacker": [CategoricalAccuracy(name='acc'),
                                       Precision(name='prec'),
                                       Recall(name='rec'),
                                       AUC(name='auc')]
                      })

        return model
