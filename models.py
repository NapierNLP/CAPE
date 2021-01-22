from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel, BertConfig


class ClassifierBuilder:

    def __init__(self, args):
        self.hidden = args.hidden_size
        self.embed_length = args.embed_length
        self.max_length = args.max_length
        self.cache = args.cache_dir

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
        pooled = embedding[:, 0, :]
        learnable = layers.Dense(self.embed_length, activation='relu')(pooled)

        # target task classifier
        linear = layers.Dense(self.hidden, activation='relu')(learnable)
        preds = layers.Dense(n_labels, activation='softmax', name="base")(linear)

        # adversary classifier
        a_linear = layers.Dense(self.hidden, activation='relu')(learnable)
        a_preds = layers.Dense(n_priv_labels, activation='softmax', name="attacker")(a_linear)

        model = keras.Model(inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                            outputs=[preds, a_preds],
                            name=name)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss={
                          "base": keras.losses.CategoricalCrossentropy(),
                          "attacker": keras.losses.CategoricalCrossentropy()
                      },
                      metrics={
                          "base": ['accuracy',
                                   keras.metrics.Precision(name='base_precision'),
                                   keras.metrics.Recall(name='base_recall')],
                          "attacker": ['accuracy',
                                       keras.metrics.Precision(name='attacker_precision'),
                                       keras.metrics.Recall(name='attacker_recall')]
                      })

        return model

