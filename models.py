from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, pipeline


class StandardClassifier:

    def __init__(self, n_labels, hidden=256, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        embed_model = TFBertModel.from_pretrained('bert-base-uncased')
        for layer in embed_model.layers:
            layer.trainable = False

        input_ids = layers.Input(shape=(max_length,), dtype='int32')
        attention_mask = layers.Input(shape=(max_length,), dtype='int32')
        embedding_layer = embed_model(input_ids, attention_mask=attention_mask).last_hidden_state
        cls_token = embedding_layer[:, 0, :]
        linear = layers.Dense(hidden, activation='relu')(cls_token)
        preds = layers.Dense(n_labels, activation='softmax')(linear)
        self.model = keras.Model(
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            outputs=preds,
            name='base_classifier')
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

        self.embedder = pipeline('feature-extraction',
                                 tokenizer=self.tokenizer,
                                 model=embed_model)


class AttackerClassifier:

    def __init__(self, dim, n_labels, hidden=256):
        inputs = layers.Input(dim)
        linear = layers.Dense(hidden, activation='relu')(inputs)
        outputs = layers.Dense(n_labels, activation='softmax')(linear)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='attacker_model')
        self.model.compile(optimizer=keras.optimizers.SGD(),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])