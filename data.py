import tensorflow as tf
import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import islice
import flatten_dict


ALLOWED_KEYS = ['location', 'user_id', 'country', 'reviews_0_text_0', 'birth_year']
LABEL_KEY = 'reviews_0_rating'
PRIV_KEY = 'gender'


class TrustDataset:
    def __init__(self, folder_name, n_labels, train_split=0.7):
        self.data, self.labels, self.privs = self.get_data(folder_name, n_labels)
        self.train_data, self.test_data, self.train_labels, self.test_labels, \
            self.train_privs, self.test_privs = self.train_test_split(train_split)

    def get_text(self, file):
        data = []
        labels = []
        privs = []
        with open(file, encoding='utf-8', mode='r') as f:
            for line in list(islice(f, 10_000)):
                flat = flatten_dict.flatten(ast.literal_eval(line), reducer='underscore', enumerate_types=(list,))
                if flat[LABEL_KEY] and PRIV_KEY in flat.keys():
                    filtered = {k: flat[k] for k in flat.keys() if k in ALLOWED_KEYS}
                    data.append(filtered)
                    labels.append(flat[LABEL_KEY])
                    privs.append(flat[PRIV_KEY])
        return data, labels, privs

    def get_data(self, folder, n_labels):
        all_data = []
        all_labels = []
        all_privs = []
        for filename in os.listdir(f'./{folder}'):
            if filename.endswith('.jsonl'):
                data, labels, privs = self.get_text(f'./{folder}/{filename}')
                all_data.extend(data)
                all_labels.extend(labels)
                all_privs.extend(privs)

        data_df = pd.DataFrame(all_data)
        data_df = data_df.rename(columns={'reviews_0_text_0': 'text'}, errors="raise")
        all_privs = pd.get_dummies(pd.DataFrame(all_privs)).to_numpy()
        all_labels = tf.keras.utils.to_categorical([int(x)-1 for x in all_labels], num_classes=n_labels)
        return data_df, all_labels, all_privs

    def train_test_split(self, train_split):
        return train_test_split(self.data, self.labels, self.privs, train_size=train_split, shuffle=True)
