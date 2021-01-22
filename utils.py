import ast
import tensorflow as tf
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer
import logging


def clean_file(file):
    clean_list = []
    if file.endswith('.jsonl.tmp'):
        with open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                data = ast.literal_eval(line)
                if 'gender' in data.keys() and data['gender']:
                    for review in data['reviews']:
                        if review['rating'] and review['text']:
                            new_data = {
                                'text': review['text'][0],
                                'rating': int(review['rating']),
                                'gender': data['gender']
                            }
                            clean_list.append(new_data)
    return clean_list


def get_tf_dataset(ds, n_labels, n_priv_labels, args):
    """
    Return a tf.data.Dataset from a datasets.Dataset.
    :param ds: datasets.Dataset to copy
    :param n_labels: number of possible labels
    :param n_priv_labels: number of possible labels for private variable
    :param args: program arguments
    :return: tf.data.Dataset
    """
    features = {x: ds[x].to_tensor(default_value=0, shape=[None, args.max_length])
                for x in ['input_ids', 'attention_mask']}
    labels = tf.keras.utils.to_categorical(ds['label'], num_classes=n_labels)
    priv_labels = tf.keras.utils.to_categorical(ds['priv_label'], num_classes=n_priv_labels)
    label_dict = {"base": labels, "attacker": priv_labels}
    dataset = tf.data.Dataset.from_tensor_slices((features, label_dict)).batch(args.batch_size).prefetch(1)
    return dataset


def get_task_data(args):
    """
    Return a set of tf.data.Dataset objects containing the data and labels loaded for the selected task.
    :param args: program arguments
    :return: train dataset, test dataset, validation dataset, number of labels, number of private variable labels
    """
    data = load_dataset('trust_dataset.py',
                        'uk',
                        cache_dir=args.cache_dir,
                        split=f'train{f"[:{args.data_split}]" if args.data_split else ""}')
    train_test = data.train_test_split(train_size=args.train_split, shuffle=not args.no_shuffle)
    if args.validate:
        train_valid = train_test['train'].train_test_split(test_size=args.val_split)
        data = DatasetDict({
            'train': train_valid['train'].flatten_indices(),
            'test': train_test['test'].flatten_indices(),
            'val': train_valid['test'].flatten_indices()
        })
    else:
        data = DatasetDict({
            'train': train_test['train'].flatten_indices(),
            'test': train_test['test'].flatten_indices()
        })
    n_labels = len(data['train'].unique('label'))
    n_priv_labels = len(data['train'].unique('priv_label'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir, do_lower_case=True)
    logging.debug("Applying tokenization...")
    data = data.map(lambda e: tokenizer(e['text'],
                                        truncation=True,
                                        padding='max_length',
                                        max_length=args.max_length,
                                        add_special_tokens=True,
                                        return_token_type_ids=False),
                    batched=True)
    logging.debug("Tokenization complete.")
    logging.debug("Creating TF datasets...")
    data.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label', 'priv_label'])
    train_ds = get_tf_dataset(data['train'], n_labels, n_priv_labels, args)
    test_ds = get_tf_dataset(data['test'], n_labels, n_priv_labels, args)
    if args.validate:
        val_ds = get_tf_dataset(data['val'], n_labels, n_priv_labels, args)
    else:
        val_ds = None
    logging.debug("Datasets created.")
    return train_ds, test_ds, val_ds, n_labels, n_priv_labels
