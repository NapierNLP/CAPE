# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import os
import json
import datasets
from utils import Utils
import pandas as pd
from tensorflow import keras


_DESCRIPTION = """\
Trustpilot dataset.
"""

_CITATION = """\
@inproceedings{Hovy,
author = {Hovy, Dirk and Johannsen, Anders and S{\o}gaard, Anders},
booktitle = {WWW '15: Proceedings of the 24th International Conference on World Wide Web},
doi = {10.1145/2736277.2741141},
isbn = {9781450334693},
pages = {452--461},
title = {{User Review Sites as a Resource for Large-Scale Sociolinguistic Studies}},
url = {https://doi.org/10.1145/2736277.2741141},
year = {2015}
}
"""

_HOMEPAGE = "https://bitbucket.org/lowlands/release/src/master/"

_LICENSE = ""

_URLs = {
    'uk': "https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.zip",
    'us': "https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/united_states.auto-adjusted_gender.geocoded.jsonl.zip"
}


class TrustDataset(datasets.GeneratorBasedBuilder):
    
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="uk", version=VERSION, description="UK results."),
        datasets.BuilderConfig(name="us", version=VERSION, description="US results.")
    ]

    DEFAULT_CONFIG_NAME = "uk"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.Value("int16"),
                "priv_label": datasets.Value("int16")
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        for filename in os.listdir(data_dir):
            if filename.endswith('.jsonl.tmp'):
                clean = Utils.clean_file(os.path.join(data_dir, filename))
                df = pd.DataFrame(clean)
                df['rating'] = df['rating'] - 1
                df['gender'] = df['gender'].astype('category').cat.codes
                with open(os.path.join(data_dir, "data.json"), encoding='utf-8', mode='w') as f:
                    df.to_json(f, orient='records', lines=True)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data.json"),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "text": data["text"],
                    "label": data["rating"],
                    "priv_label": data["gender"],
                }