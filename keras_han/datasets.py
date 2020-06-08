import csv
import gzip
import json

import tensorflow as tf
import transformers


class TvTropesMovieSingleDataset:
    DATA_SOURCES = {
        "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-train.balanced.csv",
        "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-dev1.balanced.csv",
        "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-test.balanced.csv",
    }

    def get_dataset(self, dataset_type):
        X = list()
        y = list()
        with open(transformers.cached_path(self.DATA_SOURCES[dataset_type])) as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for sentence, spoiler, verb, page, trope in reader:
                label = 1 if spoiler == "True" else 0
                X.append(sentence)
                y.append(label)

        return X, y


class TvTropesBookSingleDataset:
    DATA_SOURCES = {
        "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-train.json.gz",
        "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-val.json.gz",
        "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-test.json.gz",
    }

    def get_dataset(self, dataset_type):
        X = list()
        y = list()
        with gzip.open(
            transformers.cached_path(self.DATA_SOURCES[dataset_type])
        ) as file:
            for line in file:
                tropes_json = json.loads(line)
                sentences, labels = list(), list()
                for spoiler, sentence, _ in tropes_json["sentences"]:
                    label = 1 if spoiler == True else 0
                    X.append(sentence)
                    y.append(label)

        return X, y
