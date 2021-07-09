import os

import numpy as np
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight


class Dataset:
    def __init__(self, seed=3):  # use_all_data=False,
        self.seed = seed
        self.label_encoder = OneHotEncoder(sparse=False)

        def preprocess_data(data):
            # preprocess tweets to remove mentions, URL's
            p.set_options(p.OPT.MENTION, p.OPT.URL)  # P.OPT.HASHTAG, p.OPT.EMOJI
            data = data.apply(p.clean)

            # Tokenize special Tweet characters
            # p.set_options(p.OPT.NUMBER)  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED,
            # data = data.apply(p.tokenize)

            return data.tolist()

        # read in data
        data_filepath = os.path.join('..', 'data', 'full_dataset.tsv')
        df = pd.read_csv(data_filepath, header=None, names=['Tweet', 'Label'], delimiter='\t').dropna()
        data = preprocess_data(df['Tweet'])
        labels = self.label_encoder.fit_transform(df['Label'].values.reshape(-1, 1))

        # Split data
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(data, labels,
                                                                                test_size=0.2,
                                                                                random_state=self.seed)

        # determine class weights
        self.class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_Y),
            y=self.train_Y.argmax(axis=1)
        )
        self.class_weights = dict(enumerate(self.class_weights))

    def get_train_data(self):
        return self.train_X.copy()[:1000], self.train_Y.copy()[:1000]

    def get_train_class_weights(self):
        return self.class_weights

    def get_test_data(self):
        return self.test_X.copy(), self.test_Y.copy()


if __name__ == '__main__':
    dataset = Dataset()
    print(len(dataset.get_train_data()[0]))
    print(dataset.get_train_class_weights())
