import os

import numpy as np
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight


class Dataset:
    def __init__(self, data_filepath=os.path.join('..', 'data', 'tweet_data.txt'), seed=3):
        self.seed = seed
        # self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        self.label_encoder = OneHotEncoder(sparse=False)

        # Read in dataset
        df = pd.read_csv(data_filepath, header=None, names=['Tweet', 'Label'], delimiter='\t').dropna()
        data = df['Tweet']

        # preprocess tweets to remove mentions, URL's
        p.set_options(p.OPT.MENTION, p.OPT.URL)  # P.OPT.HASHTAG
        data = data.apply(p.clean)

        # Tokenize special Tweet characters
        # p.set_options(p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED, p.OPT.NUMBER)
        # data = data.apply(p.tokenize)

        data = data.tolist()

        # One Hot encode labels
        labels = self.label_encoder.fit_transform(df['Label'].values.reshape(-1, 1))

        # Split data
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(data, labels,
                                                                                test_size=0.2, random_state=self.seed)

    def get_train_data(self):
        return self.train_X, self.train_Y

    def get_test_data(self):
        return self.test_X, self.test_Y

    def get_active_set(self):
        labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(self.train_X, self.train_Y,
                                                                          test_size=0.8, random_state=self.seed)

        return (labeled_x, labeled_y), (unlabeled_x, unlabeled_y)


if __name__ == '__main__':
    dataset = Dataset()
    x, y = dataset.get_train_data()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=np.argmax(y, axis=1))
    class_weights = dict(enumerate(class_weights))

    print(class_weights)

