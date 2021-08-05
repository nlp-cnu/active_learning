import os

import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class Dataset:
    def __init__(self, data_filepath=None, seed=3):  # use_all_data=False,
        self.seed = seed
        self.label_encoder = OneHotEncoder(sparse=False)

        def preprocess_data(data):
            # preprocess tweets to remove mentions, URL's
            p.set_options(p.OPT.MENTION, p.OPT.URL)  # p.OPT.HASHTAG, p.OPT.EMOJI
            data = data.apply(p.clean)

            # Tokenize special Tweet characters
            # p.set_options(p.OPT.NUMBER)  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED,
            # data = data.apply(p.tokenize)

            return data.tolist()

        # read in data
        if data_filepath is None:
            data_filepath = os.path.join('..', 'data', 'full_dataset.tsv')

        df = pd.read_csv(data_filepath, delimiter='\t').dropna()
        data = preprocess_data(df['sentence'])

        if 'ISEAR' in data_filepath:
            labels = [['other'] if label != 'fear' else [label] for label in df['label'].values]
        else:
            labels = df['label'].values.reshape(-1, 1)

        labels = self.label_encoder.fit_transform(labels)

        # Split data
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(data, labels,
                                                                                test_size=0.2,
                                                                                random_state=self.seed)

    def get_train_data(self):
        return self.train_X.copy(), self.train_Y.copy()

    def get_train_size(self):
        return len(self.test_Y)

    def get_test_data(self):
        return self.test_X.copy(), self.test_Y.copy()


if __name__ == '__main__':
    data_filepath = os.path.join('..', 'data', 'full_dataset.tsv')
    dataset = Dataset(data_filepath=data_filepath)

    print(dataset.label_encoder.inverse_transform([[0, 1]]))

    # x, y = dataset.get_train_data()
    # print(x[:100])
