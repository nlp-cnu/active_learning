import os
from datetime import datetime

# Remove excessive tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Recall, Precision
from transformers import TFAutoModel, AutoTokenizer

# Fix TF for my computer
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Transformer models
BIO_BERT = 'cambridgeltl/BioRedditBERT-uncased'
BERTWEET = 'vinai/bertweet-base'


class PositiveF1Score(tf.keras.metrics.Metric):

    def __init__(self, name='positive_f1', **kwargs):
        super(PositiveF1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision(class_id=1)
        self.recall = Recall(class_id=1)
        self.f1 = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
        precision = self.precision.result()
        recall = self.recall.result()
        self.f1 = 2 * ((precision * recall) / (precision + recall))

    def result(self):
        return self.f1

    def reset_states(self):
        self.recall.reset_states()
        self.precision.reset_states()
        self.f1 = 0


class Classifier:
    def __init__(self, load_path=None, model_name=None):
        """
        Makes a classification model
        :param load_path: .h5 file of model weights
        :param model_name: save name for the model
        """
        if model_name is None:
            model_name = datetime.now().strftime('%m-%d_%H-%M-%S')
        self.model_name = model_name

        self.bio_tokenizer = AutoTokenizer.from_pretrained(BIO_BERT)
        self.bio_encoder = TFAutoModel.from_pretrained(BIO_BERT)
        self.bio_encoder.trainable = False

        # self.tweet_tokenizer = AutoTokenizer(BERTWEET)
        # self.tweet_encoder = TFAutoModel.from_pretrained(BERTWEET)
        # self.tweet_encoder.trainable = False

        self.model = self.__make_model()

        if load_path is not None:
            self.load(load_path)

    @staticmethod
    def __make_classifier():
        """
        Creates the actual classification layer
        :return:
        """
        dropout = 0.5
        classifier = Sequential()

        classifier.add(Bidirectional(LSTM(
            units=128,
            return_sequences=True,
        )))
        classifier.add(Dropout(dropout))

        classifier.add(Bidirectional(LSTM(
            units=128,
            return_sequences=False,
        )))
        classifier.add(Dropout(dropout))

        classifier.add(Dense(
            units=64,
            activation='gelu'
        ))
        classifier.add(Dropout(dropout))

        classifier.add(Dense(
            units=64,
            activation='gelu'
        ))
        classifier.add(Dropout(dropout))

        classifier.add(Dense(
            units=32,
            activation='gelu'
        ))
        classifier.add(Dropout(dropout))

        classifier.add(Dense(
            units=2,
            activation='softmax'
        ))

        return classifier

    def __make_model(self):
        """
        Creates a classification model with a BERT embedding layer
        :return:
        """
        input_ids = Input(shape=(None,), dtype=tf.int32)
        attention_mask = Input(shape=(None,), dtype=tf.int32)

        bio_embedding = self.bio_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        # todo: handle different input id types for ensemble of different BERT models
        # tweet_embedding = self.tweet_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        # classifier_input = concatenate(bio_embedding, tweet_embedding)

        self.classifier = self.__make_classifier()
        output = self.classifier(bio_embedding)

        model = Model(inputs=[input_ids, attention_mask], outputs=output)

        model.compile(
            optimizer='nadam',
            loss='categorical_crossentropy',
            metrics=[
                PositiveF1Score()
            ]
        )

        return model

    def format_input(self, x):
        """
        tokenizes raw string input
        :param x: batch of raw strings
        :return: tokenized ids, attention mask
        """
        if not isinstance(x, list):
            x = list(x)
        bio_tokenized = self.bio_tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors='tf')
        return bio_tokenized['input_ids'], bio_tokenized['attention_mask']

    @staticmethod
    def __scheduler(epoch, learning_rate):
        """
        Learning rate scheduler, reduces learning rate over time
        :param epoch:
        :param learning_rate:
        :return: updated learning rate
        """
        return max(learning_rate * np.exp(0.001 * -epoch), 0.00001)

    def fit(self, x, y):
        """
        Trains the model
        :param x: data
        :param y: labels
        """

        # set class weights for loss function
        # attempts to make model pay more attention to classes w/ less samples
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y.argmax(axis=1)
        )
        class_weights = dict(enumerate(class_weights))

        self.model.fit(
            self.format_input(x), y,
            epochs=15,
            batch_size=100,
            # validation_split=0.1,
            class_weight=class_weights,
            callbacks=[
                TensorBoard(os.path.join('..', 'logs', self.model_name)),
                # LearningRateScheduler(self.__scheduler),
                # ModelCheckpoint(os.path.join('..', 'models', 'checkpoints', self.model_name + '.h5'),
                #                 save_best_only=True, save_weights_only=True),
            ]
        )

        self.save(os.path.join('..', 'models', 'trained', self.model_name + '.h5'))

    def predict(self, x):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        return self.model.predict(x)

    def test(self, x, y_true, print_report=False):
        """
        performs an F1 test on the test data for the positive class
        :param x: data
        :param y_true: true labels
        :param print_report: whether to print a classification report from the results. Default = False
        :return: f1 score for positive class
        """
        y_true = y_true.argmax(axis=1)
        y_pred = self.predict(self.format_input(x)).argmax(axis=1)
        if print_report:
            print(classification_report(y_true, y_pred, zero_division=True))

        return f1_score(y_true, y_pred, pos_label=1)

    def save(self, filepath):
        """
        Saves the model weights
        :param filepath:
        :return:
        """
        self.model.save_weights(filepath)

    def load(self, filepath):
        """
        Loads weights for the model
        :param filepath:
        :return:
        """
        self.model.load_weights(filepath)


if __name__ == '__main__':
    a = Classifier()
