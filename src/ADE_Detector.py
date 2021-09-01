import os
from datetime import datetime

# Remove excessive tf log messages
from keras.optimizer_v2.adam import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import *
from transformers import TFAutoModel, AutoTokenizer

from Utils import *

# Fix TF for my computer (enable memory growth)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DALStopping(Callback):
    def __init__(self, monitor='accuracy', target_score=0.90):
        """
        Early stopping for DAL classifier.
        :param monitor: Metric to monitor. Defaults to accuracy.
        :param target_score: Target value for metric. Training stops when target is met or exceeded.
        """
        super().__init__()
        self.monitor = monitor
        self.goal_score = target_score
        self.verbose = 0

    def on_epoch_end(self, epoch, logs=None):

        # Only stop after 1st epoch
        if epoch == 0:
            return

        quantity = self.__get_monitor_value(logs)
        if quantity >= self.goal_score:
            self.model.stop_training = True

    def __get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            raise ValueError

        return monitor_value


class EarlyStoppingGreaterThanZero(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):

        # Only check after 1st epoch
        if epoch == 0:
            return

        quantity = self.get_monitor_value(logs)
        if quantity == 0:
            return

        super().on_epoch_end(epoch, logs)


class SingleClassF1(tf.keras.metrics.Metric):
    def __init__(self, name="positive_class_F1", class_id=1, **kwargs):
        """
        Custom F1 score metric. Designed to monitor the F1 score of a particular class.
        :param name: Name of metric.
        :param class_id: Index of class to monitor.
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)
        self.recall = tf.keras.metrics.Recall(class_id=class_id)
        self.precision = tf.keras.metrics.Precision(class_id=class_id)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.recall.update_state(y_true, y_pred, sample_weight)
        self.precision.update_state(y_true, y_pred, sample_weight)

    def result(self):
        r = self.recall.result()
        p = self.precision.result()

        if tf.equal(r, 0.0) and tf.equal(p, 0.0):
            return tf.convert_to_tensor(0.0, tf.float32)

        return 2 * ((p * r) / (p + r))

    def reset_state(self):
        self.recall.reset_states()
        self.precision.reset_states()


class ADE_Detector:
    class __DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, x_set, y_set, batch_size, bert_model=BERT_BASE, shuffle=True, seed=None):
            """

            :param x_set:
            :param y_set:
            :param batch_size:
            :param bert_model:
            :param shuffle:
            """
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.rng = np.random.default_rng(seed=seed)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model, cache_dir='BERT_tokenizer')

        def __len__(self):
            return int(np.ceil(len(self.x) / self.batch_size))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            tokenized = self.tokenizer(batch_x, padding=True, truncation=True, max_length=MAX_TOKEN_LENGTH,
                                       return_tensors='tf')
            batch_x = (tokenized['input_ids'], tokenized['attention_mask'])

            # print the tokenized tokens
            # for i, t in enumerate(tokenized['input_ids']):
            #     print("string = ", batch_x[i])
            #     print("tokens = ", self.tokenizer.convert_ids_to_tokens(t))

            if self.y is None:
                return batch_x

            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return batch_x, batch_y

        def on_epoch_end(self):
            """
            Shuffle data
            :return:
            """
            if self.shuffle:
                idxs = np.arange(len(self.x))
                self.rng.shuffle(idxs)
                self.x = [self.x[idx] for idx in idxs]
                self.y = self.y[idxs]

    def __init__(self, model_name=None, bert_model=ROBERTA_TWITTER, dropout_rate=DROPOUT,
                 num_lstm=1, lstm_size=128,
                 num_dense=1, dense_size=64, dense_activation='gelu',
                 optimizer=Adam(learning_rate=LR),
                 monitor_idx=1
                 ):
        """

        :param model_name:
        :param bert_model:
        :param dropout_rate:
        :param num_lstm:
        :param lstm_size:
        :param num_dense:
        :param dense_size:
        :param dense_activation:
        :param optimizer:
        """

        self.monitor_idx = monitor_idx

        self.model_name = model_name if model_name is not None else datetime.now().strftime('%m-%d_%H-%M-%S')

        # classifier options
        self.dropout_rate = dropout_rate
        self.num_lstm, self.lstm_size = num_lstm, lstm_size
        self.num_dense, self.dense_size, self.dense_activation = num_dense, dense_size, dense_activation

        self.bert_model = bert_model
        self.bert = TFAutoModel.from_pretrained(self.bert_model, cache_dir='BERT_model')
        self.bert.trainable = TRAIN_BERT

        self.optimizer = optimizer
        self.model = self.__make_model()

    def __make_classifier(self):
        """
        Creates the actual classification layer
        :return:
        """
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        classifier = Sequential()

        for layer in range(self.num_lstm):
            classifier.add(Bidirectional(LSTM(
                units=self.lstm_size,
                return_sequences=layer != (self.num_lstm - 1),
            )))
            classifier.add(Dropout(self.dropout_rate))

        for _ in range(self.num_dense):
            classifier.add(Dense(
                units=self.dense_size,
                activation=self.dense_activation,
            ))
            classifier.add(Dropout(self.dropout_rate))

        classifier.add(Dense(
            units=2,
            activation='softmax',
        ))

        return classifier

    def __make_model(self):
        """
        Creates a classification model with a BERT embedding layer
        :return:
        """

        self.classifier = self.__make_classifier()

        input_ids = Input(shape=(None,), dtype=tf.int32)
        attention_mask = Input(shape=(None,), dtype=tf.int32)
        embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.classifier(embedding)

        model = Model(inputs=[input_ids, attention_mask], outputs=output)

        model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                SingleClassF1(class_id=self.monitor_idx),
                # tf.keras.metrics.Recall(class_id=1),
                # tf.keras.metrics.Precision(class_id=1)
            ]
        )

        return model

    def fit(self, x: list, y: np.ndarray, val_data=None, epochs=EPOCHS, use_class_weights=True):
        """
        Trains a model.
        :param x: Inputs
        :param y: Expected outputs
        :param val_data: validation data
        :param epochs: number of epochs to train for
        :param use_class_weights: whether to use class weights or not
        :return:
        """

        train_data = self.__DataGenerator(x, y, BATCH_SIZE, bert_model=self.bert_model, seed=SEED)
        verbose = 2

        callbacks = [
            # TensorBoard(os.path.join('..', 'logs', self.model_name)),
            # ModelCheckpoint(os.path.join('..', 'models', 'checkpoints', 'temp'), save_best_only=True),
        ]

        if val_data is not None:
            val_data = self.__DataGenerator(val_data[0], val_data[1], BATCH_SIZE, bert_model=self.bert_model)
            monitor = 'val_positive_class_F1'
            mode = 'max'
            min_delta = 0.001
            patience = 5
            callbacks.append(
                EarlyStoppingGreaterThanZero(
                    monitor=monitor,
                    mode=mode,
                    min_delta=min_delta,
                    patience=patience,
                    restore_best_weights=True
                )
            )

        else:
            verbose = 0
            callbacks.append(
                DALStopping()
            )

        class_weights = None
        if use_class_weights:
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y),
                y=y.argmax(axis=1)
            )
            class_weights = dict(enumerate(class_weights))

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, x):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """

        batch_size = None

        if not isinstance(x, tf.keras.utils.Sequence):
            tokenizer = self.__DataGenerator(x, None, None).tokenizer
            tokenized = tokenizer(x, padding=True, truncation=True, max_length=MAX_TOKEN_LENGTH, return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])
            batch_size = BATCH_SIZE // 2

        return self.model.predict(x, batch_size=batch_size, verbose=0)

    def eval(self, x, y_true):
        """
        performs an F1 test on the test data for the positive class
        :param x: data
        :param y_true: true labels
        :return: f1 score for positive class
        """

        x = self.__DataGenerator(x, y_true, BATCH_SIZE, bert_model=self.bert_model, seed=SEED)
        y_true = y_true.argmax(axis=1)
        y_pred = self.predict(x).argmax(axis=1)

        print(classification_report(y_true, y_pred, zero_division=True))

        return f1_score(y_true, y_pred, pos_label=self.monitor_idx)

    def save(self, filepath=None):
        """
        Saves the model weights
        :param filepath:
        :return:
        """

        if filepath is None:
            filepath = os.path.join('..', 'models', 'trained', self.model_name)

        self.model.save_weights(filepath)

    def load(self, filepath):
        """
        Loads weights for the model
        :param filepath:
        :return:
        """
        self.model.load_weights(filepath)

    def reset_model(self):
        """
        Reset the model's learned weights
        :return:
        """
        self.model = self.__make_model()


if __name__ == '__main__':
    a = ADE_Detector()
