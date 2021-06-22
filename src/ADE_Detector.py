import os
from datetime import datetime

# Remove excessive tf log messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from transformers import TFAutoModel, AutoTokenizer

# Fix TF for my computer (enable memory growth)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Hyper parameters
DROPOUT = 0.5
EPOCHS = 100
BATCH_SIZE = 500
BASEBERT = 'bert-base-uncased'
ROBERTA_TWITTER = 'cardiffnlp/twitter-roberta-base'
BIOREDDITBERT = 'cambridgeltl/BioRedditBERT-uncased'
SEED = 2005


class PositiveClassF1(tf.keras.metrics.Metric):
    def __init__(self, name="positive_class_F1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.recall = tf.keras.metrics.Recall(class_id=1)
        self.precision = tf.keras.metrics.Precision(class_id=1)

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
        def __init__(self, x_set, y_set, batch_size, bert_model=BASEBERT, shuffle=True):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        def __len__(self):
            return int(np.ceil(len(self.x) / self.batch_size))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            tokenized = self.tokenizer(batch_x, padding=True, truncation=True, max_length=512, return_tensors='tf')

            return (tokenized['input_ids'], tokenized['attention_mask']), batch_y

        def on_epoch_end(self):
            """
            Shuffle data
            :return:
            """
            if self.shuffle:
                idxs = np.arange(len(self.x))
                np.random.shuffle(idxs)
                self.x = [self.x[idx] for idx in idxs]
                self.y = self.y[idxs]

    def __init__(self, model_name=None, bert_model=BASEBERT, dropout_rate=DROPOUT,
                 num_lstm=1, lstm_size=512,
                 num_dense=3, dense_size=128, dense_activation='gelu',
                 optimizer='adam', class_weights=None
                 ):
        """
        Makes a classification model
        """

        self.model_name = model_name if model_name is not None else datetime.now().strftime('%m-%d_%H-%M-%S')

        # classifier options
        self.dropout_rate = dropout_rate
        self.num_lstm, self.lstm_size = num_lstm, lstm_size
        self.num_dense, self.dense_size, self.dense_activation = num_dense, dense_size, dense_activation

        self.bert_model = bert_model

        self.class_weights = class_weights
        self.optimizer = optimizer
        self.model = self.__make_model()

    def __make_classifier(self):
        """
        Creates the actual classification layer
        :return:
        """

        classifier = Sequential()

        for _ in range(self.num_lstm):
            classifier.add(Bidirectional(LSTM(
                units=self.lstm_size,
                return_sequences=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)
            )))
            classifier.add(Dropout(self.dropout_rate))

        for _ in range(self.num_dense):
            classifier.add(Dense(
                units=self.dense_size,
                activation=self.dense_activation,
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)
            ))
            classifier.add(Dropout(self.dropout_rate))

        classifier.add(Dense(
            units=2,
            activation='softmax',
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED)
        ))

        return classifier

    def __make_model(self):
        """
        Creates a classification model with a BERT embedding layer
        :return:
        """
        self.bert = TFAutoModel.from_pretrained(self.bert_model)
        self.bert.trainable = False

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
                PositiveClassF1()
            ]
        )

        return model

    @staticmethod
    def __scheduler(epoch, learning_rate):
        """
        Learning rate scheduler, reduces learning rate over time
        :param epoch:
        :param learning_rate:
        :return: updated learning rate
        """
        return max(learning_rate * np.exp(0.001 * -epoch), 0.00001)

    def fit(self, x, y, val=None, epochs=EPOCHS, lr_scheduler=False):

        callbacks = [
            TensorBoard(os.path.join('..', 'logs', self.model_name)),
            # ModelCheckpoint(os.path.join('..', 'models', 'checkpoints', 'temp'), save_best_only=True),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=5
            )
        ]

        if lr_scheduler:
            callbacks.append(LearningRateScheduler(self.__scheduler))

        train = self.__DataGenerator(x, y, BATCH_SIZE, bert_model=self.bert_model)

        if val is not None:
            val = self.__DataGenerator(val[0], val[1], BATCH_SIZE, bert_model=self.bert_model)

        self.model.fit(
            train,
            epochs=epochs,
            validation_data=val,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=2
        )

        # self.save()

    def predict(self, x):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        return self.model.predict(x)

    def test(self, x, y_true):
        """
        performs an F1 test on the test data for the positive class
        :param x: data
        :param y_true: true labels
        :return: f1 score for positive class
        """
        x = self.__DataGenerator(x, y_true, BATCH_SIZE, bert_model=self.bert_model)
        y_true = y_true.argmax(axis=1)
        y_pred = self.predict(x).argmax(axis=1)

        print(classification_report(y_true, y_pred, zero_division=True))

        return f1_score(y_true, y_pred, pos_label=1)

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
        self.model = self.__make_model()


if __name__ == '__main__':
    a = ADE_Detector()
