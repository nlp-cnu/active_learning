import os
import shutil
from datetime import datetime

# Remove excessive tf log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import StratifiedKFold
from tensorboard_reducer import load_tb_events, reduce_events, write_tb_events
from tensorflow.keras.optimizers import *

from ADE_Detector import *
from Active_Learning import *
from Dataset import Dataset

# Delete Logs folder automatically for new runs
try:
    shutil.rmtree(os.path.join('..', 'logs'))
except FileNotFoundError:
    pass

TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')
# SEED = 2005

db = Dataset(seed=SEED)


def cross_validation(model):
    """
    Method for tuning model parameters using k-fold cross validation
    :param model: an untrained model
    :return:
    """

    x, y = db.get_train_data()

    scores = []

    model_name = model.model_name

    iterator = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    for idx, (train_idxs, val_idxs) in enumerate(iterator.split(x, y.argmax(axis=1))):
        train_x, train_y = [x[idx] for idx in train_idxs], y[train_idxs]
        val_x, val_y = [x[idx] for idx in val_idxs], y[val_idxs]

        model.model_name = f'{model_name}_fold_{idx + 1}'
        history = model.fit(train_x, train_y, (val_x, val_y))
        f1 = model.test(val_x, val_y)
        scores.append(f1)

        model.reset_model()

    for dataset in ['train', 'validation']:
        input_dir = os.path.join('..', 'logs', model_name + '*', dataset)
        output_dir = os.path.join('..', 'logs', model_name, dataset)
        events_dict = load_tb_events(input_dir, strict_steps=False)
        reduced_events = reduce_events(events_dict, ['mean'])
        write_tb_events(reduced_events, output_dir)

    return np.mean(scores)  # avg training,


def validation_testing():
    scores_file = os.path.join('..', 'scores.txt')
    with open(scores_file, 'a+') as f:

        f.write('Base model f1 - ')
        model = ADE_Detector(model_name='Base_BERT')
        f1_score = cross_validation(model)
        f.write(f'{f1_score}\n\n')
        f.flush()

        f.write('Base model f1 class weights - ')
        model = ADE_Detector(model_name='Weights', class_weights=db.get_train_class_weights())
        f1_score = cross_validation(model)
        f.write(f'{f1_score}\n\n')
        f.flush()

        # f.write('BERT models:\n')
        # bert_models = {
        #     # 'Base_BERT': 'bert-base-uncased',
        #     'Twitter_RoBERTa': 'cardiffnlp/twitter-roberta-base',
        #     'BioRedditBERT': 'cambridgeltl/BioRedditBERT-uncased',
        # }
        # for bert, location in bert_models.items():
        #     model = ADE_Detector(model_name=f'{bert}', bert_model=location)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{bert:15} - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # f.write('Dropout rates:\n')
        # dropout_rates = [0.0] + [num / 10 for num in range(5, 9)]  # 0.0 + 0.5 to 0.8
        # for dropout in dropout_rates:
        #     model = ADE_Detector(model_name=f'Dropout_{dropout}', dropout_rate=dropout)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{dropout} - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # f.write('Optimizers:\n')
        # optimizers = {
        #     'Adam': Adam(),
        #     'Nadam': Nadam(),
        #     'Adamax': Adamax(),
        #     'Adagrad': Adagrad(),
        #     'SGD': SGD()
        # }
        # for name, optimizer in optimizers.items():
        #     model = ADE_Detector(model_name=f'Optimizer_{name}', optimizer=optimizer)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{name:7} - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # f.write('Learning Rates:\n')
        # rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        # for lr in rates:
        #     optimizer = Adam(learning_rate=lr)
        #     model = ADE_Detector(model_name=f'Learning_Rate_{lr}', optimizer=optimizer)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{lr:7} - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # f.write('Epsilons:\n')
        # for epsilon in rates:
        #     optimizer = Adam(epsilon=epsilon)
        #     model = ADE_Detector(model_name=f'Epsilon_{epsilon}', optimizer=optimizer)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{epsilon:7} - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # f.write('LSTM:\n')
        # num_lstm = list(range(1, 3))
        # for num in num_lstm:
        #     model = ADE_Detector(model_name=f'LSTM_Layers_{num}', num_lstm=num)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{num} LSTM layers - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # lstm_sizes = [2 ** num for num in range(7, 10)]
        # for size in lstm_sizes:
        #     model = ADE_Detector(model_name=f'LSTM_Units_{num}', lstm_size=size)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{size} LSTM units - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # f.write('Dense:\n')
        # num_dense = list(range(1, 3))
        # for num in num_dense:
        #     model = ADE_Detector(model_name=f'Dense_Layers_{num}', num_dense=num)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{num} Dense layers - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # dense_sizes = [2 ** num for num in range(5, 10)]
        # for size in dense_sizes:
        #     model = ADE_Detector(model_name=f'Dense_Units_{num}', dense_size=size)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{size} Dense units - {f1_score}\n')
        #     f.flush()
        # f.write('\n')
        #
        # dense_activations = ['relu', 'elu', 'gelu', 'tanh']
        # for activation in dense_activations:
        #     model = ADE_Detector(model_name=f'Dense_Activation_{activation}', dense_activation=activation)
        #     f1_score = cross_validation(model)
        #     f.write(f'\t{activation} Dense activation - {f1_score}\n')
        #     f.flush()


if __name__ == '__main__':
    validation_testing()
