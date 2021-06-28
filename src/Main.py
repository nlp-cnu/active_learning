import os
import shutil

# Remove excessive tf log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import StratifiedKFold
from tensorboard_reducer import load_tb_events, reduce_events, write_tb_events
from tensorflow.keras.optimizers import *

from ADE_Detector import *
from Active_Learning import *
from Dataset import Dataset

# Delete Logs folder automatically for new runs
# try:
#     shutil.rmtree(os.path.join('..', 'logs'))
# except FileNotFoundError:
#     pass

TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')

db = Dataset(seed=SEED)


def cross_validation(model):
    """
    Method for tuning model parameters using k-fold cross validation
    :param model: an untrained model
    :return:
    """

    x, y = db.get_train_data()

    x = x[:100]
    y = y[:100]

    scores = []

    model_name = model.model_name

    iterator = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    for idx, (train_idxs, val_idxs) in enumerate(iterator.split(x, y.argmax(axis=1))):
        train_x, train_y = [x[idx] for idx in train_idxs], y[train_idxs]
        val_x, val_y = [x[idx] for idx in val_idxs], y[val_idxs]

        model.model_name = f'{model_name}_fold_{idx + 1}'
        model.fit(train_x, train_y, (val_x, val_y))
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
        bert_models = {
            # "BERT": BASEBERT,
            # 'RoBERTaTweet': ROBERTA_TWITTER,
            'BioRedditBERT': BIOREDDITBERT,
        }
        for bert, bert_model in bert_models.items():

            f.write(f'{bert} f1 - ')
            model = ADE_Detector(model_name=f'{bert}', bert_model=bert_model,)
            f1_score = cross_validation(model)
            f.write(f'{f1_score}\n\n')
            f.flush()

            f.write(f'{bert} f1 class weights - ')
            model = ADE_Detector(model_name=f'{bert}_Class_Weights', bert_model=bert_model,
                                 class_weights=db.get_train_class_weights())
            f1_score = cross_validation(model)
            f.write(f'{f1_score}\n\n')
            f.flush()

            f.write(f'{bert} Dropout rates:\n')
            dropout_rates = [0.0] + [num / 10 for num in range(5, 9)]  # 0.0 + 0.5 to 0.8
            for dropout in dropout_rates:
                model = ADE_Detector(model_name=f'{bert}_Dropout_{dropout}', dropout_rate=dropout, bert_model=bert_model)
                f1_score = cross_validation(model)
                f.write(f'\t{dropout} - {f1_score}\n')
                f.flush()
            f.write('\n')

            f.write(f'{bert} Optimizers:\n')
            optimizers = {
                'Adam': Adam(),
                'AMSgrad': Adam(amsgrad=True),
                'Nadam': Nadam(),
                'Adamax': Adamax(),
                'Adagrad': Adagrad(),
                'SGD': SGD()
            }
            for name, optimizer in optimizers.items():
                model = ADE_Detector(model_name=f'{bert}_Optimizer_{name}', optimizer=optimizer, bert_model=bert_model)
                f1_score = cross_validation(model)
                f.write(f'\t{name:7} - {f1_score}\n')
                f.flush()
            f.write('\n')

            f.write(f'{bert} Learning Rates:\n')
            rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            for lr in rates:
                optimizer = Adam(learning_rate=lr)
                model = ADE_Detector(model_name=f'{bert}_Learning_Rate_{lr}', optimizer=optimizer, bert_model=bert_model)
                f1_score = cross_validation(model)
                f.write(f'\t{lr:7} - {f1_score}\n')
                f.flush()
            f.write('\n')

            f.write(f'{bert} Epsilons:\n')
            rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            for epsilon in rates:
                optimizer = Adam(epsilon=epsilon)
                model = ADE_Detector(model_name=f'{bert}_Epsilon_{epsilon}', optimizer=optimizer, bert_model=bert_model)
                f1_score = cross_validation(model)
                f.write(f'\t{epsilon:7} - {f1_score}\n')
                f.flush()
            f.write('\n')

            # f.write(f'{bert} LSTM:\n')
            # num_lstm = list(range(1, 4))
            # for num in num_lstm:
            #     model = ADE_Detector(model_name=f'{bert}_LSTM_Layers_{num}', num_lstm=num, bert_model=bert_model)
            #     f1_score = cross_validation(model)
            #     f.write(f'\t{num} LSTM layers - {f1_score}\n')
            #     f.flush()
            # f.write('\n')
            #
            # lstm_sizes = [2 ** num for num in range(7, 10)]
            # for size in lstm_sizes:
            #     model = ADE_Detector(model_name=f'{bert}_LSTM_Units_{size}', lstm_size=size, bert_model=bert_model)
            #     f1_score = cross_validation(model)
            #     f.write(f'\t{size} LSTM units - {f1_score}\n')
            #     f.flush()
            # f.write('\n')
            #
            # f.write(f'{bert} Dense:\n')
            # num_dense = list(range(1, 6))
            # for num in num_dense:
            #     model = ADE_Detector(model_name=f'{bert}_Dense_Layers_{num}', num_dense=num, bert_model=bert_model)
            #     f1_score = cross_validation(model)
            #     f.write(f'\t{num} Dense layers - {f1_score}\n')
            #     f.flush()
            # f.write('\n')
            #
            # dense_sizes = [2 ** num for num in range(5, 10)]
            # for size in dense_sizes:
            #     model = ADE_Detector(model_name=f'{bert}_Dense_Units_{size}', dense_size=size, bert_model=bert_model)
            #     f1_score = cross_validation(model)
            #     f.write(f'\t{size} Dense units - {f1_score}\n')
            #     f.flush()
            # f.write('\n')
            #
            # dense_activations = ['relu', 'elu', 'gelu', 'tanh']
            # for activation in dense_activations:
            #     model = ADE_Detector(model_name=f'{bert}_Dense_Activation_{activation}', dense_activation=activation, bert_model=bert_model)
            #     f1_score = cross_validation(model)
            #     f.write(f'\t{activation} Dense activation - {f1_score}\n')
            #     f.flush()


def tune_dal():
    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    model = ADE_Detector()


def active_learning_experiment():
    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()
    test_x, test_y = db.get_test_data()

    model = ADE_Detector()
    model.fit(ux, uy)
    base_f1 = model.test(test_x, test_y)

    random_runs = []
    for budget in [10, 100, 1000]:
        scores = []
        while len(ux) > 0:
            model.reset_model()
            (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)

            model.fit(lx, ly)
            f1 = model.test(test_x, test_y)
            scores.append((f1, len(lx)))
        random_runs.append(scores)

    dal_runs = []
    for budget in [10, 100, 1000]:
        scores = []
        while len(ux) > 0:
            model.reset_model()

            (lx, ly), (ux, uy) = discriminative_active_learning((lx, ly), (ux, uy), budget, model)

            model.reset_model()

            model.fit(lx, ly)
            f1 = model.test(test_x, test_y)
            scores.append((f1, len(lx)))
        dal_runs.append(scores)



if __name__ == '__main__':
    validation_testing()
    # active_learning_experiment()
    # lstm_test()
