import os
import shutil

# Remove excessive tf log messages
import sklearn.model_selection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import StratifiedKFold
from tensorboard_reducer import load_tb_events, reduce_events, write_tb_events
from tensorflow.keras.optimizers import *

import Plotting
from ADE_Detector import *
from Active_Learning import *
from Dataset import Dataset

# Delete Logs folder automatically for new runs
try:
    shutil.rmtree(os.path.join('..', 'logs'))
except FileNotFoundError:
    pass

TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')

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

    return np.mean(scores)  # avg training


def validation_testing():
    scores_file = os.path.join('..', 'combination_scores.txt')
    with open(scores_file, 'w+') as f:

        for lr in [1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5]:
            for epsilon in [1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7]:
                for num_lstm in range(1, 3):
                    model_name = f'LR-{lr}_EP-{epsilon}_{num_lstm}-LSTM'
                    f.write(model_name + ': ')

                    optimizer = Adam(
                        learning_rate=lr,
                        epsilon=epsilon
                    )

                    model = ADE_Detector(
                        model_name=model_name,
                        optimizer=optimizer,
                        num_lstm=num_lstm,
                        class_weights=db.get_train_class_weights()
                    )

                    f1 = cross_validation(model)
                    f.write(f'{f1}\n\n')
                    f.flush()


def tune_mini_queries():
    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    model = ADE_Detector()


def active_learning_experiment():
    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()
    test_x, test_y = db.get_test_data()
    # test_x, test_y, = test_x[:100], test_y[:100]

    optimizer = Adam(
        learning_rate=0.001,
        epsilon=1E-6
    )
    model = ADE_Detector(optimizer=optimizer, class_weights=db.get_train_class_weights())
    model.fit(ux, uy, val=(test_x, test_y))
    base_f1 = model.test(test_x, test_y)

    for budget in [10, 100, 500, 1000]:  # [10, 100, 500, 1000]
        ux, uy = db.get_train_data()
        random_scores = []
        while len(ux) > 0:
            (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)

            model.reset_model()
            model.fit(lx, ly, val=(test_x, test_y))
            f1 = model.test(test_x, test_y)
            random_scores.append((f1, len(lx)))

        lx, ly = [], np.array([])
        ux, uy = db.get_train_data()
        dal_scores = []

        print(len(ux))

        while len(ux) > 0:
            model.reset_model()

            (lx, ly), (ux, uy) = discriminative_active_learning((lx, ly), (ux, uy), budget, model)

            model.reset_model()
            model.fit(lx, ly, val=(test_x, test_y))
            f1 = model.test(test_x, test_y)
            dal_scores.append((f1, len(lx)))

        print(random_scores)

        print(dal_scores)

        Plotting.al_plot(base_f1, np.array(random_scores), np.array(dal_scores),
                         f'Active Learning Results - Annotation Budget: {budget}')


if __name__ == '__main__':
    # validation_testing()
    active_learning_experiment()
    # lstm_test()
