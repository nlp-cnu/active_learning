import os
import shutil

# Remove excessive tf log messages

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

SCORES_PATH = os.path.join('..', 'active_learning_scores')
if not os.path.exists(SCORES_PATH):
    os.mkdir(SCORES_PATH)

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

    return np.mean(scores), np.std(scores)  # avg training


def validation_testing():
    scores_file = os.path.join('..', 'simple_test.txt')
    with open(scores_file, 'w+') as f:
        # model = ADE_Detector(
        #     class_weights=db.get_train_class_weights(),
        #     bert_model=BIOREDDITBERT
        # )
        #
        # f1, sd = cross_validation(model)
        # f.write(f'BioReddit - F1: {f1} SD: {sd}\n')
        # f.flush()

        model = ADE_Detector(
            class_weights=db.get_train_class_weights(),
            bert_model=ROBERTA_TWITTER
        )

        f1, sd = cross_validation(model)
        f.write(f'RoBERTa weights - F1: {f1} SD: {sd}\n')
        f.flush()

        model = ADE_Detector(
            # class_weights=db.get_train_class_weights(),
            class_weights=None,
            bert_model=ROBERTA_TWITTER
        )

        f1, sd = cross_validation(model)
        f.write(f'RoBERTa - F1: {f1} SD: {sd}\n')
        f.flush()


def tune_mini_queries():
    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    model = ADE_Detector()


def active_learning_experiment():

    x, y = db.get_train_data()
    test_x, test_y = db.get_test_data()

    print('Creating Model')
    optimizer = Adam(
        learning_rate=0.001,
        epsilon=1E-6
    )
    model = ADE_Detector(optimizer=optimizer, class_weights=db.get_train_class_weights())

    print('Testing Model on all data...')
    model.fit(x, y, val=(test_x, test_y))
    base_f1 = model.test(test_x, test_y)

    base_path = os.path.join(SCORES_PATH, 'base_f1.csv')
    with open(base_path, 'w+') as f:
        f.write('f1_score,dataset_size\n')
        f.write(f'{base_f1},{len(x)}\n')

    del x, y

    for budget in [1000, 500, 100, 10]:
        lx, ly = [], np.array([])
        ux, uy = db.get_train_data()

        random_path = os.path.join(SCORES_PATH, f'random_f1_{budget}.csv')
        with open(random_path, 'w+') as f:
            f.write('f1_score,dataset_size\n')
            while len(ux) > 0:
                print('Selecting samples with random active learning...')
                (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)
                model.reset_model()

                print(f'Random Model with {len(lx)} samples')
                model.fit(lx, ly, val=(test_x, test_y))
                f1 = model.test(test_x, test_y)
                f.write(f'{f1},{len(lx)}\n')
                f.flush()

        lx, ly = [], np.array([])
        ux, uy = db.get_train_data()

        dal_path = os.path.join(SCORES_PATH, f'dal_f1_{budget}.csv')
        with open(dal_path, 'w+') as f:
            f.write('f1_score,dataset_size\n')
            while len(ux) > 0:

                model.reset_model()
                (lx, ly), (ux, uy) = discriminative_active_learning((lx, ly), (ux, uy), budget, model)
                model.reset_model()

                print(f'DAL Model with {len(lx)} samples')
                model.fit(lx, ly, val=(test_x, test_y))
                f1 = model.test(test_x, test_y)
                f.write(f'{f1},{len(lx)}\n')
                f.flush()

        Plotting.al_plot(base_path, random_path, dal_path,
                         f'Active Learning Results - Annotation Budget: {budget}')


if __name__ == '__main__':
    # validation_testing()
    active_learning_experiment()
    # lstm_test()
