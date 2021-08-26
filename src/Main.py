import os
import shutil
from datetime import datetime

import numpy as np

from ADE_Detector import ADE_Detector
from Active_Learning import random_active_learning, discriminative_active_learning
from Dataset import Dataset
from Utils import *

# Delete Logs folder automatically for new runs
try:
    shutil.rmtree(os.path.join('..', 'logs'))
except FileNotFoundError:
    pass

SCORES_PATH = os.path.join('..', 'active_learning_scores')
if not os.path.exists(SCORES_PATH):
    os.mkdir(SCORES_PATH)

TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')

isear_path = os.path.join('..', 'data', 'ISEAR.txt')
db = Dataset(seed=SEED, data_filepath=isear_path)


def cross_validation(model):
    """
    Method for tuning model parameters using k-fold cross validation
    :param model: an untrained model
    :return:
    """
    from sklearn.model_selection import StratifiedKFold
    from tensorboard_reducer import load_tb_events, reduce_events, write_tb_events

    x, y = db.get_train_data()

    scores = []

    model_name = model.model_name

    iterator = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True).split(x, y.argmax(axis=1))
    for fold, (train_idxs, val_idxs) in enumerate(iterator):
        train_x, train_y = [x[idx] for idx in train_idxs], y[train_idxs]
        val_x, val_y = [x[idx] for idx in val_idxs], y[val_idxs]

        model.model_name = f'{model_name}_fold_{fold + 1}'
        model.fit(train_x, train_y, (val_x, val_y))
        f1 = model.eval(val_x, val_y)
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
            bert_model=ROBERTA_TWITTER
        )

        f1, sd = cross_validation(model)
        f.write(f'RoBERTa weights - F1: {f1} SD: {sd}\n')
        f.flush()

        model = ADE_Detector(
            bert_model=ROBERTA_TWITTER
        )

        f1, sd = cross_validation(model)
        f.write(f'RoBERTa - F1: {f1} SD: {sd}\n')
        f.flush()


def get_initial_datasets(initial_dataset_size=200, monitor_idx=1, seed=SEED):
    """
    Creates an artificially balanced labeled dataset for active learning.
    :param initial_dataset_size: Size of labeled dataset.
    :param monitor_idx: Index of class to monitor.
    :param seed: Seed for RNG.
    :return: Pre-balanced labeled dataset
    """
    rng = np.random.default_rng(seed=seed)

    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    # first n / 2 samples are randomly selected positive instances
    idxs = np.where(uy[:, monitor_idx])[0]
    rng.shuffle(idxs)
    idxs = idxs[:initial_dataset_size // 2]
    for idx in sorted(idxs, reverse=True):
        lx.append(ux.pop(idx))
        ly = np.concatenate((ly, [uy[idx]])) if len(ly) != 0 else np.array([uy[idx]])
        uy = np.delete(uy, idx, axis=0)

    # second n / 2 samples are randomly selected and labeled as negative
    idxs = rng.choice(len(uy), initial_dataset_size // 2, replace=False)
    neg_label = [0, 1] if monitor_idx == 0 else [1, 0]
    for idx in sorted(idxs, reverse=True):
        lx.append(ux.pop(idx))
        ly = np.concatenate((ly, [neg_label]))
        uy = np.delete(uy, idx, axis=0)

    return (lx, ly), (ux, uy)


def train_models(labeled, unlabeled, budget, max_dataset_size, file_path, positive_class_idx=1):
    """

    :param labeled: The labeled dataset.
    :param unlabeled: The unlabeled dataset.
    :param budget: Number of samples to annotate per selection round.
    :param max_dataset_size: Maximum size of the labeled dataset.
    :param file_path: File to write scores to.
    :param positive_class_idx:
    """
    (lx, ly), (ux, uy) = labeled, unlabeled
    test_x, test_y = db.get_test_data()

    if max_dataset_size is None or max_dataset_size == 0:
        max_dataset_size = len(uy)

    model = ADE_Detector(monitor_idx=positive_class_idx)

    with open(file_path, 'a+') as f:
        if len(lx) > 0:
            print('Testing on initial dataset')
            model.fit(lx, ly, val_data=db.get_val_set(), use_class_weights=False)
            f1 = model.eval(test_x, test_y)

            f.write(f'{f1},{len(lx)}\n')
            f.flush()

        while len(lx) < max_dataset_size:
            if 'random' in file_path.lower():
                selection_type = 'random'
                (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)

            else:
                selection_type = 'DAL'
                model.reset_model()
                (lx, ly), (ux, uy) = discriminative_active_learning((lx, ly), (ux, uy), budget, model)
                dataset_file = os.path.join(SCORES_PATH, f'dal_dataset_{budget}_{len(lx)}.tsv')
                with open(dataset_file, 'a+', encoding='utf8') as dal_dataset:
                    for sample, label in zip(lx, ly):
                        dal_dataset.write(f'{sample}\t{np.argmax(label)}\n')
                    dal_dataset.write('-'*30)

            print(f'Training model with {len(lx)} samples selected by {selection_type}...')
            model.reset_model()
            model.fit(lx, ly, val_data=db.get_val_set(), use_class_weights=False)
            f1 = model.eval(test_x, test_y)

            f.write(f'{f1},{len(lx)}\n')
            f.flush()


def active_learning_experiment():
    balanced_dataset_size = 1000

    for path in [
        # os.path.join(SCORES_PATH, f'random_f1_balanced_start_{balanced_dataset_size}.csv'),
        # os.path.join(SCORES_PATH, f'dal_f1_balanced_start_{balanced_dataset_size}.csv'),
        # os.path.join(SCORES_PATH, 'random_f1.csv'),
        # os.path.join(SCORES_PATH, 'dal_f1.csv'),
        os.path.join(SCORES_PATH, 'Random_ISEAR.csv'),
        os.path.join(SCORES_PATH, 'DAL_ISEAR.csv'),
    ]:

        with open(path, 'w+') as f:
            f.write('f1_score,dataset_size\n')
            f.flush()

        positive_class_idx = 0 if 'isear' in path.lower() else 1

        if 'balanced' in path.lower() or 'isear' in path.lower():
            budget = 50
            rng_seeds = [1556, 1354, 9380, 5715, 8639]
            for seed in rng_seeds:
                labeled, unlabeled = get_initial_datasets(seed=seed, monitor_idx=positive_class_idx)
                train_models(labeled, unlabeled, budget, balanced_dataset_size, path, positive_class_idx)
        else:
            budget = 1000
            for _ in range(5):
                labeled = [], np.array([])
                unlabeled = db.get_train_data()
                train_models(labeled, unlabeled, budget, None, path)


if __name__ == '__main__':
    # validation_testing()
    active_learning_experiment()
