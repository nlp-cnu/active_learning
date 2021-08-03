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

# Remove excessive tf log messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


def get_initial_datasets(initial_dataset_size=200, seed=SEED):
    """
    Creates an artificially balanced labeled dataset for active learning.
    :param initial_dataset_size: Size of labeled dataset.
    :param seed: Seed for RNG.
    :return:
    """
    rng = np.random.default_rng(seed=seed)

    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    # first 100 samples are randomly selected positive instances
    idxs = np.where(uy[:, 1])[0]
    rng.shuffle(idxs)
    for idx in sorted(idxs[:initial_dataset_size // 2], reverse=True):
        lx.append(ux.pop(idx))
        ly = np.concatenate((ly, [uy[idx]])) if len(ly) != 0 else np.array([uy[idx]])
        uy = np.delete(uy, idx, axis=0)

    # second 100 samples are randomly selected and labeled as negative
    idxs = rng.choice(len(uy), initial_dataset_size // 2, replace=False)
    for idx in sorted(idxs, reverse=True):
        lx.append(ux.pop(idx))
        ly = np.concatenate((ly, [[1, 0]]))
        uy = np.delete(uy, idx, axis=0)

    return (lx, ly), (ux, uy)


# get initial datasets with heuristic instead of using labels
def get_initial_datasets_with_heuristic(seed=SEED):
    pass


def train_models(labeled: tuple[list, np.ndarray], unlabeled: tuple[list, np.ndarray], budget: int,
                 max_dataset_size: int, file_path: str):
    """

    :param labeled:
    :param unlabeled:
    :param budget:
    :param max_dataset_size:
    :param file_path:
    :return:
    """
    (lx, ly), (ux, uy) = labeled, unlabeled
    test_x, test_y = db.get_test_data()

    optimizer = Adam(
        learning_rate=0.001,
        epsilon=1E-6
    )
    model = ADE_Detector(optimizer=optimizer)

    with open(file_path, 'a+') as f:
        if len(lx) > 0:
            print('Testing on initial dataset')

            model.fit(lx, ly, val_data=(test_x, test_y), use_class_weights=False)
            f1 = model.eval(test_x, test_y)

            f.write(f'{f1},{len(lx)}\n')
            f.flush()

        while len(lx) < max_dataset_size:
            if 'random' in file_path:
                selection_type = 'random'
                (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)
            else:
                selection_type = 'DAL'
                model.reset_model()
                (lx, ly), (ux, uy) = discriminative_active_learning((lx, ly), (ux, uy), budget, model)

            print(f'Training model with {len(lx)} samples selected by {selection_type}...')

            model.reset_model()
            model.fit(lx, ly, val_data=(test_x, test_y), use_class_weights=False)
            f1 = model.eval(test_x, test_y)

            f.write(f'{f1},{len(lx)}\n')
            f.flush()


def active_learning_experiment():
    for budget, max_dataset_size in [(50, 1000), (500, len(db.get_train_data()[1]))]:

        random_path = os.path.join(SCORES_PATH, f'random_f1_balanced_start_{max_dataset_size}.csv')
        dal_path = os.path.join(SCORES_PATH, f'dal_f1_balanced_start_{max_dataset_size}.csv')

        with open(random_path, 'w+') as f:
            f.write('f1_score,dataset_size\n')

        with open(dal_path, 'w+') as f:
            f.write('f1_score,dataset_size\n')

        # test with balanced start
        rng_seeds = [1556, 1354, 9380, 5715, 8639]
        for idx, seed in enumerate(rng_seeds):
            for file_path in [random_path, dal_path]:
                labeled, unlabeled = get_initial_datasets(seed)
                train_models(labeled, unlabeled, budget, max_dataset_size, file_path)

        random_path = os.path.join(SCORES_PATH, f'random_f1_{max_dataset_size}.csv')
        dal_path = os.path.join(SCORES_PATH, f'dal_f1_{max_dataset_size}.csv')

        with open(random_path, 'w+') as f:
            f.write('f1_score,dataset_size\n')

        with open(dal_path, 'w+') as f:
            f.write('f1_score,dataset_size\n')

        # test from scratch
        for idx in range(5):
            for file_path in [random_path, dal_path]:
                labeled = [], np.array([])
                unlabeled = db.get_train_data()
                train_models(labeled, unlabeled, budget, max_dataset_size, file_path)


# generate 1 graph w/o positive start avg over 5 runs up to 1000 samples
# generate 1 graph w/ positive start avg over 5 runs up to 1000 samples
# 1 graph w/o positive start up to dataset size interval of 500
# 1 graph w/ positive start up to dataset size interval of 500


if __name__ == '__main__':
    # validation_testing()
    active_learning_experiment()
    # lstm_test()
