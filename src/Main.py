import os
import random
from datetime import datetime

import numpy as np

from Dataset import Dataset
from Classifier import Classifier

# Make directory structure
time_stamp = datetime.now().strftime('%m-%d_%H-%M-%S')
os.makedirs(os.path.join('..', 'models'), exist_ok=True)
os.makedirs(os.path.join('..', 'models', 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join('..', 'models', 'trained'), exist_ok=True)
os.makedirs(os.path.join('..', 'scores'), exist_ok=True)
os.makedirs(os.path.join('..', 'scores', time_stamp))
FULLY_TRAINED_F1 = os.path.join('..', 'scores', time_stamp, 'fully_trained.txt')
RANDOM_F1 = os.path.join('..', 'scores', time_stamp, 'random_selection.txt')


def plot(score_dir=os.path.join('..', 'scores', time_stamp)):
    """
    Plots the results to Active Learning
    :param score_dir: directory of files of scores
    """
    import matplotlib.pyplot as plt

    score_files = [score_dir + file for file in os.listdir(score_dir)]

    plt.tight_layout()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('ADE Detection in Tweets')
    ax1.set_xlabel('Number of Training Samples')
    ax1.set_ylabel('F1 Score (Positive ADE)')

    for file, color in zip(score_files, colors):
        if '.txt' not in file:
            continue

        with open(file) as f:
            if 'fully_trained' in file:
                f1, num_samples = f.readline().rstrip().split(',')
                # todo: get better way of having the initial sample size
                temp = Dataset()
                starting_samples = len(temp.get_active_set()[0][0])
                del temp
                ax1.hlines(float(f1), starting_samples, int(num_samples), linestyles='dashed',
                           label='Full Dataset', color=color)
                continue

            plt_name = ' '.join(string.capitalize() for string in file.split('/')[-1][:-4].split('_'))
            f1_scores = []
            sample_idx = []
            for line in f:
                f1, num_samples = line.rstrip().split(',')
                f1_scores.append(float(f1))
                sample_idx.append(int(num_samples))
            ax1.plot(sample_idx, f1_scores, label=plt_name, color=color)

    ax1.legend(loc='lower right')
    plt.show()
    ax1.figure.savefig(f'../scores/{time_stamp}/plot.png')


def random_sampler(x, sample_size):
    """
    Random sampler for active learning
    :param x: the unlabeled dataset
    :param sample_size: the number of samples to pull from the dataset
    :return: the indexes of the samples to retrieve
    """
    try:
        return random.sample(range(len(x)), sample_size)
    except ValueError:
        # sample size larger than remaining samples
        return list(range(len(x)))


def main():
    """
    Main function to run tests on various models and systems
    """

    dataset = Dataset()
    my_model = Classifier(model_name=time_stamp + '_full_dataset')

    # train and test a full classifier on the training data
    x, y = dataset.get_train_data()
    my_model.fit(x, y)

    x_test, y_test = dataset.get_test_data()
    f1 = my_model.test(x_test, y_test)

    # initialize active learning
    num_samples = len(x)  # total number of samples in dataset
    sample_size = 1000  # number of samples to retrieve from unlabeled set
    del x, y

    # record fully trained results for plotting
    with open(FULLY_TRAINED_F1, 'w') as f:
        f.write(f'{f1}\t{num_samples}')

    # get AL split
    (labeled_x, labeled_y), (unlabeled_x, unlabeled_y) = dataset.get_active_set()

    # Active Learning (Random Selection)
    num_selection_rounds = (num_samples // sample_size) + int(bool((num_samples % sample_size)))
    for _ in range(num_selection_rounds):
        # remake model
        my_model = Classifier(model_name=time_stamp + '_random_selection_' + num_selection_rounds)

        # train
        my_model.fit(labeled_x, labeled_y)

        # test & record for plotting
        f1 = my_model.test(x_test, y_test)
        with open(RANDOM_F1, 'w') as f:
            f.write(f'{f1},{len(labeled_x)}\n')

        # Gather new samples
        idxs = random_sampler(unlabeled_x, sample_size)
        np.concatenate(labeled_x, unlabeled_x[idxs])
        np.concatenate(labeled_y, unlabeled_y[idxs])
        np.delete(unlabeled_x, idxs, axis=0)
        np.delete(unlabeled_y, idxs, axis=0)

    plot()


if __name__ == '__main__':
    main()
