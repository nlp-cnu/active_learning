import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot(random_path, dal_path, title):
    random_df = pd.read_csv(random_path)
    dal_df = pd.read_csv(dal_path)

    x = 'dataset_size'
    y = 'f1_score'

    sns.set_theme(style='whitegrid')
    sns.lineplot(x=x, y=y, data=random_df, ci='sd', err_style='band',
                 marker='o', linestyle='dashed', label='Random')
    sns.lineplot(x=x, y=y, data=dal_df, ci='sd', err_style='band',
                 marker='o', linestyle='solid', label='DAL')

    plt.title(title.replace('_', ' ').capitalize())
    plt.xlabel(x.replace('_', ' ').capitalize())
    plt.ylabel(y.replace('_', ' ').capitalize())
    plt.legend(loc='lower right')
    plt.show()

    plt.savefig(os.path.join('..', 'active_learning_scores', f'{title}.png'))


if __name__ == '__main__':
    root = os.path.join('..', 'active_learning_scores')

    random = os.path.join(root, 'random_f1_balanced_start_1000.csv')
    dal = os.path.join(root, 'dal_f1_balanced_start_1000.csv')

    title = 'balanced_start_results_1000'

    plot(random, dal, title)
