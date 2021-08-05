import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot(random_path, dal_path, title, use_error_bars=True):
    random_df = pd.read_csv(random_path)
    dal_df = pd.read_csv(dal_path)

    x = 'dataset_size'
    y = 'f1_score'

    ci = 'sd' if use_error_bars else None

    sns.set_theme(style='whitegrid')
    sns.lineplot(x=x, y=y, data=random_df, ci=ci,
                 marker='o', color='purple', linestyle='dashed', label='Random')
    sns.lineplot(x=x, y=y, data=dal_df, ci=ci,
                 marker='*', color='red', linestyle='solid', label='DAL')

    plt.title(title.replace('_', ' ').capitalize())
    plt.xlabel(x.replace('_', ' ').capitalize())
    plt.ylabel(y.replace('_', ' ').capitalize())
    plt.legend(loc='lower right')
    plt.show()

    plt.savefig(os.path.join('..', 'active_learning_scores', f'{title}.png'))


def main():

    root = os.path.join('..', 'active_learning_scores')

    random = os.path.join(root, 'Random_ISEAR.csv')
    dal = os.path.join(root, 'dal_f1_balanced_start_450.csv')

    title = 'ISEAR_results'

    plot(random, dal, title, use_error_bars=True)


if __name__ == '__main__':
    main()
