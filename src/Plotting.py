import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


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


def significance_test(random_path, dal_path, alpha=0.01):
    random_df = pd.read_csv(random_path)
    dal_df = pd.read_csv(dal_path).groupby('dataset_size')

    dataset_sizes = np.unique(random_df['dataset_size'])

    random_df = random_df.groupby('dataset_size')

    print("Comparing F1 values between random and DAL per dataset size...\n")
    col1 = 'Dataset Size'
    col2 = f'P-Value : Alpha = {alpha}'
    col3 = 'Significance'
    header_string = " | ".join([col1, col2, col3])
    print(header_string)
    print('-'*len(header_string))

    spacing1 = len(col1)
    spacing2 = len(col2)

    for size in dataset_sizes:
        random_scores = random_df.get_group(size)['f1_score']
        dal_scores = dal_df.get_group(size)['f1_score']
        try:
            statistic, p_value = wilcoxon(random_scores, dal_scores, correction=True)
            print(f'{size:<{spacing1}} | '
                  f'{p_value:<{spacing2}} |'
                  f'{"" if p_value < alpha else " Not"} Significantly Different')
        except ValueError as e:
            print(f'{size:<{spacing1}} | {np.nan:<{spacing2}} | Identical')


def main():
    root = os.path.join('..', 'active_learning_scores')

    random = os.path.join(root, 'random_f1.csv')
    dal = os.path.join(root, 'dal_f1.csv')
    title = 'Full Dataset'

    # random = os.path.join(root, 'random_f1.csv')
    # dal = os.path.join(root, 'dal_f1.csv')
    # title = 'Full_Dataset'

    significance_test(random, dal)
    plot(random, dal, title, use_error_bars=True)


if __name__ == '__main__':
    main()
