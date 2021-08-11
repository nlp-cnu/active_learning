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


def wilcoxon(random_path, dal_path,):
    from scipy.stats import wilcoxon, ttest_rel, ttest_ind

    alpha = 0.01

    random_df = pd.read_csv(random_path).groupby('dataset_size')
    dal_df = pd.read_csv(dal_path).groupby('dataset_size')

    print('Dataset Size | P-Value')
    for size in range(200, 1050, 50):
        try:
            random_scores = random_df.get_group(size)['f1_score']
            dal_scores = dal_df.get_group(size)['f1_score']
            statistic, p_value = wilcoxon(random_scores, dal_scores)
            print(f'{size:<12} | {p_value:<10.4f} |{" Not" if p_value < alpha else ""} Significantly Different')
        except ValueError:
            print(f'{size:<12} | {"nan":<10} | {"Identical"}')



def main():
    root = os.path.join('..', 'active_learning_scores')

    random = os.path.join(root, 'random_f1_balanced_start_1000.csv')
    dal = os.path.join(root, 'dal_f1_balanced_start_1000.csv')
    title = 'Balanced_Start_1000_samples'

    # random = os.path.join(root, 'random_f1.csv')
    # dal = os.path.join(root, 'dal_f1.csv')
    # title = 'Full_Dataset'

    wilcoxon(random, dal)
    # plot(random, dal, title, use_error_bars=True)


if __name__ == '__main__':
    main()
