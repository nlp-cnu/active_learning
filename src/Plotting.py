import pandas as pd
from matplotlib import pyplot as plt


def al_plot(base_path, random_path, dal_path, fig_name):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(fig_name)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("F1-Score")

    # read in data
    base_df = pd.read_csv(base_path)
    random_df = pd.read_csv(random_path, na_values=0).dropna()
    dal_df = pd.read_csv(dal_path, na_values=0).dropna()

    # baseline
    base_f1 = base_df.iloc[0]['f1_score']
    xmin = random_df.iloc[0]['dataset_size']
    xmax = random_df.iloc[-1]['dataset_size']
    # ax.hlines(base_f1, xmin=xmin, xmax=xmax, linestyles='dotted', label='Full Dataset', color='blue')

    # Random AL
    ax.plot(random_df['dataset_size'], random_df['f1_score'],
            color='purple', marker='o', linestyle='dashed', label='Random')

    # DAL
    ax.plot(dal_df['dataset_size'], dal_df['f1_score'],
            color='red', marker='o', linestyle='solid', label='DAL', )

    ax.legend(loc='lower right')
    # plt.xticks(np.arange(xmin, xmax + 1, 50))

    plt.show()
    fig.savefig(os.path.join('..', 'active_learning_scores', f'{fig_name}.png'))


if __name__ == '__main__':
    import os
    root = os.path.join('..', 'active_learning_scores')

    base = os.path.join(root, 'base_f1.csv')
    random = os.path.join(root, 'random_f1_50_0.csv')
    dal = os.path.join(root, 'random_f1_50.csv')

    al_plot(base, random, dal, 'Results')
