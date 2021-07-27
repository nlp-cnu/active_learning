import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorboard as tb


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
    plt.xticks(np.arange(xmin, xmax + 1, 50))

    plt.show()
    fig.savefig(os.path.join('..', 'active_learning_scores', f'{fig_name}.png'))


def plot_from_tensorboard():
    experiment_id = "mzqnKcymS5WgKTps3EBbXw"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    f = df['run'].str.contains('BERT_Optimizer_Adam')
    df = df[f]
    f = df['run'].str.endswith('validation')
    df = df[f]

    print(df)

    max_epoch = max(df['step'])

    print(max_epoch)

    # data = pd.DataFrame(columns=['run', 'step', 'epoch_loss', 'epoch_positive_class_F1'])
    # for idx, (run, tag, step, value) in df.iterrows():
    #     if run not in list(data['run']) or step not in list(data.loc[data['run'] == run]['step']):
    #         data = data.append({'run': run, 'step': step, tag: value}, ignore_index=True)
    #     else:
    #         data.loc[(data['run'] == run) & (data['step'] == step), [tag]] = value
    #
    # avg_data = pd.DataFrame(columns=['step', 'avg_loss', 'avg_F1'])
    # for idx in range(max(data['step'])):
    #     runs = data.loc[data['step'] == idx]
    #
    #     if len(runs) < 5:
    #         previous = data.loc[data['step'] == idx - 1]
    #
    #         for run in list(runs['run']):
    #             if run not in list(previous['run']):
    #                 loss = previous.loc[previous['run'] == run, 'epoch_loss']
    #
    #     loss = np.array(runs['epoch_loss']).mean()
    #     f1 = np.array(runs['epoch_positive_class_F1']).mean()
    #
    #     avg_data = avg_data.append({'step': idx, 'avg_loss': loss, 'avg_F1': f1}, ignore_index=True)
    #
    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # sns.lineplot(data=data, x="step", y="epoch_positive_class_F1").set_title("Positive F1")
    #
    # plt.subplot(1, 2, 2)
    # sns.lineplot(data=data, x="step", y="epoch_loss").set_title("Loss")

    # plt.subplot(1, 2, 1)
    # sns.lineplot(data=avg_data, x="step", y="avg_F1").set_title("Positive F1")

    # plt.subplot(1, 2, 2)
    # sns.lineplot(data=avg_data, x="step", y="avg_loss").set_title("Loss")

    # plt.show()


if __name__ == '__main__':
    import os
    root = os.path.join('..', 'active_learning_scores')

    base = os.path.join(root, 'base_f1.csv')
    random = os.path.join(root, 'random_f1_50.csv')
    dal = os.path.join(root, 'dal_f1_50.csv')

    al_plot(base, random, dal, 'Results')
