import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorboard as tb


def al_plot(base_f1, random_runs, dal_runs, fig_name):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(fig_name)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("F1-Score")

    # baseline
    ax.hlines(base_f1, xmin=0, xmax=max(random_runs[:, 1]), linestyles='dashed', label='Full Dataset', color='blue')

    # Random AL
    ax.plot(random_runs[:, 1], random_runs[:, 0], label='Random AL', color='red')

    # DAL
    ax.plot(dal_runs[:, 1], dal_runs[:, 0], label='DAL', color='green')

    ax.legend(loc='lower right')

    plt.show()
    fig.savefig(f'{fig_name}.png')


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
    # plot_test_graph(0.5, [0.1, 0.2, 0.3, 0.4, 0.5], [100, 200, 300, 400, 500], '../test_plot.png')
    plot_from_tensorboard()
