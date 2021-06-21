from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb


def plot_test_graph(baseline_f1, random_al_f1_list, num_samples_list, file_name):
    """
    Plots a graph of baseline performance to a random AL model
    :param baseline_f1:
    :param random_al_f1_list:
    :param num_samples_list:
    :param file_name: File to save plot to
    """
    print(baseline_f1)
    print(random_al_f1_list)
    print(num_samples_list)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("F1-Score Over Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("F1-Score")

    ax.hlines(baseline_f1, xmin=min(num_samples_list), xmax=max(num_samples_list),
              linestyles='dashed', label='Full Dataset', color='green')

    ax.plot(num_samples_list, random_al_f1_list, label='Random AL', color='red')
    ax.legend(loc='lower right')

    plt.show()
    fig.savefig(file_name)


def plot_from_tensorboard():
    experiment_id = "C5coRObqRkGYFX9q0Pyd9g"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    f = df['run'].str.contains('Base_BERT')
    df = df[f]
    f = df['run'].str.endswith('validation')
    df = df[f]

    data = pd.DataFrame(columns=['run', 'step', 'epoch_loss', 'epoch_positive_class_F1'])
    for idx, (run, tag, step, value) in df.iterrows():
        if run not in list(data['run']) or step not in list(data.loc[data['run'] == run]['step']):
            data = data.append({'run': run, 'step': step, tag: value}, ignore_index=True)
        else:
            data.loc[(data['run'] == run) & (data['step'] == step), [tag]] = value

    print(data)

    dfw_validation = data

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=dfw_validation, x="step", y="epoch_positive_class_F1").set_title("Positive F1")

    plt.subplot(1, 2, 2)
    sns.lineplot(data=dfw_validation, x="step", y="epoch_loss").set_title("Loss")

    plt.show()



if __name__ == '__main__':
    # plot_test_graph(0.5, [0.1, 0.2, 0.3, 0.4, 0.5], [100, 200, 300, 400, 500], '../test_plot.png')
    plot_from_tensorboard()