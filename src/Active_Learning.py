import os

import numpy as np


def random_active_learning(labeled, unlabeled, annotation_budget):
    """
    Random sampler for active learning
    :param unlabeled: the unlabeled dataset
    :param annotation_budget: the number of samples to pull from the dataset
    :return: new labeled and unlabeled datasets
    """
    lx, ly = labeled
    ux, uy = unlabeled

    rng = np.random.default_rng()

    try:
        idxs = rng.choice(len(uy), annotation_budget, replace=False)
    except ValueError:
        # annotation budget larger than remaining samples
        idxs = np.arange(len(uy))

    os.path.join('..', 'active_learning_scores', f'random_{annotation_budget}_selected_samples.tsv')

    lx += [ux[idx] for idx in idxs]
    ly = np.concatenate([ly, uy[idxs]]) if len(ly) != 0 else uy[idxs]

    for idx in sorted(idxs, reverse=True):
        ux.pop(idx)
    uy = np.delete(uy, idxs, axis=0)

    return (lx, ly), (ux, uy)


def discriminative_active_learning(labeled, unlabeled, annotation_budget, model=None, mini_queries=10):
    """
    Discriminative Active Learning
    :param unlabeled: the unlabeled dataset
    :param annotation_budget: the number of samples to pull from the dataset
    :param model: Binary classifier for DAL
    :param mini_queries: Number of samples to select before retraining model
    :return: the new labeled and unlabeled datasets
    """
    if model is None:
        raise ValueError("Model must not be Null")

    lx, ly = labeled
    ux, uy = unlabeled

    # annotation budget greater than remaining samples
    if len(ux) < annotation_budget:
        return random_active_learning(labeled, unlabeled, annotation_budget)

    for i in range(mini_queries):
        print(f'Selecting samples with DAL: {i + 1}/{mini_queries}')

        # train classifier
        x = lx + ux
        y = np.array([[1, 0] for _ in range(len(lx))] + [[0, 1] for _ in range(len(ux))])

        model.fit(x, y, use_class_weights=False)

        preds = model.predict(ux)

        batch_selection = annotation_budget // mini_queries
        selected_samples_path = os.path.join('..', 'active_learning_scores', f'DAL_{annotation_budget}_selected_samples.tsv')
        with open(selected_samples_path, 'a', encoding='utf8') as f:
            for _ in range(batch_selection):
                max_idx = np.argmax(preds, axis=0)[1]
                preds = np.delete(preds, max_idx, axis=0)

                sample = ux.pop(max_idx)
                label = uy[max_idx][1]

                f.write(f'{sample}\t{label}\n')

                lx.append(sample)
                ly = np.concatenate((ly, [uy[max_idx]])) if len(ly) != 0 else np.array([uy[max_idx]])
                uy = np.delete(uy, max_idx, axis=0)

            f.write('-' * 30 + '\n')

        model.reset_model()

    return (lx, ly), (ux, uy)


def main():
    from Dataset import Dataset
    db = Dataset()

    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    budget = 500
    while len(ux) > 0:
        (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)
        print(len(ux))


if __name__ == '__main__':
    main()
