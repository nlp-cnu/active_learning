import random

import numpy as np

from ADE_Detector import ADE_Detector


def random_active_learning(labeled, unlabeled, annotation_budget):
    """
    Random sampler for active learning
    :param unlabeled: the unlabeled dataset
    :param annotation_budget: the number of samples to pull from the dataset
    :return: new labeled and unlabeled datasets
    """
    lx, ly = labeled
    ux, uy = unlabeled

    try:
        idxs = sorted(random.sample(range(len(uy)), annotation_budget))
    except ValueError:
        # annotation budget larger than remaining samples
        idxs = list(range(len(uy)))

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
    classifier = ADE_Detector() if model is None else model

    lx, ly = labeled
    ux, uy = unlabeled

    if len(lx) == 0 or len(ux) < annotation_budget:
        return random_active_learning(labeled, unlabeled, annotation_budget)

    for i, _ in enumerate(range(mini_queries)):
        print(f'Selecting samples with DAL: {i + 1}/{mini_queries}')

        # train classifier
        x = lx + ux
        y = np.array([[1, 0] for _ in range(len(lx))] + [[0, 1] for _ in range(len(ux))])

        classifier.fit(x, y)

        batch_selection = annotation_budget // mini_queries
        for _ in range(batch_selection):
            preds = classifier.predict(ux)
            max_idx = np.argmax(preds, axis=0)[1]

            lx.append(ux.pop(max_idx))

            ly = np.concatenate((ly, [uy[max_idx]])) if len(ly) != 0 else np.array([uy[max_idx]])
            uy = np.delete(uy, max_idx, axis=0)

        classifier.reset_model()

    return (lx, ly), (ux, uy)


def main():
    from Dataset import Dataset
    db = Dataset()

    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    budget = 100
    for _ in range(10):
        (lx, ly), (ux, uy) = discriminative_active_learning((lx, ly), (ux, uy), budget)
        num_pos = 0
        num_neg = 0
        for sample in ly:
            if sample[0] == 0:
                num_pos += 1
            else:
                num_neg += 1

        print(f"{num_neg} negative samples, {num_pos} positive samples")


if __name__ == '__main__':
    main()
