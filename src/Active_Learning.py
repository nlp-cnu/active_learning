import random
# from ADE_Detector import ADE_Detector
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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

    print(len(ux), len(idxs), max(idxs))

    for idx in sorted(idxs, reverse=True):
        ux.pop(idx)
    np.delete(uy, idxs)

    print(len(ux))

    return (lx, ly), (ux, uy)


def discriminative_active_learning(labeled, unlabeled, annotation_budget):
    """
    Discriminative Active Learning
    :param unlabeled: the unlabeled dataset
    :param annotation_budget: the number of samples to pull from the dataset
    :return: the indexes of the samples to retrieve 
    """

    lx, ly = labeled
    ux, uy = unlabeled

    if len(lx) == 0:
        return random_active_learning(labeled, unlabeled, annotation_budget)

    x = lx + ux
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(np.concatenate([np.ones(len(lx)), np.zeros(len(ux))]).reshape(-1, 1))

    # classifier = ADE_Detector()
    # classifier.fit(x, y)

    return (lx, ly), (ux, uy)


if __name__ == '__main__':
    from Dataset import Dataset
    db = Dataset()

    lx, ly = [], np.array([])
    ux, uy = db.get_train_data()

    budget = 10
    for _ in range(10):
        (lx, ly), (ux, uy) = random_active_learning((lx, ly), (ux, uy), budget)
        # print(ly)
