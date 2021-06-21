import random


def random_active_learning(unlabeled_dataset, annotation_budget):
    """
    Random sampler for active learning
    :param unlabeled_dataset: the unlabeled dataset
    :param annotation_budget: the number of samples to pull from the dataset
    :return: the indexes of the samples to retrieve
    """
    try:
        return sorted(random.sample(range(len(unlabeled_dataset)), annotation_budget))
    except ValueError:
        # annotation budget larger than remaining samples
        return list(range(len(unlabeled_dataset)))


def discriminative_active_learning(unlabeled_dataset, annotation_budget):
    """
    Discriminative Active Learning
    :param unlabeled_dataset: the unlabeled dataset
    :param annotation_budget: the number of samples to pull from the dataset
    :return: the indexes of the samples to retrieve 
    """
    pass
