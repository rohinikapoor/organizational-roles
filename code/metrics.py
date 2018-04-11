import numpy as np


def hits_at_k(y_true, y_pred=None, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message
    Hits @ k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: ndarray having shape (num_examples, ) indicating whether the mail was sent by the s-r pair
    :param y_pred: ndarray having shape (num_examples, ) containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: <int> the hits @ k metric
    """
    if k is None:
        k = len(y_true)
    else:
        k = min(k, len(y_true))

    if is_l2:
        sort_order = y_pred.argsort()
        y_true = y_true[sort_order]
        correct = 0
        for i in xrange(k):
            correct += y_true[i]
        return correct, k
    else:
        # This does not have clear semantics if we are given the binary probabilities
        return np.sum(y_true[y_pred > 0.5]), k


def average_precision_at_k(y_true, y_pred=None, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message
    Average precision at k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: ndarray having shape (num_examples, ) indicating whether the mail was sent by the s-r pair
    :param y_pred: ndarray having shape (num_examples, ) containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: <float> the average precision @ k
    """
    if k is None:
        k = len(y_true)
    else:
        k = min(k, len(y_true))

    if is_l2:
        sort_order = y_pred.argsort()
    else:
        sort_order = (-y_pred).argsort()

    y_true = y_true[sort_order]
    correct = 0
    ap = 0.0
    for i in xrange(k):
        correct += y_true[i]
        precision = correct / (i+1.0)
        ap += precision * y_true[i]
    return ap / np.sum(y_true), k


def mean_average_precision_at_k(y_true, y_pred, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message for Q users
    Mean average precision at k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: iterable of length Q, each element is an ndarray having shape (num_examples, )
                                            indicating whether the mail was sent by the s-r pair
    :param y_pred: iterable of length Q, each element is an ndarray having shape (num_examples, )
                                            containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: <float> the mean average precision @ k for all Q users
    """
    mean_ap = 0.0
    for user_y_true, user_y_pred in zip(y_true, y_pred):
        ap, _ = average_precision_at_k(user_y_true, user_y_pred, k, is_l2)
        mean_ap += ap
    return mean_ap / len(y_true)


if __name__ == '__main__':
    y_true = np.array([1, 0, 1, 0, 1])
    # expected sort order -> 1 1 0 0 1
    y_pred_prob = np.array([0.6, 0.4, 0.7, 0.55, 0.0])
    y_pred_l2 = np.array([10, 30, 20, 40, 50])

    # 3 on 5
    print hits_at_k(y_true, y_pred_l2, is_l2=True)
    print hits_at_k(y_true, y_pred_prob, is_l2=False)

    # (1/1 + 2/2 + 3/5) / 3
    print average_precision_at_k(y_true, y_pred_l2, is_l2=True)
    print average_precision_at_k(y_true, y_pred_prob, is_l2=False)

    y2_true = np.array([1, 1, 0, 1, 1])
    # expected sort order -> 1 0 1 1 1
    y2_pred_prob = np.array([0.7, 0.4, 0.6, 0.55, 0.0])
    y2_pred_l2 = np.array([10, 30, 20, 40, 50])

    # (1/1 + 2/3 + 3/4 + 4/5) / 4
    print average_precision_at_k(y2_true, y2_pred_l2, is_l2=True)
    print average_precision_at_k(y2_true, y2_pred_prob, is_l2=False)

    # (0.866 + 0.804) / 2
    print mean_average_precision_at_k([y_true, y2_true], [y_pred_l2, y2_pred_l2], is_l2=True)
    print mean_average_precision_at_k([y_true, y2_true], [y_pred_prob, y2_pred_prob], is_l2=False)