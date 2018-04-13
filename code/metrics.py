import numpy as np


def _separate_pos_neg(y_true, y_pred, is_l2):
    if is_l2:
        sort_order = y_pred.argsort()
        y_true_pos = y_true[sort_order]
        y_true_neg = np.flip(1 - y_true[sort_order], 0)
    else:
        sort_order = (-y_pred).argsort()
        y_pred = y_pred[sort_order]
        y_true = y_true[sort_order]
        y_true_pos = y_true[y_pred > 0.5]
        y_true_neg = np.flip(1 - y_true[y_pred < 0.5], 0)
    return y_true_pos, y_true_neg


def _get_effective_k(y_true_pos, y_true_neg, k):
    if k is None:
        k_pos = len(y_true_pos)
        k_neg = len(y_true_neg)
    else:
        k_pos = min(k, len(y_true_pos))
        k_neg = min(k, len(y_true_neg))
    return k_pos, k_neg


def hits_at_k(y_true, y_pred=None, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message
    Hits @ k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: ndarray having shape (num_examples, ) indicating whether the mail was sent by the s-r pair
    :param y_pred: ndarray having shape (num_examples, ) containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: (<positive_hits_at_k>, <k_pos>, <negative_hits_at_k>, <k_neg>)
    """
    y_true_pos, y_true_neg = _separate_pos_neg(y_true, y_pred, is_l2)
    k_pos, k_neg = _get_effective_k(y_true_pos, y_true_neg, k)

    hits_pos = np.sum(y_true_pos[:k_pos])
    hits_neg = np.sum(y_true_neg[:k_neg])
    return hits_pos, k_pos, hits_neg, k_neg


def _calc_average_precision(y_true, k):
    correct = 0
    ap = 0.0
    for i in xrange(k):
        correct += y_true[i]
        precision = correct / (i + 1.0)
        ap += precision * y_true[i]
    return ap / min(k, np.asscalar(np.sum(y_true)))


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
    y_true_pos, y_true_neg = _separate_pos_neg(y_true, y_pred, is_l2)
    k_pos, k_neg = _get_effective_k(y_true_pos, y_true_neg, k)

    ap_pos = _calc_average_precision(y_true_pos, k_pos)
    ap_neg = _calc_average_precision(y_true_neg, k_neg)
    return ap_pos, k_pos, ap_neg, k_neg


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
    mean_ap_pos, mean_ap_neg = 0.0, 0.0
    for user_y_true, user_y_pred in zip(y_true, y_pred):
        ap_pos, _, ap_neg, _ = average_precision_at_k(user_y_true, user_y_pred, k, is_l2)
        mean_ap_pos += ap_pos
        mean_ap_neg += ap_neg
    return mean_ap_pos / len(y_true), mean_ap_neg / len(y_true)


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