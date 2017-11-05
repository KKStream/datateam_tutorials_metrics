"""
a sample implementation of 'area under roc' metric.
"""


def auc(guess, truth):
    """
    calculate area under the receiver operating characteristic

    guess:
        binary classification in probabilities:
        [0.0, 0.2, 0.9, ...., 0.99]
    truth:
        binary classification:
        [0, 1, 0, ..., 1]

    return:
        area under roc
    """
    assert len(guess) == len(truth), 'len(guess) == len(truth) should be true'

    positive_num = sum(truth)
    negative_num = len(truth) - positive_num

    # [g0, g1, g2, ..., gn] => [(0, g0), (1, g1), (2, g2), ..., (n, gn)]
    guess = [(i, v) for i, v in enumerate(guess)]

    # [(0, g0), (1, g1), (2, g2), ..., (n, gn)] =>
    # [(1, g1), (0, g0), (2, g2), ..., (x, gx)]
    # sort based on probabilities of guess, ascending
    guess = sorted(guess, key=lambda x: x[1])

    # when threshold is 0.0, everything is positive!
    # => all positive predictions are true positive
    #    true positive rate is 1.0.
    # => all negative predictions are false positive
    #    false positive rate is 1.0.
    t_positive_num = positive_num
    f_positive_num = negative_num

    t_positive_rate = 1.0
    f_positive_rate = 1.0

    # for accumulating area pieces.
    auc_value = 0.0

    # process guess one by one.
    # NOTE: since guess is sorted, we are increasing 'threshold' while
    #       processing 'guess' in ascending order.
    for g in guess:
        if truth[g[0]]:
            # if g is true (in ground truth)
            # => true positive rate decreases
            # => false positive rate remains
            # NOTE: since the false positive rate remains, there is no new auc
            #       piece for accumulating.
            t_positive_num -= 1

            t_positive_rate = t_positive_num / float(positive_num)
        else:
            # if g is false (in ground truth)
            # => true positive rate remains
            # => false positive rate decreases
            f_positive_num -= 1

            f_positive_rate_temp = f_positive_num / float(negative_num)

            # NOTE: there is a new piece to accumulate since false positive
            #       rate changes.
            auc_value += \
                (f_positive_rate - f_positive_rate_temp) * t_positive_rate

            f_positive_rate = f_positive_rate_temp

    return auc_value


if __name__ == '__main__':
    import random

    # labels of a binary classification problem.
    # a list contains only 0 or 1.
    truth = [random.randint(0, 1) for _ in range(1000000)]

    # probabilities (predictions) of a binary classification problem.
    # a list contains float between 0.0 and 1.0.
    guess = [random.uniform(0.0, 1.0) for _ in range(1000000)]

    # guess is random, so the auc should be around 0.5
    print auc(guess, truth)

    # 100% correct predictions should give 1.0 auc.
    print auc(truth, truth)
