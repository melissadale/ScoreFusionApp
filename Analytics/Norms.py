import math

import numpy as np
from scipy import stats
import statsmodels.api as sm


def my_minmax(train, test):
    """
    zi = (xi-min(x))/(max(x)-min(x))
    :param x:
    :return:
    """
    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    min_x = train.min()
    max_x = train.max()

    for i in range(len(train)):
        train_transformed[i] = (train[i]-min_x)/(max_x-min_x)

    for i in range(len(test)):
        if test[i] > max_x:
            test[i] = max_x
        if test[i] < min_x:
            test[i] = min_x

        test_transformed[i] = (test[i]-min_x)/(max_x-min_x)

    assert np.min(train_transformed) == 0.0
    assert np.max(train_transformed) == 1.0

    return [train_transformed, test_transformed]


def my_zscore(train, test):
    """
    z = (x - u) / s
    :param x:
    :return:
    """
    # tmp = train.stack()
    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    u = train.mean()
    s = train.std()

    for i in range(len(train)):
        train_transformed[i] = (train[i]-u)/(s)

    for i in range(len(test)):
        test_transformed[i] = (test[i]-u)/(s)

    return [train_transformed, test_transformed]


def my_med_mad(train, test):
    """
    z = (s-median)/MAD
    (MAD = median(|s-median|))
    :param x:
    :return:
    """
    # tmp = train.stack()
    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    mad = stats.median_absolute_deviation(train)
    med = np.median(train)

    for i in range(len(train)):
        train_transformed[i] = (train[i]-med)/(mad)

    for i in range(len(test)):
        test_transformed[i] = (test[i]-med)/(mad)

    return [train_transformed, test_transformed]


def my_decimal(train, test):
    """
    z = s/10^n,
    n = log_10(max(si)))
    :param x:
    :return:
    """
    # tmp = train.stack()
    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    maxx = train.max()
    n = math.log10(maxx)

    for i in range(len(train)):
        train_transformed[i] = (train[i])/(10**n)

    for i in range(len(test)):
        test_transformed[i] = (test[i])/(10**n)

    return [train_transformed, test_transformed]


def my_double_sigmoid(train, test, t, r1, r2):
    """
    :param x:
    :return:
    """
    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    for i in range(len(train)):
        if train[i] < t:
            train_transformed[i] = 1/(1+np.exp(-2((train[i]-t)/r1)))
        else:
            train_transformed[i] = 1/(1+np.exp(-2((train[i]-t)/r2)))

    for i in range(len(test)):
        if test[i] < t:
            test_transformed[i] = 1/(1+np.exp(-2((test[i]-t)/r1)))
        else:
            test_transformed[i] = 1/(1+np.exp(-2((test[i]-t)/r2)))
    return [train_transformed, test_transformed]


def my_biweight(train, test):
    """
    z = s/10^n,
    n = log_10(max(si)))
    :param x:
    :return:
    """

    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    psi = sm.robust.norms.TukeyBiweight(c=4.685).psi(train)

    MUgh = psi.mean()
    SIGgh = psi.std()

    for i in range(len(train)):
        train_transformed[i] = 0.5 * (np.tan(0.01*((train[i] - MUgh) / SIGgh))+1)

    for i in range(len(test)):
        test_transformed[i] = 0.5 * (np.tan(0.01*((test[i] - MUgh) / SIGgh))+1)

    return [train_transformed, test_transformed]


def my_tanh(train, test, a, b, c):
    """
    z = 1/2{tanh((sk - MUgh)/SIGgh))+1}
    :param x:
    :return:
    """
    train_transformed = np.zeros(shape=train.shape)
    test_transformed = np.zeros(shape=test.shape)

    psi = sm.robust.norms.Hampel(a=a, b=b, c=c).psi(train)

    MUgh = psi.mean()
    SIGgh = psi.std()

    for i in range(len(train)):
        train_transformed[i] = 0.5 * (np.tan(0.01*((train[i] - MUgh) / SIGgh))+1)

    for i in range(len(test)):
        test_transformed[i] = 0.5 * (np.tan(0.01*((test[i] - MUgh) / SIGgh))+1)

    return [train_transformed, test_transformed]
