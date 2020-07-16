# Wrapper for the shortest augmenting path algorithm for solving the
# rectangular linear sum assignment problem.  The original code was an
# implementation of the Hungarian algorithm (Kuhn-Munkres) taken from
# scikit-learn, based on original code by Brian Clapper and adapted to NumPy
# by Gael Varoquaux. Further improvements by Ben Root, Vlad Niculae, Lars
# Buitinck, and Peter Larsen.
#
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD

import numpy as np
from . import _lsap_module

# TODO string 'method' parameter or bool 'approx' parameter?


def linear_sum_assignment(cost_matrix, maximize=False, method='exact'):
    """Solve the linear sum assignment problem.

    The linear sum assignment problem is also known as minimum weight matching
    in bipartite graphs. A problem instance is described by a matrix C, where
    each C[i,j] is the cost of matching vertex i of the first partite set
    (a "worker") and vertex j of the second set (a "job"). The goal is to find
    a complete assignment of workers to jobs of minimal cost.

    Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is
    assigned to column j. Then the optimal assignment has cost

    .. math::
        \\min \\sum_i \\sum_j C_{i,j} X_{i,j}

    where, in the case where the matrix X is square, each row is assigned to
    exactly one column, and each column to exactly one row.

    This function can also solve a generalization of the classic assignment
    problem where the cost matrix is rectangular. If it has more rows than
    columns, then not every row needs to be assigned to a column, and vice
    versa.

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    maximize : bool (default: False)
        Calculates a maximum weight matching if true.

    method :
    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.

    Notes
    -----
    .. versionadded:: 0.17.0

    References
    ----------

    1. https://en.wikipedia.org/wiki/Assignment_problem

    2. DF Crouse. On implementing 2D rectangular assignment algorithms.
       *IEEE Transactions on Aerospace and Electronic Systems*,
       52(4):1679-1696, August 2016, https://doi.org/10.1109/TAES.2016.140952

    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> from scipy.optimize import linear_sum_assignment
    >>> row_ind, col_ind = linear_sum_assignment(cost)
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind].sum()
    5
    """
    cost_matrix = np.asarray(cost_matrix)
    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-D array), got a %r array"
                         % (cost_matrix.shape,))

    if not (np.issubdtype(cost_matrix.dtype, np.number) or
            cost_matrix.dtype == np.dtype(np.bool_)):
        raise ValueError("expected a matrix containing numerical entries, got %s"
                         % (cost_matrix.dtype,))

    if method == 'approx':
        return _alap(cost_matrix, maximize)
    else:
        if maximize:
            cost_matrix = -cost_matrix

        if np.any(np.isneginf(cost_matrix) | np.isnan(cost_matrix)):
            raise ValueError("matrix contains invalid numeric entries")

        cost_matrix = cost_matrix.astype(np.double)
        a = np.arange(np.min(cost_matrix.shape))
        # The algorithm expects more columns than rows in the cost matrix.
        if cost_matrix.shape[1] < cost_matrix.shape[0]:
            b = _lsap_module.calculate_assignment(cost_matrix.T)
            indices = np.argsort(b)
            return (b[indices], a[indices])
        else:
            b = _lsap_module.calculate_assignment(cost_matrix)
            return (a, b)


def _alap(cost_matrix, maximize):
    if not maximize:
        cost_matrix = -cost_matrix
    n_row, n_col = cost_matrix.shape
    n = n_row + n_col
    matched = np.empty(n) * np.nan
    cv = np.zeros(n)
    col_argmax = np.argmax(cost_matrix, axis=0)
    row_argmax = np.argmax(cost_matrix, axis=1)

    # remove full zero rows and columns (match them)

    cv[:n_col] = col_argmax + n_col  # first half points to second, vice versa
    cv[n_col:] = row_argmax
    cv = cv.astype(int)

    dom_ind = (cv[cv] == np.arange(n))
    matched[dom_ind] = cv[dom_ind]  # matched indices, everywhere else nan
    qc, = np.nonzero(dom_ind)  # dominating vertices

    while len(qc) > 0 and np.isnan(
            matched[n_col:]).any():  # loop while qc not empty, ie new matchings still being found

        temp = np.arange(n)[np.in1d(cv, qc)]  # indices of qc in cv
        qt = temp[~np.in1d(temp, matched[qc])]  # indices of unmatched verts in cv and qc

        qt_p = qt[qt >= n_col]
        qt_n = qt[qt < n_col]

        m_row = np.arange(n_row)[np.isnan(matched[n_col:])]  # unmatched rows to check
        m_col = np.arange(n_col)[np.isnan(matched[:n_col])]  # unmatched cols

        col_argmax = np.argmax(cost_matrix[np.ix_(m_row, qt_n)], axis=0)
        row_argmax = np.argmax(cost_matrix[np.ix_(qt_p - n_col, m_col)], axis=1)

        col_argmax = m_row[col_argmax]
        row_argmax = m_col[row_argmax]

        cv[qt_n] = col_argmax + n_col
        cv[qt_p] = row_argmax
        cv = cv.astype(int)

        dom_ind = (cv[cv[qt]] == qt)
        qt = qt[dom_ind]
        matched[qt] = cv[qt]  # adding new dominating indices to matching
        matched[cv[qt]] = qt

        qn = np.zeros(n)  # store new matchings
        qn[qt] = qt
        qn[cv[qt]] = cv[qt]
        qc = qn[qn > 0].astype(int)

    matching = matched[n_col:]
    rows = np.arange(n_row)[~np.isnan(matching)]
    matching = matching[~np.isnan(matching)].astype(int)
    return (rows, matching)
