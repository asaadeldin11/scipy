import numpy as np
import operator
from . import (linear_sum_assignment, minimize_scalar, OptimizeResult)
from .optimize import _check_unknown_options

from scipy._lib._util import check_random_state
import itertools


def quadratic_assignment(A, B, method="faq", options=None):
    r"""
    Solve the quadratic assignment problem.

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP).

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices. For the default
    algorithm :ref:`'faq' <optimize.qap-faq>`, all elements of
    :math:`A` and :math:`B` must be non-negative.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    differences of edge weight disagreements.

    Note that the quadratic assignment problem is NP-hard, is not
    known to be solvable in polynomial time, and is computationally
    intractable. Therefore, the results given are approximations,
    not guaranteed to be exact solutions.


    Parameters
    ----------
    A : 2d-array, square
        The square matrix :math:`A` in the objective function above.
        Elements must be non-negative for method
        :ref:`'faq' <optimize.qap-faq>`.

    B : 2d-array, square
        The square matrix :math:`B` in the objective function above.
        Elements must be non-negative for method
        :ref:`'faq' <optimize.qap-faq>`.

    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem.
        :ref:`'faq' <optimize.qap-faq>` (default) and
        :ref:`'2opt' <optimize.qap-2opt>` are available.

    options : dict, optional
        A dictionary of solver options. All solvers support the following:

        partial_match : 2d-array of integers, optional, (default = None)
            Allows the user to fix part of the matching between the two
            matrices. In the literature, a partial match is also
            known as a "seed" [2]_.

            Each row of `partial_match` specifies the indices of a pair of
            corresponding nodes, that is, node ``partial_match[i, 0]` of `A` is
            matched to node ``partial_match[i, 1]`` of `B`. Accordingly,
            ``partial_match`` is an array of size ``(m , 2)``, where ``m`` is
            less than the number of nodes.

        maximize : bool (default = False)
            Setting `maximize` to ``True`` solves the Graph Matching Problem
            (GMP) rather than the Quadratic Assingnment Problem (QAP).

        rng : {None, int, `~np.random.RandomState`, `~np.random.Generator`}
            This parameter defines the object to use for drawing random
            variates.
            If `rng` is ``None`` the `~np.random.RandomState` singleton is
            used.
            If `rng` is an int, a new ``RandomState`` instance is used,
            seeded with `rng`.
            If `rng` is already a ``RandomState`` or ``Generator``
            instance, then that object is used.
            Default is None.

        For method-specific options, see
        :func:`show_options('quadratic_assignment') <show_options>`.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` containing the following
        fields.

        col_ind : 1-D array
            An array of column indices corresponding with the best
            permutation of the nodes of `B` found.
        score : float
            The corresponding value of the objective function.
        nit : int
            The number of iterations performed during optimization.

    Notes
    -----
    The default method :ref:`'faq' <optimize.qap-faq>` uses the Fast
    Approximate QAP algorithm [1]_; it is typically offers the best
    combination of speed and accuracy.
    Method :ref:`'2opt' <optimize.qap-2opt>` can be computationally expensive,
    but may be a useful alternative, or it can be used to refine the solution
    returned by another method.

    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           https://doi.org/10.1371/journal.pone.0121002

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, https://doi.org/10.1016/j.patcog.2018.09.014

    .. [3] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import quadratic_assignment
    >>> A = np.array([[0, 80, 150, 170], [80, 0, 130, 100],
    ...              [150, 130, 0, 120], [170, 100, 120, 0]])
    >>> B = np.array([[0, 5, 2, 7], [0, 0, 3, 8],
    ...              [0, 0, 0, 3], [0, 0, 0, 0]])
    >>> res = quadratic_assignment(A, B)
    >>> print(res)
     col_ind: array([0, 3, 2, 1])
         nit: 9
       score: 3260

    The see the relationship between the returned ``col_ind`` and ``score``,
    use ``col_ind`` to form the best permutation matrix found, then evaluate
    the objective function :math:`f(P) = trace(A^T P B P^T )`.

    >>> n = A.shape[0]
    >>> perm = res['col_ind']
    >>> P = np.eye(n)[perm]
    >>> score = int(np.trace(A.T @ P @ B @ P.T))
    >>> print(score)
    3260

    Alternatively, to avoid constructing the permutation matrix explicitly,
    directly permute the rows and columns of the distance matrix.

    >>> score = np.trace(A.T @ B[perm][:, perm])
    >>> print(score)
    3260

    Although not guaranteed in general, ``quadratic_assignment`` happens to
    have found the globally optimal solution.

    >>> from itertools import permutations
    >>> perm_opt, score_opt = None, np.inf
    >>> for perm in permutations([0, 1, 2, 3]):
    ...     perm = np.array(perm)
    ...     score = int(np.trace(A.T @ B[perm][:, perm]))
    ...     if score < score_opt:
    ...         score_opt, perm_opt = score, perm
    >>> print(np.equal(perm_opt, res['col_ind']))
    True
    """

    if options is None:
        options = {}

    method = method.lower()
    methods = {"faq": _quadratic_assignment_faq,
               "2opt": _quadratic_assignment_2opt}
    if method not in methods:
        raise ValueError(f"method {method} must be in {methods}.")
    res = methods[method](A, B, **options)
    return res


def _calc_score(A, B, perm):
    # equivalent to objective function but avoids matmul
    return np.sum(A * B[perm][:, perm])


def _common_input_validation(A, B, partial_match, maximize):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    if partial_match is None:
        partial_match = np.array([[], []]).T
    partial_match = np.atleast_2d(partial_match)

    msg = None
    if A.shape[0] != A.shape[1]:
        msg = "`A` must be square"
    elif B.shape[0] != B.shape[1]:
        msg = "`B` must be square"
    elif A.ndim != 2 or B.ndim != 2:
        msg = "`A` and `B` must have exactly two dimensions"
    elif A.shape != B.shape:
        msg = "`A` and `B` matrices must be of equal size"
    elif partial_match.shape[0] > A.shape[0]:
        msg = "There cannot be more seeds than there are nodes"
    elif partial_match.shape[1] != 2:
        msg = "`partial_match` must have two columns"
    elif partial_match.ndim != 2:
        msg = "`partial_match` must have exactly two dimensions"
    elif (partial_match < 0).any():
        msg = "`partial_match` contains negative entries"
    elif (partial_match >= len(A)).any():
        msg = "`partial_match` entries must be less than number of nodes"
    elif not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or not \
            len(set(partial_match[:, 1])) == len(partial_match[:, 1]):
        msg = "`partial_match` column entries must be unique"

    if msg is not None:
        raise ValueError(msg)

    return A, B, partial_match


def _quadratic_assignment_faq(A, B,
                              maximize=False, partial_match=None, rng=None,
                              init_J="barycenter", init_weight=None, init_k=1,
                              maxiter=30, shuffle_input=True, eps=0.05,
                              **unknown_options
                              ):
    r"""
    Solve the quadratic assignment problem (approximately).

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the Fast Approximate QAP Algorithm
    (FAQ) [1]_.

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices with non-negative elements.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    differences of edge weight disagreements.

    Note that the quadratic assignment problem is NP-hard, is not
    known to be solvable in polynomial time, and is computationally
    intractable. Therefore, the results given are approximations,
    not guaranteed to be exact solutions.


    Parameters
    ----------
    A : 2d-array, square, non-negative
        The square matrix :math:`A` in the objective function above.

    B : 2d-array, square, non-negative
        The square matrix :math:`B` in the objective function above.

    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for 'faq'.
        :ref:`'2opt' <optimize.qap-2opt>` is also available.

    Options
    -------

    partial_match : 2d-array of integers, optional, (default = None)
        Allows the user to fix part of the matching between the two
        matrices. In the literature, a partial match is also known as a
        "seed".

        Each row of `partial_match` specifies the indices of a pair of
        corresponding nodes, that is, node ``partial_match[i, 0]`` of `A` is
        matched to node ``partial_match[i, 1]`` of `B`. Accordingly,
        ``partial_match`` is an array of size ``(m , 2)``, where ``m`` is
        less than the number of nodes.

    maximize : bool (default = False)
        Setting `maximize` to ``True`` solves the Graph Matching Problem (GMP)
        rather than the Quadratic Assingnment Problem (QAP). This is
        accomplished through trivial negation of the objective function.

    init_J : 2d-array or "barycenter" (default = "barycenter")
        The initial (guess) permutation matrix or search "position"
        :math:`J`.

        :math:`J` need not be a proper permutation matrix;
        however, it must have the same shape as `A` and
        `B`, and it must be doubly stochastic: each of its
        rows and columns must sum to 1.

        If unspecified or ``"barycenter"``, the non-informative "flat
        doubly stochastic matrix" :math:`1*1^T/n`, where :math:`n`
        is the number of nodes and :math:`1` is a :math:`n \times 1`
        array of ones, is used. This is the "barycenter" of the
        search space of doubly-stochastic matrices.
    init_weight : float in range [0, 1]
        Allows the user to specify the weight of the provided
        search position :math:`J` relative to random perturbations
        :math:`K`.

        The algorithm will be repeated :math:`k` times
        from randomized initial search positions
        :math:`P_0 = (\alpha J + (1- \alpha) K`, where
        :math:`J` is given by option `init_J`,
        :math:`\alpha` is given by option `init_weight`,
        :math:`k` is given by option `init_k`, and
        :math:`K` is a random doubly stochastic matrix.

        Default is 1 if `init_k` is 1, 0 otherwise.
    init_k : int, positive (default = 1)
        The number of randomized initial search positions :math:`P_0`
        from which the FAQ algorithm will proceed.
    maxiter : int, positive (default = 30)
        Integer specifying the max number of Franke-Wolfe iterations
        per initial search position. FAQ typically converges within a
        modest number of iterations.
    shuffle_input : bool (default = True)
        To avoid artificially high or low matching due to inherent
        sorting of input matrices, gives users the option
        to shuffle the nodes. Results are then unshuffled so that the
        returned results correspond with the node order of inputs.
    eps : float (default = 0.05)
        A threshold for the stopping criterion. Franke-Wolfe
        iteration terminates when the change in search position between
        iterations is sufficiently small, that is, when the Frobenius
        norm of :math:`(P_{i}-P_{i+1}) \leq eps`, where :math:`i` is
        the iteration number.
    rng : {None, int, `~np.random.RandomState`, `~np.random.Generator`}
        This parameter defines the object to use for drawing random
        variates.
        If `rng` is ``None`` the `~np.random.RandomState` singleton is
        used.
        If `rng` is an int, a new ``RandomState`` instance is used,
        seeded with `rng`.
        If `rng` is already a ``RandomState`` or ``Generator``
        instance, then that object is used.
        Default is None.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` containing the following
        fields.

        col_ind : 1-D array
            An array of column indices corresponding with the best
            permutation of the nodes of `B` found.
        score : float
            The corresponding value of the objective function.
        nit : int
            The number of Franke-Wolfe iterations performed during
            the initialization resulting in the permutation
            returned.

    Notes
    -----
    The algorithm may be sensitive to the initial permutation matrix (or
    search "position") due to the possibility of several local minima
    within the feasible region. A barycenter initialization is more likely to
    result in a better solution than a single random initialization. However,
    use of several randomized initializations  (through `init_weight` and
    `init_k`) will likely result in a better solution at the cost of higher
    runtime.

    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           https://doi.org/10.1371/journal.pone.0121002

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, https://doi.org/10.1016/j.patcog.2018.09.014
    """

    _check_unknown_options(unknown_options)

    init_k = operator.index(init_k)
    if init_k > 1 and init_weight is None:
        init_weight = 0
    maxiter = operator.index(maxiter)

    # ValueError check
    A, B, partial_match = _common_input_validation(
            A, B, partial_match, maximize)

    msg = None
    if (A < 0).any() or (B < 0).any():
        msg = "`A` and `B` matrices must contain only non-negative elements."
    elif isinstance(init_J, str) and init_J not in {'barycenter'}:
        msg = "Invalid 'init_J' parameter string"
    elif init_weight is not None and (init_weight < 0 or init_weight > 1):
        msg = "'init_weight' must be strictly between zero and one"
    elif init_k <= 0:
        msg = "'init_k' must be a positive integer"
    elif maxiter <= 0:
        msg = "'maxiter' must be a positive integer"
    elif eps <= 0:
        msg = "'eps' must be a positive float"
    if msg is not None:
        raise ValueError(msg)

    rng = check_random_state(rng)
    n = A.shape[0]  # number of vertices in graphs
    n_seeds = partial_match.shape[0]  # number of seeds
    n_unseed = n - n_seeds

    perm_inds = np.zeros(n)

    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1
    best_score = obj_func_scalar * np.inf

    seed_dist_c = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        seed_dist_c = rng.permutation(seed_dist_c)
        # shuffle_input to avoid results from inputs that were already matched

    seed_cost_c = np.setdiff1d(range(n), partial_match[:, 0])
    permutation_cost = np.concatenate([partial_match[:, 0],
                                       seed_cost_c], axis=None).astype(int)
    permutation_dist = np.concatenate([partial_match[:, 1],
                                       seed_dist_c], axis=None).astype(int)
    A_orig, B_orig = A, B
    A = A[np.ix_(permutation_cost, permutation_cost)]
    B = B[np.ix_(permutation_dist, permutation_dist)]

    # definitions according to Seeded Graph Matching [2].
    A11 = A[:n_seeds, :n_seeds]
    A12 = A[:n_seeds, n_seeds:]
    A21 = A[n_seeds:, :n_seeds]
    A22 = A[n_seeds:, n_seeds:]
    B11 = B[:n_seeds, :n_seeds]
    B12 = B[:n_seeds, n_seeds:]
    B21 = B[n_seeds:, :n_seeds]
    B22 = B[n_seeds:, n_seeds:]

    # setting initialization matrix
    if isinstance(init_J, str) and init_J == 'barycenter':
        J = np.ones((n_unseed, n_unseed)) / float(n_unseed)
    else:
        _check_init_input(init_J, n_unseed)
        J = init_J

    total_iter = 0
    for i in range(init_k):
        if init_weight is not None:
            # generate a nxn matrix where each entry is a random number [0, 1]
            K = rng.random((n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            # initialize J, a doubly stochastic barycenter
            P = J * init_weight + (1 - init_weight) * K
        else:
            P = J
        const_sum = A21 @ B21.T + A12.T @ B12
        grad_P = np.inf  # gradient of P
        n_iter = 0  # number of FW iterations

        # OPTIMIZATION WHILE LOOP BEGINS
        while grad_P > eps and n_iter < maxiter:
            # computing the gradient of f(P) = -tr(APB^tP^t)
            delta_f = (const_sum + A22 @ P @ B22.T + A22.T @ P @ B22)
            # run hungarian algorithm on gradient(f(P))
            rows, cols = linear_sum_assignment(obj_func_scalar * delta_f)
            Q = np.zeros((n_unseed, n_unseed))
            Q[rows, cols] = 1  # initialize search direction matrix Q

            def f(x):  # computing the original optimization function
                xP1xQ = x * P + (1 - x) * Q
                # Sums below are np.trace(A11.T @ B11)
                # + np.trace(xP1xQ.T @ A21 @ B21.T)
                # + np.trace(xP1xQ.T @ A12.T @ B12)
                # + np.trace(A22.T @ xP1xQ @ B22 @ xP1xQ.T)
                # This is more efficient, but can we do even better?
                return obj_func_scalar * (
                    (A11 * B11).sum()
                    + (xP1xQ.T @ A21 * B21).sum()
                    + (xP1xQ.T @ A12.T * B12.T).sum()
                    + ((xP1xQ.T @ A22) * (B22 @ xP1xQ.T)).sum()
                )

            # computing the step size
            alpha = minimize_scalar(f, bounds=(0, 1), method="bounded").x
            P_i1 = alpha * P + (1 - alpha) * Q  # Update P
            grad_P = np.linalg.norm(P - P_i1)
            P = P_i1
            n_iter += 1
        # end of FW optimization loop

        # Project onto the set of permutation matrices
        row, col = linear_sum_assignment(-P)
        perm_inds_new = np.concatenate((np.arange(n_seeds), col + n_seeds))

        score = _calc_score(A, B, perm_inds_new)

        # minimizing
        if obj_func_scalar * score < obj_func_scalar * best_score:
            best_score = score
            perm_inds = np.zeros(n, dtype=int)
            perm_inds[permutation_cost] = permutation_dist[perm_inds_new]
            total_iter = n_iter

    best_score = _calc_score(A_orig, B_orig, perm_inds)
    res = {"col_ind": perm_inds, "score": best_score, "nit": total_iter}

    return OptimizeResult(res)


def _check_init_input(init_J, n):
    row_sum = np.sum(init_J, axis=0)
    col_sum = np.sum(init_J, axis=1)
    tol = 1e-3
    msg = None
    if init_J.shape != (n, n):
        msg = "`init_J` matrix must have same shape as A and B"
    elif ((~np.isclose(row_sum, np.ones(n), atol=tol)).any() or
          (~np.isclose(col_sum, np.ones(n), atol=tol)).any() or
          (init_J < 0).any()):
        msg = "`init_J` matrix must be doubly stochastic"
    if msg is not None:
        raise ValueError(msg)


def _doubly_stochastic(P, eps=1e-3):
    # cleaner implementation of btaba/sinkhorn_knopp

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if ((np.abs(P_eps.sum(axis=1) - 1) < eps).all() and
                (np.abs(P_eps.sum(axis=0) - 1) < eps).all()):
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


def _quadratic_assignment_2opt(A, B, maximize=False, partial_match=None,
                               rng=None, **unknown_options):
    r"""
    Solve the quadratic assignment problem (approximately).

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the 2-opt algorithm [3]_.

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    differences of edge weight disagreements.

    Note that the quadratic assignment problem is NP-hard, is not
    known to be solvable in polynomial time, and is computationally
    intractable. Therefore, the results given are approximations,
    not guaranteed to be exact solutions.


    Parameters
    ----------
    A : 2d-array, square, non-negative
        The square matrix :math:`A` in the objective function above.

    B : 2d-array, square, non-negative
        The square matrix :math:`B` in the objective function above.

    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for '2opt'.
        :ref:`'2opt' <optimize.qap-faq>` is also available.

    Options
    -------

    partial_match : 2d-array of integers, optional, (default = None)
        Allows the user to fix part of the matching between the two
        matrices. In the literature, a partial match is also known as a
        "seed".

        Each row of `partial_match` specifies the indices of a pair of
        corresponding nodes, that is, node ``partial_match[i, 0]` of `A` is
        matched to node ``partial_match[i, 1]`` of `B`. Accordingly,
        ``partial_match`` is an array of size ``(m , 2)``, where ``m`` is
        less than the number of nodes.

    maximize : bool (default = False)
        Setting `maximize` to ``True`` solves the Graph Matching Problem (GMP)
        rather than the Quadratic Assingnment Problem (QAP).

    rng : {None, int, `~np.random.RandomState`, `~np.random.Generator`}
        This parameter defines the object to use for drawing random
        variates.
        If `rng` is ``None`` the `~np.random.RandomState` singleton is
        used.
        If `rng` is an int, a new ``RandomState`` instance is used,
        seeded with `rng`.
        If `rng` is already a ``RandomState`` or ``Generator``
        instance, then that object is used.
        Default is None.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` containing the following
        fields.

        col_ind : 1-D array
            An array of column indices corresponding with the best
            permutation of the nodes of `B` found.
        score : float
            The corresponding value of the objective function.
        nit : int
            The number of iterations performed during optimization.

    Notes
    -----
    This is a greedy algorithm that works similarly to bubble sort: beginning
    with an initial permutation, it iteratively swaps pairs of indices to
    improve the objective function until no such improvements are possible.

    References
    ----------
    .. [3] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt
    """

    _check_unknown_options(unknown_options)
    rng = check_random_state(rng)
    A, B, partial_match = _common_input_validation(
            A, B, partial_match, maximize)

    N = len(A)

    if partial_match.size:
        # use seed for initial permutation, but randomly permute the rest
        r, c = partial_match.T
        fixed_rows = np.zeros(N, dtype=bool)
        fixed_cols = np.zeros(N, dtype=bool)
        fixed_rows[r] = True
        fixed_cols[c] = True

        perm = np.zeros(N, dtype=int)
        perm[fixed_rows] = c
        perm[~fixed_rows] = rng.permutation(np.arange(N)[~fixed_cols])
    else:
        perm = rng.permutation(np.arange(N))

    best_score = _calc_score(A, B, perm)

    better = operator.gt if maximize else operator.lt
    n_iter = 0
    done = False
    while not done:
        # equivalent to nested for loops i in range(N), j in range(i, N)
        for i, j in itertools.combinations_with_replacement(range(N), 2):
            n_iter += 1
            perm[i], perm[j] = perm[j], perm[i]
            score = _calc_score(A, B, perm)
            if better(score, best_score):
                best_score = score
                break
            # faster to swap back than to create a new list every time
            perm[i], perm[j] = perm[j], perm[i]
        else:  # no swaps made
            done = True

    res = {"col_ind": perm, "score": best_score, "nit": n_iter}

    return OptimizeResult(res)
