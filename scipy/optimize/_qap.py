import numpy as np
import math
from . import linear_sum_assignment
from . import minimize_scalar


def quadratic_assignment(
    cost_matrix,
    dist_matrix,
    seed_cost=[],
    seed_dist=[],
    maximize=False,
    n_init=1,
    init_method="barycenter",
    max_iter=30,
    shuffle_input=True,
    eps=0.1,
):
    """Solve the quadratic assignment problem

        This class solves the Quadratic Assignment Problem and the Graph Matching Problem
        (QAP) through an implementation of the Fast Approximate QAP Algorithm (FAQ) (these
        two problems are the same up to a sign change) [1].

        This algorithm can be thought of as finding an alignment of the vertices of two
        graphs which minimizes the number of induced edge disagreements, or, in the case
        of weighted graphs, the sum of squared differences of edge weight disagreements.
        The option to add seeds (known vertex correspondence between some nodes) is also
        available [2].


        Parameters
        ----------
        cost_matrix : 2d-array, square, positive
            A square adjacency matrix

        dist_matrix : 2d-array, square, positive
            A square adjacency matrix

        seed_cost : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `cost_matrix`.

        seeds_dist : 1d-array, shape (m , 1) where m <= number of nodes (default = [])
            An array where each entry is an index of a node in `dist_matrix` The elements of
            `seed_cost` and `seed_dist` are vertices which are known to be matched, that is,
            `seed_cost[i]` is matched to vertex `seed_dist[i]`.

        maximize : bool (default = False)
            Gives users the option to solve the Graph Matching Problem (GMP) rather than QAP.
            This is accomplished through trivial negation of the objective function.

        n_init : int, positive (default = 1)
            Number of random initializations of the starting permutation matrix that
            the FAQ algorithm will undergo. n_init automatically set to 1 if
            init_method = 'barycenter'

        init_method : string (default = 'barycenter')
            The initial position chosen

            "barycenter" : the non-informative “flat doubly stochastic matrix,”
            :math:`J=1*1^T /n` , i.e the barycenter of the feasible region

            "rand" : some random point near :math:`J, (J+K)/2`, where K is some random doubly
            stochastic matrix

        max_iter : int, positive (default = 30)
            Integer specifying the max number of Franke-Wolfe iterations.
            FAQ typically converges with modest number of iterations.

        shuffle_input : bool (default = True)
            Gives users the option to shuffle the nodes of A matrix to avoid results
            from inputs that were already matched.

        eps : float (default = 0.1)
            A positive, threshold stopping criteria such that FW continues to iterate
            while Frobenius norm of :math:`(P_{i}-P_{i+1}) > eps`

        Returns
        -------

        row_ind, col_ind : array
            An array of row indices and one of corresponding column indices giving
            the optimal optimal permutation (with the fixed seeds given) on the nodes of B,
            to best minimize the objective function :math:`f(P) = trace(A^T PBP^T )`.


        References
        ----------
        .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik, S.G. Kratzer,
            E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and C.E. Priebe, “Fast
            approximate quadratic programming for graph matching,” PLOS one, vol. 10,
            no. 4, p. e0121002, 2015.

        .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski, C. Priebe,
            Seeded graph matching, Pattern Recognit. 87 (2019) 203–215



        """

    cost_matrix = np.asarray(cost_matrix)
    dist_matrix = np.asarray(dist_matrix)
    seed_cost = np.asarray(seed_cost)
    seed_dist = np.asarray(seed_dist)

    if cost_matrix.shape[0] != dist_matrix.shape[0]:
        msg = "Adjacency matrices must be of equal size"
        raise ValueError(msg)
    elif (
        cost_matrix.shape[0] != cost_matrix.shape[1]
        or dist_matrix.shape[0] != dist_matrix.shape[1]
    ):
        msg = "Adjacency matrix entries must be square"
        raise ValueError(msg)
    elif (cost_matrix >= 0).all() == False or (dist_matrix >= 0).all() == False:
        msg = "Adjacency matrix entries must be greater than or equal to zero"
        raise ValueError(msg)
    elif seed_cost.shape[0] != seed_dist.shape[0]:
        msg = "Seed arrays must be of equal size"
        raise ValueError(msg)
    elif seed_cost.shape[0] > cost_matrix.shape[0]:
        msg = "There cannot be more seeds than there are nodes"
        raise ValueError(msg)
    elif (seed_cost >= 0).all() == False or (seed_dist >= 0).all() == False:
        msg = "Seed array entries must be greater than or equal to zero"
        raise ValueError(msg)
    elif (seed_cost <= (cost_matrix.shape[0] - 1)).all() == False or (
        seed_dist <= (cost_matrix.shape[0] - 1)
    ).all() == False:
        msg = "Seed array entries must be less than or equal to n-1"
        raise ValueError(msg)
    elif type(n_init) is not int and n_init <= 0:
        msg = '"n_init" must be a positive integer'
        raise TypeError(msg)
    elif not init_method == "barycenter" and not init_method == "rand":
        msg = 'Invalid "init_method" parameter string'
        raise ValueError(msg)
    elif max_iter <= 0 and type(max_iter) is not int:
        msg = '"max_iter" must be a positive integer'
        raise TypeError(msg)
    elif type(shuffle_input) is not bool:
        msg = '"shuffle_input" must be a boolean'
        raise TypeError(msg)
    elif eps <= 0 and type(eps) is not float:
        msg = '"eps" must be a positive float'
        raise TypeError(msg)
    elif type(maximize) is not bool:
        msg = '"maximize" must be a boolean'
        raise TypeError(msg)

    n = cost_matrix.shape[0]  # number of vertices in graphs
    n_seeds = seed_cost.shape[0]  # number of seeds
    n_unseed = n - n_seeds

    score = math.inf
    perm_inds = np.zeros(n)

    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1
        score = 0

    seed_dist_c = np.setdiff1d(range(n), seed_dist)
    if shuffle_input:
        seed_dist_c = np.random.permutation(seed_dist_c)
        # shuffle_input to avoid results from inputs that were already matched

    seed_cost_c = np.setdiff1d(range(n), seed_cost)
    permutation_cost = np.concatenate([seed_cost, seed_cost_c], axis=None).astype(int)
    permutation_dist = np.concatenate([seed_dist, seed_dist_c], axis=None).astype(int)
    cost_matrix = cost_matrix[np.ix_(permutation_cost, permutation_cost)]
    dist_matrix = dist_matrix[np.ix_(permutation_dist, permutation_dist)]

    # definitions according to Seeded Graph Matching [2].
    A11 = cost_matrix[:n_seeds, :n_seeds]
    A12 = cost_matrix[:n_seeds, n_seeds:]
    A21 = cost_matrix[n_seeds:, :n_seeds]
    A22 = cost_matrix[n_seeds:, n_seeds:]
    B11 = dist_matrix[:n_seeds, :n_seeds]
    B12 = dist_matrix[:n_seeds, n_seeds:]
    B21 = dist_matrix[n_seeds:, :n_seeds]
    B22 = dist_matrix[n_seeds:, n_seeds:]
    A11T = np.transpose(A11)
    A12T = np.transpose(A12)
    A22T = np.transpose(A22)
    B21T = np.transpose(B21)
    B22T = np.transpose(B22)

    for i in range(n_init):
        # setting initialization matrix
        if init_method == "rand":
            K = np.random.rand(
                n_unseed, n_unseed
            )  # generate a nxn matrix where each entry is a random integer [0,1]
            for i in range(10):  # perform 10 iterations of Sinkhorn balancing
                K = _doubly_stochastic(K)
            J = np.ones((n_unseed, n_unseed)) / float(
                n_unseed
            )  # initialize J, a doubly stochastic barycenter
            P = (K + J) / 2
        elif init_method == "barycenter":
            P = np.ones((n_unseed, n_unseed)) / float(n_unseed)

        const_sum = A21 @ np.transpose(B21) + np.transpose(A12) @ B12
        grad_P = math.inf  # gradient of P
        n_iter = 0  # number of FW iterations

        # OPTIMIZATION WHILE LOOP BEGINS
        while grad_P > eps and n_iter < max_iter:
            delta_f = (
                const_sum + A22 @ P @ B22T + A22T @ P @ B22
            )  # computing the gradient of f(P) = -tr(APB^tP^t)
            rows, cols = linear_sum_assignment(
                obj_func_scalar * delta_f
            )  # run hungarian algorithm on gradient(f(P))
            Q = np.zeros((n_unseed, n_unseed))
            Q[rows, cols] = 1  # initialize search direction matrix Q

            def f(x):  # computing the original optimization function
                return obj_func_scalar * (
                    np.trace(A11T @ B11)
                    + np.trace(np.transpose(x * P + (1 - x) * Q) @ A21 @ B21T)
                    + np.trace(np.transpose(x * P + (1 - x) * Q) @ A12T @ B12)
                    + np.trace(
                        A22T
                        @ (x * P + (1 - x) * Q)
                        @ B22
                        @ np.transpose(x * P + (1 - x) * Q)
                    )
                )

            alpha = minimize_scalar(
                f, bounds=(0, 1), method="bounded"
            ).x  # computing the step size
            P_i1 = alpha * P + (1 - alpha) * Q  # Update P
            grad_P = np.linalg.norm(P - P_i1)
            P = P_i1
            n_iter += 1
        # end of FW optimization loop

        row, col = linear_sum_assignment(
            -P
        )  # Project onto the set of permutation matrices
        perm_inds_new = np.concatenate(
            (np.arange(n_seeds), np.array([x + n_seeds for x in col]))
        )

        score_new = np.trace(
            np.transpose(cost_matrix)
            @ dist_matrix[np.ix_(perm_inds_new, perm_inds_new)]
        )  # computing objective function value

        if obj_func_scalar * score_new < obj_func_scalar * score:  # minimizing
            score = score_new
            perm_inds = np.zeros(n, dtype=int)
            perm_inds[permutation_cost] = permutation_dist[perm_inds_new]

    return (np.arange(n), perm_inds)


def _unshuffle(array, n):
    unshuffle = np.array(range(n))
    unshuffle[array] = np.array(range(n))
    return unshuffle


def _doubly_stochastic(P):
    # Title: sinkhorn_knopp Source Code
    # Author: Tabanpour, B
    # Date: 2018
    # Code version:  0.2
    # Availability: https://pypi.org/project/sinkhorn_knopp/
    #
    # The MIT License
    #
    # Copyright (c) 2016 Baruch Tabanpour
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.
    max_iter = 1000
    epsilon = 1e-3
    epsilon = int(epsilon)
    stopping_condition = None
    iterations = 0
    D1 = np.ones(1)
    D2 = np.ones(1)
    P = np.asarray(P)

    assert np.all(P >= 0)
    assert P.ndim == 2
    assert P.shape[0] == P.shape[1]

    N = P.shape[0]
    max_thresh = 1 + epsilon
    min_thresh = 1 - epsilon

    r = np.ones((N, 1))
    pdotr = P.T.dot(r)
    c = 1 / pdotr
    pdotc = P.dot(c)
    r = 1 / pdotc
    del pdotr, pdotc

    P_eps = np.copy(P)
    while (
        np.any(np.sum(P_eps, axis=1) < min_thresh)
        or np.any(np.sum(P_eps, axis=1) > max_thresh)
        or np.any(np.sum(P_eps, axis=0) < min_thresh)
        or np.any(np.sum(P_eps, axis=0) > max_thresh)
    ):

        c = 1 / P.T.dot(r)
        r = 1 / P.dot(c)

        D1 = np.diag(np.squeeze(r))
        D2 = np.diag(np.squeeze(c))
        P_eps = D1.dot(P).dot(D2)

        iterations += 1

        if iterations >= max_iter:
            stopping_condition = "max_iter"
            break

    if not stopping_condition:
        stopping_condition = "epsilon"

    D1 = np.diag(np.squeeze(r))
    D2 = np.diag(np.squeeze(c))
    P_eps = D1.dot(P).dot(D2)

    return P_eps
