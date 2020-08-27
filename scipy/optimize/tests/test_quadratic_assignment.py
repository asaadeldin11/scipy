import pytest
import numpy as np
from scipy.optimize import quadratic_assignment
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_


################
# Common Tests #
################

def chr12c():
    A = [
        [0, 90, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [90, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
        [0, 0, 43, 0, 0, 0, 26, 0, 0, 0, 0, 0],
        [0, 0, 0, 88, 0, 0, 0, 16, 0, 0, 0, 0],
        [0, 0, 0, 0, 26, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 16, 0, 0, 0, 96, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 29, 0],
        [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
        [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0],
    ]
    B = [
        [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
        [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
        [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
        [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
        [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
        [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
        [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
        [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
        [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
        [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
        [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
        [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0],
    ]
    A, B = np.array(A), np.array(B)
    n = A.shape[0]

    opt_perm = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n

    return A, B, opt_perm


class QAPCommonTests(object):
    """
    Base class for `quadratic_assignment` tests.
    """
    def setup_method(self):
        np.random.rand(0)

    # Test global optima of problem from Umeyama IVB
    # https://pcl.sitehost.iu.edu/rgoldsto/papers/weighted%20graph%20match2.pdf
    # Graph matching maximum is in the paper
    # QAP minimum determined by brute force
    def test_accuracy_1(self):
        # besides testing accuracy, check that A and B can be lists
        A = [[0, 3, 4, 2],
             [0, 0, 1, 2],
             [1, 0, 0, 1],
             [0, 0, 1, 0]]

        B = [[0, 4, 2, 4],
             [0, 0, 1, 0],
             [0, 2, 0, 2],
             [0, 1, 2, 0]]

        res = quadratic_assignment(A, B, method=self.method,
                                   options={"init_weight": 0, "rng": 0,
                                            "maximize": False})
        assert_equal(res.score, 10)
        assert_equal(res.col_ind, np.array([1, 2, 3, 0]))

        res = quadratic_assignment(A, B, method=self.method,
                                   options={"init_weight": 0, "rng": 0,
                                            "maximize": True})
        assert_equal(res.score, 40)
        assert_equal(res.col_ind, np.array([0, 3, 1, 2]))

    # Test global optima of problem from Umeyama IIIB
    # https://pcl.sitehost.iu.edu/rgoldsto/papers/weighted%20graph%20match2.pdf
    # Graph matching maximum is in the paper
    # QAP minimum determined by brute force
    def test_accuracy_2(self):

        A = np.array([[0, 5, 8, 6],
                      [5, 0, 5, 1],
                      [8, 5, 0, 2],
                      [6, 1, 2, 0]])

        B = np.array([[0, 1, 8, 4],
                      [1, 0, 5, 2],
                      [8, 5, 0, 5],
                      [4, 2, 5, 0]])

        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": False})
        if self.method == 'faq':
            # Global optimum is 176, but FAQ gets 178
            assert_equal(res.score, 178)
            assert_equal(res.col_ind, np.array([1, 0, 3, 2]))
        else:
            assert_equal(res.score, 176)
            assert_equal(res.col_ind, np.array([1, 2, 3, 0]))

        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, "maximize": True})
        assert_equal(res.score, 286)
        assert_equal(res.col_ind, np.array([2, 3, 0, 1]))

    def test_accuracy_3(self):

        A, B, opt_perm = chr12c()

        # basic minimization
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0})
        assert_(11156 <= res.score < 21000)
        assert_equal(res.score, _score(A, B, res.col_ind))

        # basic maximization
        res = quadratic_assignment(A, B, method=self.method,
                                   options={"rng": 0, 'maximize': True})
        assert_(74000 <= res.score < 85000)
        assert_equal(res.score, _score(A, B, res.col_ind))

        # check ofv with partial match
        seed_cost = np.array([4, 8, 10])
        seed = np.asarray([seed_cost, opt_perm[seed_cost]]).T
        res = quadratic_assignment(A, B, method=self.method,
                                   options={'partial_match': seed})
        assert_(11156 <= res.score < 21000)

        # check performance when partial match is the global optimum
        seed = np.asarray([np.arange(len(A)), opt_perm]).T
        res = quadratic_assignment(A, B, method=self.method,
                                   options={'partial_match': seed})
        assert_equal(res.score, 11156)


class TestFAQ(QAPCommonTests):
    method = "faq"

    def test_options(self):
        # cost and distance matrices of QAPLIB instance chr12c
        A, B, opt_perm = chr12c()
        n = len(A)

        # check that max_iter is obeying with low input value
        res = quadratic_assignment(A, B,
                                   options={'maxiter': 5})
        assert_equal(res.nit, 5)

        # test with no shuffle
        res = quadratic_assignment(A, B,
                                   options={'shuffle_input': False})
        assert_(11156 <= res.score < 21000)

        # check with specified init_J
        K = np.ones((n, n)) / float(n)
        K = _doubly_stochastic(K)
        res = quadratic_assignment(A, B,
                                   options={'init_J': K})
        assert_(11156 <= res.score < 21000)

    def test_specific_input_validation(self):

        A = np.identity(2)
        B = A

        # method is implicitly faq

        # test that cost/dist matrices must be nonnegative
        with pytest.raises(
                ValueError, match="`A` and `B` matrices must contain only"):
            quadratic_assignment(
                -np.random.random((3, 3)),
                np.random.random((3, 3))
            )
        # ValueError Checks: making sure single value parameters are of
        # correct value
        with pytest.raises(ValueError, match="Invalid 'init_J' parameter"):
            quadratic_assignment(A, B, options={'init_J': "random"})
        with pytest.raises(
                ValueError,
                match="'init_weight' must be strictly between zero and one"):
            quadratic_assignment(A, B, options={'init_weight': 2})
        with pytest.raises(
                ValueError, match="'init_k' must be a positive integer"):
            quadratic_assignment(A, B, options={'init_k': -1})
        with pytest.raises(
                ValueError, match="'maxiter' must be a positive integer"):
            quadratic_assignment(A, B, options={'maxiter': -1})
        with pytest.raises(ValueError, match="'eps' must be a positive float"):
            quadratic_assignment(A, B, options={'eps': -1})

        # TypeError Checks: making sure single value parameters are of
        # correct type
        with pytest.raises(TypeError):
            quadratic_assignment(A, B, options={'init_k': 1.5})
        with pytest.raises(TypeError):
            quadratic_assignment(A, B, options={'maxiter': 1.5})

        # test init matrix input
        with pytest.raises(
                ValueError,
                match="`init_J` matrix must have same shape as A and B"):
            quadratic_assignment(
                np.identity(4), np.identity(4),
                options={'init_J': np.ones((3, 3))}
            )

        K = np.random.rand(3, 3)
        K = _doubly_stochastic(K, eps=1)
        # matrix that isn't quite doubly stochastic
        with pytest.raises(
                ValueError, match="`init_J` matrix must be doubly stochastic"):
            quadratic_assignment(
                np.identity(3), np.identity(3), options={'init_J': K}
            )


class Test2opt(QAPCommonTests):
    method = "2opt"


class TestQAPOnce():
    def setup_method(self):
        np.random.rand(0)

    # these don't need to be repeated for each method
    def test_input_validation(self):
        # test that non square matrices return error
        with pytest.raises(ValueError, match="`A` must be square"):
            quadratic_assignment(
                np.random.random((3, 4)),
                np.random.random((3, 3)),
            )
        with pytest.raises(ValueError, match="`B` must be square"):
            quadratic_assignment(
                np.random.random((3, 3)),
                np.random.random((3, 4)),
            )
        # test that cost and dist matrices have no more than two dimensions
        with pytest.raises(
                ValueError, match="`A` and `B` must have exactly two"):
            quadratic_assignment(
                np.random.random((3, 3, 3)),
                np.random.random((3, 3, 3)),
            )
        # test that cost and dist matrices of different sizes return error
        with pytest.raises(
                ValueError,
                match="`A` and `B` matrices must be of equal size"):
            quadratic_assignment(
                np.random.random((3, 3)),
                np.random.random((4, 4)),
            )
        # can't have more seed nodes than cost/dist nodes
        _rm = _range_matrix
        with pytest.raises(
                ValueError,
                match="There cannot be more seeds than there are nodes"):
            quadratic_assignment(np.identity(3), np.identity(3),
                                 options={'partial_match': _rm(5, 2)})
        # test for only two seed columns
        with pytest.raises(
                ValueError, match="`partial_match` must have two columns"):
            quadratic_assignment(
                np.identity(3), np.identity(3),
                options={'partial_match': _range_matrix(2, 3)}
            )
        # test that seed has no more than two dimensions
        with pytest.raises(
                ValueError, match="`partial_match` must have exactly two"):
            quadratic_assignment(
                np.identity(3), np.identity(3),
                options={'partial_match': np.random.rand(3, 2, 2)}
            )
        # seeds cannot be negative valued
        with pytest.raises(
                ValueError, match="`partial_match` contains negative entries"):
            quadratic_assignment(
                np.identity(3), np.identity(3),
                options={'partial_match': -1 * _range_matrix(2, 2)}
            )
        # seeds can't have values greater than number of nodes
        with pytest.raises(
                ValueError,
                match="`partial_match` entries must be less than number"):
            quadratic_assignment(
                np.identity(5), np.identity(5),
                options={'partial_match': 2 * _range_matrix(4, 2)}
            )
        # columns of seed matrix must be unique
        with pytest.raises(
                ValueError,
                match="`partial_match` column entries must be unique"):
            quadratic_assignment(
                np.identity(3), np.identity(3),
                options={'partial_match': np.ones((2, 2))}
            )


@pytest.mark.slow
def test_rand_qap():
    A, B, opt_perm = chr12c()

    # check ofv with 100 random initializations
    res = quadratic_assignment(
        A, B, options={'init_weight': 0.5, 'init_k': 100}
    )

    assert_(11156 <= res.score < 14500)


def _range_matrix(a, b):
    mat = np.zeros((a, b))
    for i in range(b):
        mat[:, i] = np.arange(a)
    return mat


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
