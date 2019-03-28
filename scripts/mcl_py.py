import numpy as np


class MCLPy:
    """
    This is a crude reimplementation of the Markov Cluster Algorithm presented by Stijn van Dongen in his PhD thesis
    "GRAPH CLUSTERING by FLOW SIMULATION" at the University of Utrecht.
    It is supposed to illustrate the simplicity of the algorithm.

    The general idea of the algorithm is that graphs possess natural groups which can be found by simulating flow within
    the graph. Flow will naturally be strong within groups and weak between groups.
    In order to find such groups a loop of expanding and inflating the matrix is performed until it is (nearly-)
    idempotent.
    Expansion equals p Markov transitions and inflation equals raising each matrix element to the power of r followed by
    scaling.
    """

    def run(self, M, p, r, maxiter=200):
        """

        Runs the loop of the MCL consisting of expansion and inflation maxiter times.

        :param M:       numpy.ndarray Matrix (m)_ij consisting the edge weightings of the vertices i and j
                                      of the Markov graph.
        :param p:       int           Expansion parameter. p > 2 has been reported to yield too few clusters in
                                      practice.
        :param r:       float         Inflation parameter. r in (0,1) increases column-homogeneity of M, while
                                                           r in (1,inf) decreases column-homogeneity.
        :param maxiter: int           Maximum number of iterations to be performed. In the theoretical MCL the algorithm
                                      is performed until M does not change anymore, i.e. is idempotent.
        :return:
        """
        for _ in range(maxiter):
            M = self.expansion(M, p)
            M = self.inflation(M, r)
        return M

    def expansion(self, M, p):
        """
        Expands the matrix M, i.e. multiplies it with itsself p amount of times, which promotes flow between regions.
        This is equivalent to calculating the p-step transition matrix of M.

        :param M:       numpy.ndarray Matrix (m)_ij containing the edge weightings of the vertices i and j
                                      of the Markov graph.
        :param p:       int           Expansion parameter. p > 2 has been reported to yield too few clusters in
                                      practice.

        :return:        numpy.ndarray Matrix (m)_ij containing the edge weightings of the vertices i and j
                                      of the Markov graph.
        """
        for _ in range(p):
            M = np.dot(M, M)
        return M

    def inflation(self, M, r):
        """
        Inflates the matrix M according to the inflation parameter r. This promotes flow within groups and hinders flow
        between groups.

        :param M:       numpy.ndarray Matrix (m)_ij containing the edge weightings of the vertices i and j
                                      of the Markov graph.
        :param r:       float         Inflation parameter. r in (0,1) increases column-homogeneity of M, while
                                                           r in (1,inf) decreases column-homogeneity.
        :param maxiter: int           Maximum number of iterations to be performed.

        :return:        numpy.ndarray MMatrix (m)_ij containing the edge weightings of the vertices i and j
                                      of the Markov graph.
        """
        M = M**r

        for i in range(M.shape[0]):
            sum_ = np.sum(M[i])
            M[i] = np.divide(M[i], sum_)
        return M
