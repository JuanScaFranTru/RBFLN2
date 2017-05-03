class RBFLN(object):

    def __init__(self, xs, ts, M, N, I=100, etas=None, variance=None):
        """Create a Radial Basis Functional Link Network.

        Create a RBFLN with N neurons in the input layer, M neurons in the
        hidden layer and 1 neuron in the output layer.
        The xs and ts parameters should have the same length.
        The lengths of all the elements of xs should be equal to N.
        The lengths of all the elements of ts should be equal to 1.

        :param xs: input feature vectors used for training.
        :param ts: associated output target vectors used for training.
        :param M: Number of neurons in the hidden layer.
        :param N: Number of neurons in the input layer.
        :param I: Number of iterations.
        :param etas: Learning rates.
        :param variance: The initial variance of the RBF.

        :type M: int
        :type N: int
        :type I: int
        :type etas: list of int
        :type variance: float
        """
        pass

    def _sum_sq_error(self, x, t):
        """Partial sum squared errors of the given training input feature
        vectors and associated output target vectors.

        :param x: input feature vector.
        :param t: associated output target vector.
        :type x: vector of float
        :type t: float
        """
        return (t - self.evaluate(x)) ** 2

    def total_sq_error(self, xs, ts):
        """Sum of the partial sum squared errors.

        :param xs: input feature vectors.
        :param ts: associated output target vectors.

        :type xs: list of vector of float
        :type ts: list of float
        """
        msg = 'Input and output vectors should have the same length'
        assert len(xs) == len(ts), msg

        error = 0
        Q = len(xs)
        for i in range(Q):
            t = ts[i]
            x = xs[i]
            error += self._sum_sq_error(x, t)

        return error

    def evaluate(self, x):
        """Evaluate the model in the given input vector.

        :param x: input feature vector.
        :type x: vector of float

        :return: output of the model.
        :rtype: float
        """
        pass

    def _update_weights(self):
        """Update weights using gradient descent."""
        pass

    def _update_center_vectors(self):
        """Update center vectors using gradient descent."""
        pass

    def _update_variance(self):
        """Update variance value using gradient descent."""
        pass

    def _init_weights(self):
        """Init linear and non-linear weights.

        Select all weights randomly between -0.5 and 0.5."""
        pass

    def _init_variance(self):
        """Compute initial values for variances."""
        # (1/2) * (1/M) ** 1/N
        pass

    def _init_center_vectors(self):
        """Init center vectors.

        Let Q be the number of input feature vectors.
        Initialize RBF center vectors by putting v(m) = x(m) if M <= Q, else
        put v(q) = x(q) , q = 1,...,Q, and draw the remaining M - Q centers at
        random in the feature space.
        """
        pass

    def _init_learning_rates(self):
        """Init learning rates."""
        pass
