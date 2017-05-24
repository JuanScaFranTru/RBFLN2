import numpy as np
from numpy.linalg import norm
from math import exp


class RBFLN(object):

    def __init__(self, xs, ts, xs_validation, ts_validation, M, N, niter=100,
                 eta_linear_weights=None,
                 eta_non_linear_weights=None,
                 eta_variance=None,
                 eta_center_vectors=None,
                 variance=None):
        """Create a Radial Basis Functional Link Network.

        Create a RBFLN with N neurons in the input layer, M neurons in the
        hidden layer and 1 neuron in the output layer.
        The xs and ts parameters should have the same length.
        The lengths of all the elements of xs should be equal to n.
        The lengths of all the elements of ts should be equal to 1.

        :param xs: input feature vectors used for training.
        :param ts: associated output target vectors used for training.
        :param xs_validation: input feature vectors used for validation.
        :param ts_validation: associated output target vectors used for
                              validation.
        :param M: Number of neurons in the hidden layer.
        :param N: Number of neurons in the input layer.
        :param niter: Number of iterations.
        :param eta_linear_weights: Learning rate of linear weights.
        :param eta_non_linear_weights: Learning rate of non linear weights.
        :param eta_variance: Learning rate of variance.
        :param eta_center_vectors: Learning rate of center vectors.
        :param variance: The initial variance of the RBF.

        :type xs: list of vector of float
        :type ts: list of float
        :type xs_validation: list of vector of float
        :type ts_validation: list of float
        :type M: int
        :type N: int
        :type niter: int
        :type eta_linear_weights: float
        :type eta_non_linear_weights: float
        :type eta_variance: float
        :type eta_center_vectors: float
        :type variance: float

        """
        self.xs = np.array([np.array(x) for x in xs])
        self.ts = np.array(ts)
        self.xs_validation = np.array([np.array(x) for x in xs_validation])
        self.ts_validation = np.array(ts_validation)
        self.M = M
        self.N = N
        self.niter = niter
        self.eta_linear_weights = eta_linear_weights
        self.eta_non_linear_weights = eta_non_linear_weights
        self.eta_variance = eta_variance
        self.eta_center_vectors = eta_center_vectors
        self.variance = variance

        msg = 'The xs and ts parameters should have the same length'
        assert len(xs) == len(ts), msg

        # Initialize variables
        self._init_center_vectors()
        self._init_variances()
        self._init_weights()
        self._init_learning_rates()

        # Train the model using the training data
        old_validation_error = float('inf')
        for i in range(niter):
            self._update_variables()
            error = self.total_sq_error(xs, ts)
            validation_error = self.total_sq_error(xs_validation,
                                                   ts_validation)
            print("  {:2.4f}   {:2.4f} ".format(error, validation_error),
                  end='\r')
            if validation_error > old_validation_error:
                break
            old_validation_error = validation_error
        print()

    def _sum_sq_error(self, x, t):
        """Partial sum squared errors of the given training input feature
        vectors and associated output target vectors.

        :param x: input feature vector.
        :param t: associated output target vector.
        :type x: vector of float
        :type t: float
        """
        return (t - self.predict(x)) ** 2

    def total_sq_error(self, xs, ts):
        """Sum of the partial sum squared errors.

        :param xs: input feature vectors.
        :param ts: associated output target vectors.

        :type xs: list of vector of float
        :type ts: list of float
        """
        msg = 'Input and output vectors should have the same length'
        assert len(xs) == len(ts), msg

        return sum([self._sum_sq_error(x, t) for x, t in zip(xs, ts)])

    def _ys(self, x):
        """Calculate the RBF output of every hidden neuron.

        :param x: input feature vector.
        :type x: vector of float

        :return: Output of the hidden layer.
        :rtype: vector of float
        """
        vs = self.vs
        variances = self.variances
        squared_norms = np.array([norm(x - v) ** 2 for v in vs])
        return np.array([exp(-sn / (2 * var))
                         for var, sn in zip(variances, squared_norms)])

    def _z(self, x):
        """Calculate the output of the RBFLN model.

        :param x: input feature vector.
        :type x: vector of float

        :return: output of the model.
        :rtype: float
        """
        M = self.M
        N = self.N
        us = self.us
        ws = self.ws

        ys = self._ys(x)  # TODO optimize using default argument ys=None

        linear_component = np.dot(x, ws)
        nonlinear_component = np.dot(ys, us)

        return (1 / (M + N)) * (linear_component + nonlinear_component)

    def predict(self, x):
        """Predict the output using the model in the given input vector.

        :param x: input feature vector.
        :type x: vector of float

        :return: output of the model.
        :rtype: float
        """
        return self._z(x)

    def _update_variables(self):
        """Update weights, center vectors and variances via gradient descent"""
        eta1 = self.eta_non_linear_weights
        eta2 = self.eta_linear_weights
        eta3 = self.eta_center_vectors
        eta4 = self.eta_variance
        vs = self.vs
        us = self.us
        ws = self.ws
        ts = self.ts
        xs = self.xs
        ys = np.apply_along_axis(self._ys, 1, xs)
        zs = np.apply_along_axis(self._z, 1, xs)
        variances = self.variances
        M = self.M
        N = self.N
        Q = len(xs)

        assert len(ys) == len(zs) == len(ts) == Q

        new_us = us + eta1/(M + N) * \
            np.sum([(t - z) * y for t, z, y in zip(ts, zs, ys)], axis=0)

        new_ws = ws + eta2/(M + N) * \
            np.sum([(t - z) * x for t, z, x in zip(ts, zs, xs)], axis=0)

        new_vs = np.array([None] * M)
        for m, v in enumerate(vs):
            new_vs[m] = v + eta3 / variances[m] * \
                np.sum([(t - z) * us[m] * y[m] * (x - v)
                        for x, y, z, t in zip(xs, ys, zs, ts)], axis=0)

        new_variances = np.array([None] * M)
        for m, variance in enumerate(variances):
            new_variances[m] = variance + eta4/(variance ** 2) * \
                np.sum([(t - z) * us[m] * y[m] * norm(x - vs[m]) ** 2
                        for x, y, z, t in zip(xs, ys, zs, ts)], axis=0)

        self.us = new_us
        self.ws = new_ws
        self.vs = new_vs
        self.variances = new_variances

    def _init_weights(self):
        """Init linear and non-linear weights.

        Select all weights randomly between -0.5 and 0.5."""
        MIN, MAX = -0.5, 0.5
        self.us = np.random.uniform(MIN, MAX, (self.M))
        self.ws = np.random.uniform(MIN, MAX, (self.N))

    def _init_variances(self):
        """Compute initial values for variances."""
        variance = self.variance
        N = self.N
        M = self.M

        if variance is None:
            variance = (1/(2*M)) ** (1/N)

        self.variances = np.array([variance] * M)

    def _init_center_vectors(self):
        """Init center vectors.

        Let Q be the number of input feature vectors.
        Initialize RBF center vectors by putting v(m) = x(m) if M <= Q, else
        put v(q) = x(q) , q = 1,...,Q, and draw the remaining M - Q centers at
        random in the feature space.
        """
        M = self.M
        N = self.N
        xs = self.xs
        Q = len(xs)

        self.vs = vs = xs[:M]
        if M > Q:
            vs = np.concatenate((vs, np.random.uniform(0, 1, (M - Q, N))))

    def _init_learning_rates(self):
        """Init learning rates."""
        if self.eta_linear_weights is None:
            self.eta_linear_weights = 2

        if self.eta_non_linear_weights is None:
            self.eta_non_linear_weights = 2

        if self.eta_variance is None:
            self.eta_variance = 0.01

        if self.eta_center_vectors is None:
            self.eta_center_vectors = 0.01
