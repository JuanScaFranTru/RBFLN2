import numpy as np


class RBFLN(object):

    def __init__(self, xs, ts, M, N, niter=100,
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
        :type M: int
        :type N: int
        :type niter: int
        :type eta_linear_weights: float
        :type eta_non_linear_weights: float
        :type eta_variance: float
        :type eta_center_vectors: float
        :type variance: float
        """
        self.xs = np.array(xs)
        self.ts = np.array(ts)
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

        self.x_to_q = {xs[q]: q for q in range(len(xs))}

        # Initialize variables
        self._init_center_vectors()
        self._init_variances()
        self._init_weights()
        self._init_learning_rates()

        # Train the model using the training data
        pass  # TODO

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

    def _squared_norms(self, x):
        """Calculate Squared Norm value for every hidden neuron.

        :param x: input feature vector.
        :type x: vector of float

        :rtype: vector of float
        """
        M = self.M
        vs = self.vs

        return [np.linalg.norm(x - vs[m]) ** 2 for m in range(M)]

    def _ys(self, x):
        """Calculate the RBF output of every hidden neuron.

        :param x: input feature vector.
        :type x: vector of float

        :return: Output of the hidden layer.
        :rtype: vector of float
        """
        variances = self.variances

        if x in self.x_to_q:
            q = self.x_to_q[x]
            squared_norms = self.squared_norms[q]
        else:
            squared_norms = self._squared_norms(x)

        return np.exp(- squared_norms / (2 * variances))

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

        if x in self.x_to_q:
            q = self.x_to_q[x]
            ys = self.ys[q]
        else:
            ys = self._ys(x)

        linear_component = np.dot(x, ws)
        nonlinear_component = np.dot(ys, us)

        return (1 / (M + N)) * (linear_component + nonlinear_component)

    def _precalculate(self):
        """Precalculate:
            - Norms
            - Ys
            - Zs
        """
        xs = self.xs

        self.squared_norms = np.apply_along_axis(self._squared_norms, 1, xs)
        self.ys = np.apply_along_axis(self._ys, 1, xs)
        self.z = np.apply_along_axis(self._z, 1, xs)

    def predict(self, x):
        """Predict the output using the model in the given input vector.

        :param x: input feature vector.
        :type x: vector of float

        :return: output of the model.
        :rtype: float
        """
        return self._z(x)

    def _update_weights(self):
        """Update weights using gradient descent."""
        eta1 = self.eta_non_linear_weights
        eta2 = self.eta_linear_weights
        us = self.us
        ws = self.ws
        ts = self.ts
        ys = self.ys
        xs = self.xs
        z = self.z

        M = self.M
        N = self.N
        us = us + eta1/(M + N) * np.dot((ts - z), ys)
        ws = ws + eta2/(M + N) * np.dot((ts - z), xs)

    def _update_center_vectors(self):
        """Update center vectors using gradient descent."""
        pass

    def _update_variance(self):
        """Update variance value using gradient descent."""
        pass

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
            variance = 0.5 * (1/M) ** (1/N)

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
            self.eta_variance = 0.1

        if self.eta_center_vectors is None:
            self.eta_center_vectors = 0.1
