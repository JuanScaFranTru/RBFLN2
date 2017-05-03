class RBFLN(object):

    def __init__(self):
        pass

    def _sum_sq_error(self, xs, ts):
        pass

    def total_sq_error(self, xs, ts):
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
        random in the feature space."""
        pass

    def _init_learning_rates(self):
        """Init learning rates."""
        pass
