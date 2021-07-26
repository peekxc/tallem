import numpy as np
# from https://github.com/sylvchev/pymanopt/blob/checkgrad/pymanopt/tools/diagnostics.py

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def identify_linear_piece(x, y, window_length):
    """Identify a segment of the curve (x, y) that appears to be linear.
    This function attempts to identify a contiguous segment of the curve
    defined by the vectors x and y that appears to be linear. A line is fit
    through the data over all windows of length window_length and the best
    fit is retained. The output specifies the range of indices such that
    x(segment) is the portion over which (x, y) is the most linear and the
    output poly specifies a first order polynomial that best fits (x, y) over
    that segment (highest degree coefficients first).
    See also: check_directional_derivative check_gradient
    """
    residues = np.zeros(len(x)-window_length)
    polys = np.zeros(shape=(2, len(residues)))
    for i in range(len(residues)):
        segment = range(i, (i+window_length)+1)
        poly, residuals, _, _, _ = np.polyfit(x[segment], y[segment],
                                              1, full=True)
        residues[i] = np.linalg.norm(residuals)
        polys[:, i] = poly
    best = np.argmin(residues)
    segment = range(best, best+window_length+1)
    poly = polys[:, best]
    return segment, poly


def check_directional_derivative(problem, x=None, d=None):
    """Checks the consistency of the cost function and directional derivatives.
    check_directional_derivative performs a numerical test to check that the
    directional derivatives defined in the problem structure agree up to first
    order with the cost function at some point x, along some direction d. The
    test is based on a truncated Taylor series (see online pymanopt
    documentation).
    Both x and d are optional and will be sampled at random if omitted.
    See also: check_gradient
    """
    #  If x and / or d are not specified, pick them at random.
    if d is not None and x is None:
        raise ValueError("If d is provided, x must be too, "
                         "since d is tangent at x.")
    if x is None:
        x = problem.manifold.rand()
    if d is None:
        d = problem.manifold.randvec(x)

    # Compute the value f0 at f and directional derivative at x along d.
    f0 = problem.cost(x)
    grad = problem.grad(x)
    df0 = problem.manifold.inner(x, grad, d)

    # Compute the value of f at points on the geodesic (or approximation
    # of it) originating from x, along direction d, for stepsizes in a
    # large range given by h.
    h = np.logspace(-8, 0, 51)
    value = np.zeros_like(h)
    for i, h_k in enumerate(h):
        try:
            y = problem.manifold.exp(x, h_k * d)
        except NotImplementedError:
            y = problem.manifold.retr(x, h_k * d)
        value[i] = problem.cost(y)

    # Compute the linear approximation of the cost function using f0 and
    # df0 at the same points.
    model = np.polyval([df0, f0], h)

    # Compute the approximation error
    err = np.abs(model - value)
    model_is_exact = not np.all(err < 1e-12)
    if model_is_exact:
        print("Directional derivative check. "
              "It seems the linear model is exact: "
              "Model error is numerically zero for all h.")
        # The 1st order model is exact: all errors are (numerically) zero
        # Fit line from all points, use log scale only in h.
        segment = range(len(h))
        poly = np.polyfit(np.log10(h), err, 1)
        # Set mean error in log scale for plot.
        poly[-1] = np.log10(poly[-1])
    else:
        print("Directional derivative check. The slope of the "
              "continuous line should match that of the dashed "
              "(reference) line over at least a few orders of "
              "magnitude for h.")
        # In a numerically reasonable neighborhood, the error should
        # decrease as the square of the stepsize, i.e., in loglog scale,
        # the error should have a slope of 2.
        window_len = 10
        segment, poly = identify_linear_piece(np.log10(h), np.log10(err),
                                              window_len)
    return h, err, segment, poly


def check_gradient(problem, x=None, d=None):
    """Checks the consistency of the cost function and the gradient.
    check_gradient performs a numerical test to check that the gradient
    defined in the problem structure agrees up to first order with the cost
    function at some point x, along some direction d. The test is based on a
    truncated Taylor series.
    It is also tested that the gradient is indeed a tangent vector.
    Both x and d are optional and will be sampled at random if omitted.
    """
    #  If x and / or d are not specified, pick them at random.
    if plt is None:
        raise RuntimeError("The 'check_gradient' function requires matplotlib")
    if d is not None and x is None:
        raise ValueError("If d is provided, x must be too,"
                         "since d is tangent at x.")
    if x is None:
        x = problem.manifold.rand()
    if d is None:
        d = problem.manifold.randvec(x)

    h, err, segment, poly = check_directional_derivative(problem, x, d)

    # plot
    plt.figure()
    plt.loglog(h, err)
    plt.xlabel("h")
    plt.ylabel("Approximation error")
    plt.loglog(h[segment], 10**np.polyval(poly, np.log10(h[segment])),
               linewidth=3)
    plt.autoscale(False)
    plt.plot([1e-8, 1e0], [1e-8, 1e8], linestyle="--", color="k")

    plt.title("Gradient check\nThe slope of the continuous line "
              "should match that of the dashed\n(reference) line "
              "over at least a few orders of magnitude for h.")
    plt.show()

    # Try to check that the gradient is a tangent vector
    grad = problem.grad(x)
    if hasattr(problem.manifold, "tangent"):
        projected_grad = problem.manifold.tangent(x, grad)
    else:
        print("Unfortunately, pymanopt was unable to verify that the gradient "
              "is indeed a tangent vector. Please verify this manually or "
              "implement the 'tangent' function in your manifold structure.")
        projected_grad = problem.manifold.proj(x, grad)
    residual = grad - projected_grad
    err = problem.manifold.norm(x, residual)
    print("The residual should be 0, or very close. "
          "Residual: {:g}.".format(err))
    print("If it is far from 0, then the gradient "
          "is not in the tangent space.")