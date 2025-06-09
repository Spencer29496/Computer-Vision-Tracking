import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    Q3.3
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
    """

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    # p = np.zeros((6,1))
    npDtype = np.float64
    W = np.eye(3, dtype=npDtype)    # This might be a better format than p
    x1, y1, x2, y2 = rect

    # YOUR IMPLEMENTATION HERE
    h, w = It.shape
    _y = np.arange(h)
    _x = np.arange(w)

    splineT = RectBivariateSpline(_y, _x, It)
    splineI = RectBivariateSpline(_y, _x, It1)

    X = np.arange(x1, x2, 1, dtype=npDtype)
    Y = np.arange(y1, y2, 1, dtype=npDtype)
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_flat = X_grid.ravel()
    Y_flat = Y_grid.ravel()
    ones = np.ones_like(X_flat)

    # Compute gradient of template image
    T       = splineT.ev(Y_flat, X_flat).reshape(-1, 1)
    Ix     = splineT.ev(Y_flat, X_flat, dx = 0, dy = 1)
    Iy     = splineT.ev(Y_flat, X_flat, dx = 1, dy = 0)   

    # Compute Jacobian
    A = np.vstack((Ix * X_flat,
                   Ix * Y_flat,
                   Ix,
                   Iy * X_flat,
                   Iy * Y_flat,
                   Iy)).T

    # Compute Hessian
    H = np.linalg.inv(A.T @ A + 1e-8 * np.eye(6))

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        warped = W @ np.vstack((X_flat, Y_flat, ones))
        X_w = warped[0, :]
        Y_w = warped[1, :]

        valid = (X_w >= 0) & (X_w <= w-1) & (Y_w >= 0) & (Y_w <= h-1)
        if not np.any(valid):
            break

        I_warp = splineI.ev(Y_w[valid], X_w[valid]).reshape(-1, 1)

        # Compute error image
        error = I_warp - T[valid, :]

        # Compute deltaP
        A_v = A[valid, :]
        H_inv = np.linalg.inv(A_v.T @ A_v + 1e-8 * np.eye(6))
        deltaP = H_inv @ (A_v.T @ error)

        # Compute new W
        dp = deltaP.flatten()
        W_delta = np.array([[1 + dp[0], dp[1], dp[2]],
                            [dp[3], 1 + dp[4], dp[5]],
                            [0, 0, 1]],
                            dtype=npDtype)
        
        W = W @ np.linalg.inv(W_delta)

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break


    # reshape the output affine matrix
    # M = np.array([[1.0+p[0], p[1],    p[2]],
    #              [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
    M = W[:2, :]

    return M
