import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    """
    Q3.2
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be helpful
    p = np.zeros((6, 1), dtype=npDtype) # OR p = np.zeros((6,1))
    x1, y1, x2, y2 = rect

    # YOUR IMPLEMENTATION HERE
    h, w = It.shape
    _x = np.arange(w)
    _y = np.arange(h)
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    nX = int(x2 - x1)
    nY = int(y2 - y1)
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)
    X, Y = np.meshgrid(coordsX, coordsY)

    X_flat = X.ravel()
    Y_flat = Y.ravel()
    ones = np.ones_like(X_flat)

    T_flat = splineT.ev(X_flat, Y_flat).reshape(-1, 1)

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        M = np.array([[1.0 + p[0, 0], p[1, 0], p[2, 0]],
                      [p[3, 0],     1.0 + p[4, 0], p[5, 0]]],
                      dtype=npDtype)

        # Warp image
        #   1. warp coordinates
        warped = M @ np.vstack((X_flat, Y_flat, ones))
        xx_prime = warped[0, :]
        yy_prime = warped[1, :]

        valid = (xx_prime >= 0) & (xx_prime <= w-1) & \
                (yy_prime >= 0) & (yy_prime <= h-1)
        if not np.any(valid):
            break

        xx_prime_v = xx_prime[valid]
        yy_prime_v = yy_prime[valid]

        #   2. warp image (get image from new image locations)
        warpedI = splineI.ev(xx_prime_v, yy_prime_v).reshape(-1, 1)

        T_v = T_flat[valid, :]
        X_v = X_flat[valid]
        Y_v = Y_flat[valid]

        # Compute error image
        error = (T_v - warpedI)

        # Compute gradient of warped image
        Ix = splineI.ev(xx_prime_v, yy_prime_v, dx = 1, dy = 0)
        Iy = splineI.ev(xx_prime_v, yy_prime_v, dx = 0, dy = 1)

        # Compute Jacobian and Hessian
        A = np.vstack((Ix * X_v,
                       Ix * Y_v,
                       Ix,
                       Iy * X_v,
                       Iy * Y_v,
                       Iy)).T
        
        H = A.T @ A

        # Calculate deltaP
        deltaP = np.linalg.inv(H) @ (A.T @ error)

        # Update p
        p += deltaP

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break


    # Reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
