import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    """
    Q3.1
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] p: movement vector dx, dy
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be useful
    # p := dx, dy
    p = np.zeros(2, dtype=npDtype)  # OR p = np.zeros(2)
    x1, y1, x2, y2 = rect

    # Crop template image
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    # This returns a class object; note the swap/transpose
    # Use spline.ev() for getting values at locations
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    nX, nY = int(x2 - x1), int(y2 - y1)
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)

    # YOUR IMPLEMENTATION STARTS HERE
    xx, yy =np.meshgrid(coordsX, coordsY)
    template = splineT.ev(xx, yy)
    template_flat = template.flatten()

    thresold = threshold

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        #   1. warp coordinates
        xx_prime = xx + p[0]
        yy_prime = yy + p[1]

        #   2. warp image (get image from new image locations)
        warpedI = splineI.ev(xx_prime, yy_prime)

        # Compute error image
        error = template_flat - warpedI.flatten()

        # Compute gradient of warped image
        Ix = splineI.ev(xx_prime, yy_prime, dx = 1, dy = 0).flatten()
        Iy = splineI.ev(xx_prime, yy_prime, dx = 0, dy = 1).flatten()

        # Compute Hessian; It is a special case
        A = np.vstack((Ix, Iy)).T
        H = A.T @ A

        # Calculate deltaP
        deltaP = np.linalg.inv(H) @ (A.T @ error)

        # Update p
        p += deltaP

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < thresold:
            break

    return p
