import numpy as np

def skewSym(x):
    return np.array([
                        [0., -x[2], x[1]],
                        [x[2], 0., -x[0]],
                        [-x[1], x[0], 0.]
                    ])

def spatialForceTransform(E, r):
    """
    Transform a spatial force vector from original frame (A) to a new frame (B).

    Args:
        E: Rotation matrix (3x3) representing the orientation of original frame to the new frame.
        r: Position vector (3,) representing the position of the original frame in the new frame.
        f: Spatial force vector (6,) in the original frame.

    Returns:
        Transformed spatial force vector (6,) in the new frame.
    """
    SpatialRotation = np.block([
                                [E, np.zeros_like(E)],
                                [np.zeros_like(E), E]
                                ])
    SpatialTranslation = np.block([
                                    [np.eye(3), -skewSym(r)],
                                    [np.zeros((3, 3)), np.eye(3)]
                                    ])
    XForce = SpatialRotation @ SpatialTranslation
    return XForce

def spatialForceTransformInv(E, r):
    """
    Transform a spatial force vector from a new frame (B) to a original frame (A).

    Args:
        E: Rotation matrix (3x3) representing the orientation of original frame to the new frame.
        r: Position vector (3,) representing the position of the original frame in the new frame.
        f: Spatial force vector (6,) in the original frame.

    Returns:
        Transformed spatial force vector (6,) in the new frame.
    """
    SpatialRotation = np.block([
                                [E.T, np.zeros_like(E)],
                                [np.zeros_like(E), E.T]
                                ])
    SpatialTranslation = np.block([
                                    [np.eye(3), skewSym(r)],
                                    [np.zeros((3, 3)), np.eye(3)]
                                    ])
    XForce = SpatialTranslation @ SpatialRotation
    return XForce

def wrench_transform_body(E, r, wrench):
    """
    Transform a wrench vector from original frame (A) to a new frame (B).

    Args:
        E: Rotation matrix (3x3) representing the orientation of original frame to the new frame.
        r: Position vector (3,) representing the position of the original frame in the new frame.
        wrench: Wrench vector (6,) in the original frame.

    Returns:
        Transformed wrench vector (6,) in the new frame.
    """
    return spatialForceTransform(E, r) @ wrench

def wrench_transform_base(E, r, wrench):
    """
    Transform a wrench vector from a new frame (B) to a original frame (A).

    Args:
        E: Rotation matrix (3x3) representing the orientation of original frame to the new frame.
        r: Position vector (3,) representing the position of the original frame in the new frame.
        wrench: Wrench vector (6,) in the original frame.

    Returns:
        Transformed wrench vector (6,) in the new frame.
    """
    return spatialForceTransformInv(E, r) @ wrench