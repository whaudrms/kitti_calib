import numpy as np
from scipy.spatial.transform import Rotation as R

def random_rotation_matrix(max_deg=10):
    angles = np.deg2rad(np.random.uniform(-max_deg, max_deg, size=3))
    return R.from_euler('xyz', angles).as_matrix()
