import numpy as np
from scipy.special import erfc, erfcinv

def to_vec_Nx3(N,val):
    """
    Broadcasting helper: Ensures the input 'val' is shaped as (N, 3).
    
    Useful for expanding a single position/normal vector to match a batch of N elements.
    
    Args:
        N (int): Target number of rows.
        val (array-like): Input vector(s). Can be (3,), (1, 3), or (N, 3).
        
    Returns:
        np.ndarray: Array of shape (N, 3).
    """
    arr = np.array(val)
    if arr.ndim == 1:
    # e.g., [0,0,1] -> (1,3) -> tile to (N,3)
        arr = arr.reshape(1, 3)
    if arr.shape[0] == 1 and N > 1:
        arr = np.tile(arr, (N, 1))
    return arr

    # --- Helper for Scalar Fields (N, 1) ---
def to_scal_Nx1(N,val, default_val=0):
    """
    Broadcasting helper: Ensures the input 'val' is shaped as (N, 1).
    
    Useful for expanding a scalar property (like Power or Area) to match a batch of N elements.
    
    Args:
        N (int): Target number of rows.
        val (scalar or array-like): Input value.
        default_val (float): Value to use if 'val' is None.
        
    Returns:
        np.ndarray: Array of shape (N, 1).
    """
    if val is None:
        val = default_val

    if is_scalar(val):
        return np.full((N, 1), val)

    arr = np.array(val)
    if arr.ndim == 1:
        return arr.reshape(N, 1)
    return arr

def normalize_bool_array(x, N):
    """
    Converts a boolean scalar or list into a boolean numpy array of size N.
    """
    if isinstance(x, (bool, np.bool_)):
        return np.full(N, x, dtype=bool)
    return np.asarray(x, dtype=bool)

def as_array_of_size(x, N):
    """
    Enforces that input 'x' becomes an array of size N.
    Raises ValueError if dimensions mismatch.
    """
    if np.isscalar(x):
        return np.full(N, x)
    arr = np.asarray(x)
    
    #if arr.size == 1:
     #   return np.full(N, arr.item())
    
    if arr.size != N:
        raise ValueError(f"Expected size {N}, got {arr.size}")
    return arr

def solar_panel_angular_efficiency(cos_inc: np.ndarray) -> np.ndarray:
    """
    Models the angular loss of a Solar Panel (deviations from Lambertian).
    
    Calculates efficiency degradation based on the incidence angle (theta) using 
    a 5th-order polynomial fit derived from experimental data.
    
    Args:
        cos_inc (np.ndarray): Cosine of the incidence angle.
        
    Returns:
        np.ndarray: Efficiency scaling factor (0.0 to ~1.0).
    """
    p_p = np.array([-1.81907071e-09,  3.00750020e-07, -1.82841164e-05,  4.57546496e-04,
                    -4.11754977e-03,  1.00666212e+00])
    p_s = np.poly1d(p_p)
    theta = np.rad2deg(np.arccos(np.clip(cos_inc, -1.0, 1.0)))
    efficiency = p_s(theta)
    efficiency[theta >= 90] = 0
    return np.maximum(0, efficiency)

def is_scalar(x):
    """
    Checks if a value is effectively a scalar (int, float, or 0-d array).
    """
    if x is None:
        return False
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)

def Qfunction(x):
    """
    Standard Gaussian Q-function.
    Q(x) = 0.5 * erfc(x / sqrt(2))
    """
    return 0.5 * erfc( x/np.sqrt(2) )

def Qinv(y):
    """
    Inverse Gaussian Q-function.
    """
    return np.sqrt(2) * erfcinv( 2 * y )
