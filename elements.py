import numpy as np
import numpy as np
import copy
from dataclasses import field,dataclass,fields
from const import *
from spatial import *

@dataclass
class Elements:
    """
    Base data structure representing a batch of spatial elements in 3D space.
    
    This class handles the position and orientation vectors for N elements and 
    provides methods to merge multiple batches together.
    
    Attributes:
        r (np.ndarray): Position vectors of shape (N, 3).
        n (np.ndarray): Orientation/Normal vectors of shape (N, 3). Defaults to [0, 0, 1].
    """
    r: np.ndarray  # (N, 3) Position
    n: np.ndarray  = field(default_factory=lambda: np.array([0,0,1]))  # (N, 3) Orientation/Normal


    def __post_init__(self):
        """
        Validates inputs and ensures data is stored as (N, 3) numpy arrays.
        
        Calculates the number of elements (N) based on the position array 'r'.
        """
        # Cast to numpy arrays first
        self.r = np.array(self.r)
        self.n = np.array(self.n)

        if self.r.ndim == 1:
            self.r = self.r.reshape(1, 3)

        self.N = self.r.shape[0]
        self.n = to_vec_Nx3(self.N, self.n)
        norms = np.linalg.norm(self.n, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0):
            self.n = self.n / norms

    def __add__(self, other):
        """
        Allows using '+' to combine two batches of elements:
        batch_combined = batch1 + batch2
        
        Args:
            other (Elements): The other batch to append to this one.
            
        Returns:
            Elements: A new instance containing the vertically stacked data of both batches.
            
        Raises:
            TypeError: If 'other' is not the same class type.
            ValueError: If optional fields are present in one object but not the other.
        """
        if other is None:
            return copy.deepcopy(self)

        # Ensure we are adding compatible types (e.g., Rx + Rx, not Rx + Tx)
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

        # Create a copy to store the result
        new_obj = copy.deepcopy(self)

        # Automatically loop through all defined fields (r, n, A, + subclass fields)
        for field in fields(self):
            name = field.name

            val_self = getattr(self, name)
            val_other = getattr(other, name)

            # Handle optional fields that might be None (like refl)
            if val_self is None and val_other is None:
                continue
            if val_self is None or val_other is None:
                # If one is defined and the other isn't, we can't safely stack
                raise ValueError(f"Field '{name}' is None in one object but not the other.")

            # Stack the arrays vertically
            stacked_val = np.vstack([val_self, val_other])
            setattr(new_obj, name, stacked_val)

        # Manually update N (since it's not a dataclass field, the loop skips it)
        new_obj.N = new_obj.r.shape[0]

        return new_obj

    @classmethod
    def merge(cls, batch_list):
        """
        Merges a list of Element batches into one.
        
        This is more efficient than repeated addition for large lists. It also handles
        optional fields (like 'refl') by padding missing values with zeros if necessary.
        
        Args:
            batch_list (list): A list of Element objects (or subclasses) to merge.
            
        Returns:
            Elements: A single merged instance, or None if the list is empty.
        """
        if not batch_list:
            return None

        # Use the first item as a template
        merged = copy.deepcopy(batch_list[0])

        # Loop through fields and stack everything
        for field in fields(cls):
            name = field.name


            values = [getattr(b, name) for b in batch_list]

            # Handle cases where some batches might have None for optional fields
            if any(v is None for v in values):
                # If all are None, skip (result stays None)
                if all(v is None for v in values):
                    continue

                # Otherwise, fill None gaps with Zeros so we don't crash
                # We use 'b.N' to know how many zeros to generate for that batch
                valid_sample = next(v for v in values if v is not None)
                cols = valid_sample.shape[1] if valid_sample.ndim > 1 else 1

                values = [
                    v if v is not None else np.zeros((b.N, cols))
                    for v, b in zip(values, batch_list)
                ]

            
            setattr(merged, name, np.vstack(values))

        # Update the total count N
        merged.N = merged.r.shape[0]

        return merged

@dataclass
class OpticalTxElements(Elements):
    """
    Properties specific to Optical Transmitters (e.g., LEDs, Lasers).
    
    Attributes:
        p (np.ndarray): Optical power per element (Watts).
        m (np.ndarray): Lambertian order of emission (mode number).
    """
    """Properties specific to Transmitters."""
    p: np.ndarray = None # Default to None to trigger the safety net
    m: np.ndarray = None # Default to None to trigger the safety net

    def __post_init__(self):
        super().__post_init__() # Inherits self.N from Elements base class

        # 1. Fallback to SimulationDefaults if nothing was passed
        p_val = self.p if self.p is not None else SimulationDefaults.p
        m_val = self.m if self.m is not None else SimulationDefaults.m

        ## 2. Broadcast to (N, 1) to ensure compatibility with channel gain math
        self.p = to_scal_Nx1(self.N, self.p)
        self.m = to_scal_Nx1(self.N, self.m)

@dataclass
class OpticalRxElements(Elements):
    """
    Properties specific to Optical Receivers.
    
    Attributes:
        A (np.ndarray): Active detection area (m^2).
        fov (np.ndarray): Field of View (half-angle in radians).
        refl (np.ndarray): Reflection coefficient (optional).
        type_Rx (np.ndarray): Identifier for receiver type (e.g., 0 for photodiode / surface elements, 1 for photovoltaic panel).
    """
    """Properties specific to Receivers."""
    A: np.ndarray = None        # Active detection area (m^2)
    fov: np.ndarray = None      # Field of View (half-angle radians)
    refl: np.ndarray = None     # Reflection coefficient
    type_Rx: np.ndarray = None  # Identifier (0=PD, 1=PV)

    def __post_init__(self):
        super().__post_init__()

        a_val = self.A if self.A is not None else SimulationDefaults.rx_area
        f_val = self.fov if self.fov is not None else SimulationDefaults.fov
        t_val = self.type_Rx if self.type_Rx is not None else SimulationDefaults.rx_type
        r_val = self.refl if self.refl is not None else SimulationDefaults.reflectivity

        # 2. Broadcast to (N, 1) to match geometry matrices
        self.A = to_scal_Nx1(self.N, np.array(a_val))
        self.fov = to_scal_Nx1(self.N, np.array(f_val))
        self.type_Rx = to_scal_Nx1(self.N, np.array(t_val))
        self.refl = to_scal_Nx1(self.N, np.array(r_val))

@dataclass
class RFTxElements(Elements):
    """
    Properties specific to RF (Radio Frequency) Transmitters.
    
    Attributes:
        p (np.ndarray): Transmission power (dBm).
    """
    """Properties specific to Receivers."""
    p : np.ndarray = None #dBm


    def __post_init__(self):
        super().__post_init__()

        p_val = self.p if self.p is not None else SimulationDefaults.rf_driver['p_min']
