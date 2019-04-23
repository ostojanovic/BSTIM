import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point, MultiPoint


R = 6365.902 #km; Radius of the earth at a location within Germany.

def jacobian_sq(latitude):
    """
        jacobian_sq(latitude)
        
    Computes the "square root" (Cholesky factor) of the Jacobian of the cartesian projection from polar coordinates (in degrees longitude, latitude) onto cartesian coordinates (in km east/west, north/south) at a given latitude (the projection's Jacobian is invariante wrt. longitude).
    """
    return R*(np.pi/180.0)*np.array([[np.abs(np.cos(np.deg2rad(latitude))), 0.0],[0.0, 1.0]])

def compute_interaction(tp_1, tp_2, σs):
    """
        compute_interaction(tp_1, tp_2, σs)
    
    For interaction effects between two counties, Gaussian basis functions with radii σs are applied to the distances between all pairs of testpoints from both counties and averaged.
    
    Arguments:
    ==========
        tp_1:       representative testpoints from within county 1
        tp_2:       representative testpoints from within county 2
        σs:         standard deviations of the Gaussian basis-functions
    """
    centroid = (tp_1.mean(axis=0) + tp_2.mean(axis=0))/2
    Σ_sq = jacobian_sq(centroid[1])
    dx = cdist(tp_1, tp_2, "mahalanobis", VI=Σ_sq**2)

    interaction = np.zeros_like(σs)
    for k,σ in enumerate(σs):
        interaction[k] = np.mean(np.exp(-0.5*(dx/σ)**2)/np.sqrt(2*np.pi*σ**2))
    return interaction
    
    
    
def compute_loss(shape, testpoints, σs, ϵ = np.random.randn(100,2)):
    """
        compute_loss(shape, testpoints, σs, ϵ)
        
    For interaction effects between two counties, Gaussian basis functions are applied to the distance between testpoints from both counties. Cases occuring outside the `shape` of the region for which the reported cases are known should similarly contribute to interaction effects, but are missing in the dataset. To compensate for this bias, the expected fraction of 'mass' that a basis function centered around a testpoint in the county would assign to points outside the region of interest is estimated through rejection sampling.
    
    Arguments:
    ==========
        
        shape:      the shape of the region of interest surrounding the county
        testpoints: representative testpoints from within the county
        σs:         standard deviations of the Gaussian basis-functions
        ϵ:          noise term used for sampling points (num_partition_samples × 2)
    """
    borderlosses = np.zeros(len(σs))
    for tp in testpoints:
        Σ_sq = jacobian_sq(tp[1])
        Σ_sqinv = np.diag(1/np.diag(Σ_sq))

        # draw points surrounding testpoint in a Gaussian with radius 1km
        other_offsets = ϵ.dot(Σ_sqinv.T)

        # check border effects for each radius
        for j,σ in enumerate(σs):
            # draw samples around each testpoint in the county and check if they land in the region
            smp = MultiPoint(tp+other_offsets*σ)
            pts = shape.intersection(smp)
            borderlosses[j] += (1 if isinstance(pts, Point) else len(pts))
    return np.clip(borderlosses/(len(testpoints)*len(ϵ)), 0.0, 1.0)
