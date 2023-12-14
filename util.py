import math
import sys

import numpy as np

from tqdm import tqdm

OneMinusEpsilon = 0.999999940395355225


def random_uniform_sphere(u,v):
    phi = u * 2 * np.pi
    v *= 2
    v -= 1
    cos_theta = v
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    return np.array([[np.cos(phi).item() * sin_theta], [np.sin(phi).item() * sin_theta], [cos_theta]])

def random_uniform_hemisphere(u, v):
    phi = u * 2 * np.pi
    cos_theta = v
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    return np.array([[np.cos(phi).item() * sin_theta], [np.sin(phi).item() * sin_theta], [cos_theta]])


# Sample from theta [0,pi/2] and phi[0,pi/2]
def random_uniform_reduced_hemisphere(u,v):
    phi = u * np.pi / 2
    cos_theta = v
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    return np.array([[np.cos(phi).item() * sin_theta], [np.sin(phi).item() * sin_theta], [cos_theta]])

# Sample a half vector with phi being a fixed value
def random_uniform_isotropic_halfvector(u):
    phi = 0.0
    cos_theta = u
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    return np.array([[np.cos(phi).item() * sin_theta], [np.sin(phi).item() * sin_theta], [cos_theta]])

# Map any phi_d to 0,pi/2
def reduce_phi_d(phi):
    if np.pi / 2 <= phi < np.pi:
        phi = np.pi - phi
    elif np.pi <= phi < 3 * np.pi / 2:
        phi = phi + np.pi
        phi = phi - np.pi * 2
    elif 3 * np.pi / 2 <= phi <= 2 * np.pi:
        phi = phi + np.pi
        phi = phi - np.pi * 2
        phi = np.pi - phi
    elif 0 <= phi < np.pi / 2:
        pass
    else:
        print("Wrong phi")
    return phi


def reduce_phi_d_reciprocity(phi):
    if np.pi <= phi <= np.pi * 2:
        phi = np.pi * 2 - phi
    elif 0 <= phi < np.pi:
        pass
    else:
        print("wrong phi")

    return phi

def extend_phi_d_reciprocity(phi):
    phi1 = np.pi * 2 - phi
    return [phi,phi1]


def extend_phi_d(phi):
    #phi should be within [0,pi/2]
    phi1 = np.pi - phi
    phi2 = phi + np.pi
    phi3 = 2 * np.pi - phi
    return [phi,phi1,phi2,phi3]

def clamp(a, b, val):
    if val < a:
        return a
    elif val > b:
        return b
    else:
        return val


# wi's direction is 'to' the surface
def reflect(wi, wm):
    wo = wi - 2 * dot(wi, wm) * wm
    return wo


def sample_cosine_hemisphere(N, u, v):
    phi = 2 * np.pi * u
    v = v * 2.0 - 1.0
    sphere = np.zeros((3, 1))

    sphere[0][0] = np.cos(phi) * np.sqrt(1.0 - v * v)
    sphere[1][0] = np.sin(phi) * np.sqrt(1.0 - v * v)
    sphere[2][0] = v

    direction = N + sphere
    direction = direction / np.linalg.norm(direction)

    return direction


def get_half_vector(wi, wo):
    wh = wi + wo
    wh = wh / np.linalg.norm(wh)

    # Make sure wh is on positive hemisphere
    if wh[2][0] < 0.0:
        wh = -wh

    return wh


def normalize(v : np.ndarray):
    return v / np.linalg.norm(v)


# x,y,z should be normalized
def makeVector(x,y,z):
    return np.array([[x],[y],[z]]).astype(np.float32)

# cross product, hope it will run faster than np.cross
# The vector is not normalized in this case
def cross_3D(A,B):
    x = A[1][0] * B[2][0] - A[2][0] * B[1][0]
    y = A[2][0] * B[0][0] - A[0][0] * B[2][0]
    z = A[0][0] * B[1][0] - A[1][0] * B[0][0]

    return makeVector(x, y, z)


def norm_3D(A):
    x2 = A[0][0] * A[0][0]
    y2 = A[1][0] * A[1][0]
    z2 = A[2][0] * A[2][0]

    return math.sqrt(x2 + y2 + z2)


# Return t * a + (1 - t) * b
def lerp(a, b, t):
    return t * a + (1 - t) * b


def to_spherical(w):
    # if w[2][0] > 0.9999:
    #     return 0.0,0.0
    # elif w[2][0] < -0.9999:
    #     return np.pi,0.0
    # else:
    theta = np.arccos(w[2][0])
    phi = np.arctan2(w[1][0], w[0][0])
    if phi < 0.0:
        phi += 2 * np.pi
    return theta, phi

def to_spherical_1D(w):
    # if w[2][0] > 0.9999:
    #     return 0.0,0.0
    # elif w[2][0] < -0.9999:
    #     return np.pi,0.0
    # else:
    theta = np.arccos(w[2])
    phi = np.arctan2(w[1], w[0])
    if phi < 0.0:
        phi += 2 * np.pi
    return theta, phi

def to_spherical_vectorized(w):
    theta = np.arccos(w[2,:])
    phi = np.arctan2(w[1,:],w[0,:])
    phi = np.where(phi < 0.0, phi + np.pi * 2, phi)
    return theta,phi

def to_spherical_stable(w):
    theta = 2 * np.arcsin(0.5 * np.sqrt(w[0][0] ** 2 + w[1][0] ** 2 + (w[2][0]-1) ** 2))
    phi = np.arctan2(w[1][0],w[0][0])
    if phi < 0.0:
        phi += 2 * np.pi
    return theta, phi



def to_cartesian(theta, phi):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    # if math.fabs(np.pi / 2 - theta) < 0.0001:
    #     return np.array([[cos_phi], [sin_phi], [0.0]])
    # elif theta < 0.0001:
    #     return np.array([[0.0], [0.0], [1.0]])
    # elif np.pi - theta < 0.0001:
    #     return np.array([[0.0],[0.0],[-1.0]])
    arr = np.array([[sin_theta * cos_phi], [sin_theta * sin_phi], [cos_theta]])

    if arr[2] == 1.0:
        return makeVector(0.0,0.0,1.0)
    else:
        return (arr / np.linalg.norm(arr)).astype(np.float32)


def to_cartesian_vectorized(theta,phi):
    theta = np.where(np.abs(np.pi / 2 - theta) < 0.0001, np.pi / 2, theta)
    theta = np.where(theta < 0.0001, 0.0, theta)
    theta = np.where( np.pi - theta < 0.0001, np.pi, theta)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    first_row = sin_theta * cos_phi
    second_row = sin_theta * sin_phi
    third_row = cos_theta

    default = np.stack([first_row,second_row,third_row])
    return default


def clamp_dot(v1, v2):
    val = dot(v1, v2)
    if val > 0:
        return val
    else:
        return 0


def heaviside(v1, v2):
    val = dot(v1, v2)
    if val > 0:
        return 1.0
    else:
        return 0.0

def heaviside_vectorized(v1,v2):
    val = dot_vectorized(v1,v2)
    tmp = np.where(val > 0, 1.0 , 0.0)
    return tmp

# All vectors are row vector
def dot(v1, v2):
    result = np.dot(v1.T, v2).item()

    #result = v1[0][0] * v2[0][0] + v1[1][0] * v1[1][0] + v2[1][0] * v2[1][0]

    if result == 0.0:
        result = 0.00001
    return result

def dot_vectorized(v1,v2):
    tmp = v1 * v2
    result = np.sum(tmp, axis=0)
    return result


# return the string of a [3,1] np array
def str_vector(w):
    return f'{w[0][0]:.3f},{w[1][0]:.3f},{w[2][0]:.3f}'


# Return the degree of the theta angle of a vector
def elevation_degree(w):
    return np.degrees(np.arccos(w[2][0]))


# Convert a coordinate to tensorflow tensor (in one row)
def to_tensor(w):
    return np.array([w[0][0], w[1][0], w[2][0]])


# convert a tensor to numpy array?
def to_arr(t):
    ret = t.numpy()
    return np.array([[ret[0]], [ret[1]], [ret[2]]])


# Spherical Gaussian
# vMF - 3d
# https://hal.science/hal-04004568/document
def random_VMF(mu, kappa, rng, size=None, ):
    """
    Von Mises-Fisher distribution sampler with
    mean direction mu and concentration kappa .
    Source : https://hal.science/hal-04004568
    """

    # parse input parameters
    n = 1 if size is None else np.product(size)
    shape = () if size is None else tuple(np.ravel(size))

    # the function supports row vector, we use column vector
    mu = np.squeeze(np.asarray(mu))

    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape
    # z component : radial samples perpendicular to mu
    # z = np.random.normal(0, 1, (n, d))
    z = rng.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # Note:
    # z @ mu[:,None] is the dot product of z and mu
    # So z @ mu[:,None] * mu[None,:] is nothing but z's projection on mu
    # z - z's projection results in a vector perpendicular to mu
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # sample angles ( in cos and sin form )
    cos = _random_VMF_cos(d, kappa, n, rng)
    sin = np.sqrt(1 - cos ** 2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None, :]
    tmp = x.reshape((*shape, d)).T
    return np.array([[tmp[0]], [tmp[1]], [tmp[2]]])


# Possible improvement: https://isas.iar.kit.edu/pdf/SDF15_Kurz-VMFSampling.pdf
def _random_VMF_cos(d: int, kappa: float, n: int, rng):
    """
    Generate n iid samples t with density function given by
    p ( t ) = someConstant * (1 - t ** 2 ) **(( d - 2 ) / 2 ) * exp ( kappa * t )
    """

    # b = Eq.4 of https://doi.org/10.1080/03610919408813161
    b = (d - 1) / (2 * kappa + (4 * kappa ** 2 + (d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        # z = np.random.beta((d - 1) / 2, (d - 1) / 2, size=m)
        z = rng.beta((d - 1) / 2, (d - 1) / 2, size=m)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        # accept = test >= - np.random.exponential(size=m)
        accept = test >= - rng.exponential(size=m)
        out.append(t[accept])
        found += len(out[- 1])
    return np.concatenate(out)[: n]


# An estimation of the vMF(dimension=3) pdf
# From https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
def vMF_eval(mean, w, kappa):
    if kappa == 0:
        return 0.25 / np.pi
    else:
        factor = kappa / (2 * np.pi * (1 - np.exp(-2 * kappa)))
        power = kappa * (np.matmul(mean.T, w).item() - 1)
        e = np.exp(power)
        return factor * e


# Binary search to find the interval
def find_interval(arr, size, idx):
    size = int(size)
    if idx < arr[0]:
        return 0
    elif idx > arr[size - 1]:
        return size - 2
    else:
        i = np.searchsorted(arr, idx)
        return i - 1


# Do a bi-linear interpolation
#      t1
# x1  ----  x2
# | t2       |
# |          |
# x3  ----  x4
def bilerp(x1, x2, x3, x4, t1, t2):
    tmp1 = lerp(x1, x2, t1)
    tmp2 = lerp(x3, x4, t1)
    ret = lerp(tmp1, tmp2, t2)
    return ret


def hprod(t):
    result = t[0]
    for i in range(t.size - 1):
        result *= t[i + 1]

    return result

def rgb2luminance(rgb):
    l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return l

def toRowVec(arr):
    return np.squeeze(arr)

def toColVec(arr):
    return np.array([[arr[0]],[arr[1]],[arr[2]]])


# u is a random number from 0 to 1
def u2theta(u):
    return np.pi / 2 * u * u


def u2phi(u):
    return np.pi * 2 * u

# if sin(theta) == 0, return epsilon
def safesin(theta):
    return np.max([sys.float_info.epsilon,np.sin(theta)])

def safecos(theta):
    return np.max([sys.float_info.epsilon,np.cos(theta)])


def getwiwo(wh,wd):
    if wh.ndim == 2:
        theta,phi = to_spherical(wh)
    else:
        print("Error ndim")
    x_axis = np.array([
        [np.sin(theta + np.pi / 2) * np.cos(phi)],
        [np.sin(theta + np.pi / 2) * np.sin(phi)],
        [np.cos(theta + np.pi / 2)]
    ])
    z_axis = wh

    y_axis = cross_3D(x_axis,z_axis)

    #y_axis = np.cross(x_axis.T, z_axis.T).T

    R_to_h = np.array([
        [x_axis[0].item(), y_axis[0].item(), z_axis[0].item()],
        [x_axis[1].item(), y_axis[1].item(), z_axis[1].item()],
        [x_axis[2].item(), y_axis[2].item(), z_axis[2].item()]
    ])

    R_to_z = np.linalg.inv(R_to_h)

    wi = np.matmul(R_to_h, wd)

    wo = reflect(-wi,wh)

    return wi, wo

def getwdwh(wi,wo):
    wh = get_half_vector(wi,wo)
    theta, phi = to_spherical_stable(wh)

    x_axis = np.array([
        [np.sin(theta + np.pi / 2) * np.cos(phi)],
        [np.sin(theta + np.pi / 2) * np.sin(phi)],
        [np.cos(theta + np.pi / 2)]
    ])
    z_axis = wh
    y_axis = cross_3D(x_axis, z_axis)

    #y_axis1 = np.cross(x_axis.T, z_axis.T).T

    R_to_h = np.array([
        [x_axis[0].item(), y_axis[0].item(), z_axis[0].item()],
        [x_axis[1].item(), y_axis[1].item(), z_axis[1].item()],
        [x_axis[2].item(), y_axis[2].item(), z_axis[2].item()]
    ])

    R_to_z = np.linalg.inv(R_to_h)

    wd = np.matmul(R_to_z, wi)

    return wd/norm_3D(wd) , wh




def grid_u2theta(u):
    return u**2 * np.pi / 2


def grid_theta2u(theta):
    return np.sqrt(theta * (2.0/np.pi))


def grid_u2phi(u):
    return u * np.pi * 2


def grid_phi2u(phi):
    return phi / np.pi / 2.0


def fresnel_term(wi, wm, eta_i, eta_t):
    if eta_t == 1.0 and eta_i == 1.0:
        return 1.0

    cos_theta_i = dot(wi, wm)
    if cos_theta_i > 1.0:
        cos_theta_i = 1.0
    if cos_theta_i < 0:
        eta_i, eta_t = eta_t, eta_i
    sin_theta_i = np.sqrt(1 - cos_theta_i * cos_theta_i)
    sin_theta_t = eta_i / eta_t * sin_theta_i
    if sin_theta_t >= 1.0:
        return 1.0
    cos_theta_t = np.sqrt(1 - sin_theta_t * sin_theta_t)

    r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t))
    r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t))
    return (r_perp * r_perp + r_parl * r_parl) / 2


def fresnel_vectorized(wi,wm,eta_i,eta_t):
    cos_theta_i = dot_vectorized(wi, wm)
    ret_val = np.ones_like(cos_theta_i)
    if eta_t == 1.0 and eta_i == 1.0:
        return ret_val


    # if cos_theta_i < 0:
    #     eta_i, eta_t = eta_t, eta_i

    # eta_t > eta_i ?
    sin_theta_i = np.sqrt(1 - cos_theta_i * cos_theta_i)
    sin_theta_t = eta_i / eta_t * sin_theta_i
    # if sin_theta_t >= 1.0:
    #     return 1.0

    cos_theta_t = np.sqrt(1 - sin_theta_t * sin_theta_t)
    r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t))
    r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t))
    tmp = (r_perp * r_perp + r_parl * r_parl) / 2

    return np.where(sin_theta_t >= 1.0, ret_val, tmp)

# from https://github.com/jdupuy/dj_brdf/blob/phd-2016/dj_brdf.h#L1292
# as mentioned in the Cook-Torrance paper
def unpolarized_eval(cos_theta_d, ior):
    c = cos_theta_d
    n = ior
    g = np.sqrt(n * n + c * c - 1.0)
    tmp1 = c * (g + c) - 1.0
    tmp2 = c * (g - c) + 1.0
    tmp3 = (tmp1 * tmp1) / (tmp2 * tmp2)
    tmp4 = ((g-c) * (g-c)) / ((g + c) * (g + c))
    return (0.5 * tmp4) * (1.0 + tmp3)

def unpolarized_eval_vectorized(wi, wm, ior):
    c = dot_vectorized(wi,wm)
    c = np.clip(c,0.0,1.0)
    n = ior
    g = np.sqrt(n * n + c * c - 1.0)
    tmp1 = c * (g + c) - 1.0
    tmp2 = c * (g - c) + 1.0
    tmp3 = (tmp1 * tmp1) / (tmp2 * tmp2)
    tmp4 = ((g-c) * (g-c)) / ((g + c) * (g + c))
    return (0.5 * tmp4) * (1.0 + tmp3)


def test_difference(a,b,difference):
    if math.fabs(a-b) / a > difference or math.fabs(a-b) / b > difference:
        return True
    else:
        return False


# avoid close-to-zero sample value
def clip_sample(u):
    return np.clip(u,1-OneMinusEpsilon,OneMinusEpsilon)
