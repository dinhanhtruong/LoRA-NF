import os
import re
import sys
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import packaging
import torch
from scripts.common import read_image, write_image
import trimesh
import pysdf


def listdir_by_time(path, endswith, prepend_path=True):
    """List files in a directory sorted by modification time."""
    files = os.listdir(path)
    files = [f for f in files if f.endswith(endswith)]
    files_with_mtime = [(f, os.path.getmtime(os.path.join(path, f))) for f in files]
    files_with_mtime.sort(key=lambda x: x[1])  # Sort by modification time (oldest first)
    if prepend_path:
        return [f"{path}/{f[0]}" for f in files_with_mtime]
    return [f"{f[0]}" for f in files_with_mtime]

def sort_files_with_number_at_end(files):
    def extract_number(filename):
        match = re.search(r'(\d+)\D*$', filename)
        return int(match.group(1)) if match else float('-inf')
    return sorted(files, key=lambda x: (extract_number(x), x))

def edit_region_percentage_to_normalized_bounds(y_low, y_high, x_low, x_high):
    """
    y_low, y_high, x_low, x_high:  coordinates in percentage coord system of image (top left = 0%), i.e. in [0,100]
    Returns same thing but in [-1,1]^2 normalized coord system
    """
    normalized = [(coord/100 - 0.5)*2 for coord in [y_low, y_high, x_low, x_high]]
    # round to 2 dp
    return [round(x, 2) for x in normalized]

# borrowed from scikit-image
def check_shape_equality(*images):
    """Check that all images have the same shape"""
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')
    return


#################
## borrowed from scikit-image for psnr metric
dtype_range = {
    bool: (False, True),
    np.bool_: (False, True),
    float: (-1, 1),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}
new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}
def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)
def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = _supported_float_type((image0.dtype, image1.dtype))
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1
def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            print(
                "Inputs have mismatched dtype.  Setting data_range based on "
                "image_true."
            )
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range."
            )
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    data_range = float(data_range)  # prevent overflow for small integer types
    return 10 * np.log10((data_range**2) / err)
#################
#################


def measure_gpu_memory(device):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        max_allocated_memory = torch.cuda.max_memory_allocated(device)
        print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
        # print(f"Max allocated memory: {max_allocated_memory / 1024**2:.2f} MB")


def get_model_checkpoint_size_mb(checkpoint_path, return_param_count=False):
    '''
    checkpoint_path: full path containing file extension.
    '''
    checkpoint = torch.load(checkpoint_path, weights_only=True)["model_state_dict"]
    size_model = 0
    # for k,v in checkpoint.items():
    #     print(k, v.shape)
    param_count = 0
    for param in checkpoint.values():
        # if param.is_floating_point():
        size_model += param.nelement() * param.element_size()
        param_count += param.nelement()
        # else:
            # size_model += param.numel() * torch.iinfo(param.dtype).bits
    print(f"\tmodel size: {(size_model / 1e6):.3f} MB")
    print(f"\tnum params: {param_count}")
    if return_param_count:
        return size_model / 1e6, param_count
    return size_model / 1e6




def mean_relative_l2(pred, target, eps=0.01):
    loss = (pred - target.to(pred.dtype))**2 / (pred.detach()**2 + eps)
    return loss.mean()

def mape_loss(pred, target, reduction='mean'):
    # pred, target: [B, 1], torch tenspr
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    if reduction == 'mean':
        loss = loss.mean()
    
    return loss

### marching cubes helpers from torch-ngp ###
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def extract_fields(bound_min, bound_max, resolution, query_func, device=torch.device("cpu")):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).to(device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).to(device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).to(device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles



def save_mesh(save_path, sdf_model, device, resolution=256):
    print(f"==> Saving mesh to {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def query_func(pts):
        pts = pts.to(device)
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=False):
            sdfs = sdf_model(pts)
        return sdfs

    bounds_min = torch.FloatTensor([-1, -1, -1])
    bounds_max = torch.FloatTensor([1, 1, 1])

    vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)

    mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    mesh.export(save_path)

    print(f"==> Finished saving mesh.")