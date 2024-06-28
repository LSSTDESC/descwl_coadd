"""
Much of this code was copied or inspired by the interpolation code in
https://github.com/beckermr/pizza-cutter/
"""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

import numba
from numba import njit
from .defaults import FLAGS2INTERP


@njit
def _get_nearby_good_pixels(image, bad_msk, nbad, buff=4):
    """
    get the set of good pixels surrounding bad pixels.

    Parameters
    ----------
    image: array
        The image data
    bad_msk: bool array
        2d array of mask bits.  True means it is a bad
        pixel

    Returns
    -------
    bad_pix:
        bad pix is the set of bad pixels, shape [nbad, 2]
    good_pix:
        good pix is the set of bood pixels around the bad
        pixels, shape [ngood, 2]
    good_im:
        the set of good image values, shape [ngood]
    good_ind:
        the 1d indices of the good pixels row*ncol + col
    """

    nrows, ncols = bad_msk.shape

    ngood = nbad*(2*buff+1)**2
    good_pix = np.zeros((ngood, 2), dtype=numba.int64)
    good_ind = np.zeros(ngood, dtype=numba.int64)
    bad_pix = np.zeros((ngood, 2), dtype=numba.int64)
    good_im = np.zeros(ngood, dtype=image.dtype)

    ibad = 0
    igood = 0
    for row in range(nrows):
        for col in range(ncols):
            val = bad_msk[row, col]
            if val:
                bad_pix[ibad] = (row, col)
                ibad += 1

                row_start = row - buff
                row_end = row + buff
                col_start = col - buff
                col_end = col + buff

                if row_start < 0:
                    row_start = 0
                if row_end > (nrows-1):
                    row_end = nrows-1
                if col_start < 0:
                    col_start = 0
                if col_end > (ncols-1):
                    col_end = ncols-1

                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:

                            if igood == ngood:
                                raise RuntimeError('good_pix too small')

                            # got a good one, add it to the list
                            good_pix[igood] = (rc, cc)
                            good_im[igood] = image[rc, cc]

                            # keep track of index
                            ind = rc*ncols + cc
                            good_ind[igood] = ind
                            igood += 1

    bad_pix = bad_pix[:ibad, :]

    good_pix = good_pix[:igood, :]
    good_ind = good_ind[:igood]
    good_im = good_im[:igood]

    return bad_pix, good_pix, good_im, good_ind


def interp_image_nocheck(image, bad_msk):
    """
    interpolate the bad pixels in an image with no checking on the fraction of
    masked pixels

    Parameters
    ----------
    image: array
        the pixel data
    bad_msk: array
        boolean array, True means it is a bad pixel

    Returns
    -------
    interp_image: ndarray
        The interpolated image
    """

    nbad = bad_msk.sum()

    bad_pix, good_pix, good_im, good_ind = \
        _get_nearby_good_pixels(image, bad_msk, nbad)

    # extract unique ones
    gi, ind = np.unique(good_ind, return_index=True)

    good_pix = good_pix[ind, :]
    good_im = good_im[ind]

    img_interp = CloughTocher2DInterpolator(
        good_pix,
        good_im,
        fill_value=0.0,
    )
    interp_image = image.copy()
    interp_image[bad_msk] = img_interp(bad_pix)

    return interp_image


class CTInterpolator(object):
    """
    A class wrapping the interp_image_nocheck function to do
    an inplace interp and sets the INTRP bit

    This conforms to the interface required for the interpolator
    sent to descwl_coadd.coadd.warp_exposures

    This is a "functor" meaning the object can be called
        interpolator(exposure)
    """
    def __init__(self, bad_mask_planes=FLAGS2INTERP):
        self.bad_mask_planes = bad_mask_planes

    def __call__(self, exp):
        """
        Interpolate the exposure in place

        Parameters
        ----------
        exp: ExposureF
            The exposure object to interpolate.  The INTRP flag will be
            set for any pixels that are interpolated
        """
        bad_msk, _ = get_bad_mask(
            exp=exp, bad_mask_planes=self.bad_mask_planes,
        )
        iimage = interp_image_nocheck(exp.image.array, bad_msk)

        exp.image.array[:, :] = iimage

        interp_flag = exp.mask.getPlaneBitMask('INTRP')
        exp.mask.array[bad_msk] |= interp_flag

        assert not np.any(np.isnan(exp.image.array[bad_msk]))


def get_bad_mask(exp, bad_mask_planes=FLAGS2INTERP):
    """
    get the bad mask and masked fraction

    Parameters
    ----------
    exp: lsst.afw.ExposureF
        The exposure data
    bad_mask_planes: list, optional
        List of mask planes to consider bad

    Returns
    -------
    bad_msk: ndarray
        A bool array with True if weight <= 0 or defaults.FLAGS2INTERP
        is set
    maskfrac: float
        The masked fraction
    """
    var = exp.variance.array
    weight = 1/var

    bmask = exp.mask.array

    flags2interp = exp.mask.getPlaneBitMask(bad_mask_planes)
    bad_msk = (weight <= 0) | ((bmask & flags2interp) != 0)

    npix = bad_msk.size

    nbad = bad_msk.sum()
    maskfrac = nbad/npix

    return bad_msk, maskfrac


def replace_flag_with_noise(*, rng, image, noise_image, weight, mask, flag):
    """
    Replace regions with noise.

    We currently pull the bitmask value from the descwl_shear_sims package.

    NOTE: This function operates IN PLACE on the input arrays.

    Parameters
    ----------
    rng : np.random.RandomState
        The RNG instance to use for noise generation.
    image : np.ndarray
        The image to interpolate.
    noise_image : np.ndarray
        The noise image to interpolate.
    weight : np.ndarray
        The weight map for the image.
    mask : np.ndarray
        The bit mask for the image.
    flag : int
        The mask bit for which the image and noise image will be replaced by noise.
    """

    mravel = mask.ravel()
    w, = np.where((mravel & flag) != 0)

    if w.size > 0:
        # ravel the rest
        imravel = image.ravel()
        nimravel = noise_image.ravel()
        wtravel = weight.ravel()

        medweight = np.median(weight)
        err = np.sqrt(1.0/medweight)
        imravel[w] = rng.normal(scale=err, size=w.size)
        nimravel[w] = rng.normal(scale=err, size=w.size)
        wtravel[w] = medweight
