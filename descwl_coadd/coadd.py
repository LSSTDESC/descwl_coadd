import numpy as np

import lsst.afw.math as afw_math
import lsst.afw.image as afw_image
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.daf.butler import DeferredDatasetHandle
import lsst.geom as geom
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel

from . import vis
from .interp import interpolate_image_and_noise
from .util import get_coadd_center
from .coadd_obs import CoaddObs
from .exceptions import WarpBoundaryError
from .procflags import HIGH_MASKFRAC, WARP_BOUNDARY
from .defaults import (
    DEFAULT_INTERP,
    FLAGS2INTERP,
    BOUNDARY_BIT_NAME,
    BOUNDARY_SIZE,
    MAX_MASKFRAC,
)
from esutil.pbar import PBar
import logging

LOG = logging.getLogger('descwl_coadd.coadd')


def make_coadd_obs(
    exps, coadd_wcs, coadd_bbox, psf_dims, rng, remove_poisson,
    max_maskfrac=MAX_MASKFRAC,
):
    """
    Make a coadd from the input exposures and store in a CoaddObs, which
    inherits from ngmix.Observation. See make_coadd for docs on online
    coadding.

    Parameters
    ----------
    exps: list
        Either a list of exposures or a list of DeferredDatasetHandle
    coadd_wcs: DM wcs object
        The target wcs
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system
    psf_dims: tuple
        The dimensions of the psf
    rng: np.random.RandomState
        The random number generator for making noise images
    remove_poisson: bool
        If True, remove the poisson noise from the variance
        estimate.
    max_maskfrac: float
        Maximum allowed masked fraction.  Images masked more than
        this will not be included in the coadd.

    Returns
    -------
    CoaddObs, exp_info
        CoaddObs inherits from ngmix.Observation

        exp_info is keyed by exposure id with each entry
        a dict holding info about the processing.  These are currently
            flags: zero if the exposure was kept
    """

    coadd_data = make_coadd(
        exps=exps, coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=rng, remove_poisson=remove_poisson,
    )

    exp_info = coadd_data['exp_info']

    if coadd_data['nkept'] == 0:
        coadd_obs = None
    else:
        coadd_obs = CoaddObs(
            coadd_exp=coadd_data["coadd_exp"],
            coadd_noise_exp=coadd_data["coadd_noise_exp"],
            coadd_psf_exp=coadd_data["coadd_psf_exp"],
            coadd_mfrac_exp=coadd_data["coadd_mfrac_exp"],
        )
    return coadd_obs, exp_info


def make_coadd(
    exps, coadd_wcs, coadd_bbox, psf_dims, rng, remove_poisson,
    max_maskfrac=MAX_MASKFRAC,
):
    """
    make a coadd from the input exposures, working in "online mode",
    adding each exposure separately.  This saves memory when
    the exposures are being read from disk

    Parameters
    ----------
    exps: list
        Either a list of exposures or a list of DeferredDatasetHandle
    coadd_wcs: DM wcs object
        The target wcs
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system
    psf_dims: tuple
        The dimensions of the psf
    rng: np.random.RandomState
        The random number generator for making noise images
    remove_poisson: bool
        If True, remove the poisson noise from the variance
        estimate.
    max_maskfrac: float
        Maximum allowed masked fraction.  Images masked more than
        this will not be included in the coadd.

    Returns
    -------
    coadd_data : dict
        A dict with keys and values:

            nkept: int
                Number of exposures deemed valid for coadding
            coadd_exp : ExposureF
                The coadded image.
            coadd_noise_exp : ExposureF
                The coadded noise image.
            coadd_psf_exp : ExposureF
                The coadded PSF image.
            coadd_mfrac_exp : ExposureF
                The fraction of SE images interpolated in each coadd pixel.
    """

    filter_label = exps[0].getFilterLabel()

    # this is the requested coadd psf dims
    check_psf_dims(psf_dims)

    # Get integer center of coadd and corresponding sky center.  This is used
    # to construct the coadd psf bounding box and to reconstruct the psfs
    coadd_cen_integer, coadd_cen_skypos = get_coadd_center(
        coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
    )

    coadd_psf_bbox = get_coadd_psf_bbox(cen=coadd_cen_integer, dim=psf_dims[0])
    coadd_psf_wcs = coadd_wcs

    # separately stack data, noise, and psf
    coadd_exp = make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label)
    coadd_noise_exp = make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label)
    coadd_psf_exp = make_coadd_exposure(coadd_psf_bbox, coadd_psf_wcs, filter_label)
    coadd_mfrac_exp = make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label)

    coadd_dims = coadd_exp.image.array.shape
    stacker = make_stacker(coadd_dims=coadd_dims)
    noise_stacker = make_stacker(coadd_dims=coadd_dims)
    psf_stacker = make_stacker(coadd_dims=psf_dims)
    mfrac_stacker = make_stacker(coadd_dims=coadd_dims)

    # can re-use the warper for each coadd type except the mfrac where we use
    # linear
    warp_config = afw_math.Warper.ConfigClass()
    warp_config.warpingKernelName = DEFAULT_INTERP
    warper = afw_math.Warper.fromConfig(warp_config)

    warp_config = afw_math.Warper.ConfigClass()
    warp_config.warpingKernelName = "bilinear"
    mfrac_warper = afw_math.Warper.fromConfig(warp_config)

    # will zip these with the exposures to warp and add
    stackers = [stacker, noise_stacker, psf_stacker, mfrac_stacker]
    wcss = [coadd_wcs, coadd_wcs, coadd_psf_wcs, coadd_wcs]
    bboxes = [coadd_bbox, coadd_bbox, coadd_psf_bbox, coadd_bbox]
    warpers = [warper, warper, warper, mfrac_warper]

    # PSF will generally have NO_DATA in warp as we tend to use the same psf
    # stamp size for input and output psf and just zero out wherever there is
    # no data

    verify = [True, True, False, True]

    LOG.info('warping and adding exposures')

    # TODO use exp.getId() when it arrives in weekly 44.  For now use an index
    # 0::len(exps)
    exp_info = {}

    for exp_id, exp_or_ref in enumerate(PBar(exps)):

        if isinstance(exp_or_ref, DeferredDatasetHandle):
            exp = exp_or_ref.get()
        else:
            exp = exp_or_ref

        # exp_id = exp.getId()
        exp_info[exp_id] = {'flags': 0}

        # exp is modified internally
        noise_exp, medvar, mfrac_exp, maskfrac = interp_and_get_noise(
            exp=exp, rng=rng, remove_poisson=remove_poisson,
            max_maskfrac=max_maskfrac,
        )
        exp_info[exp_id]['maskfrac'] = maskfrac
        if noise_exp is None:
            exp_info[exp_id]['flags'] |= HIGH_MASKFRAC
            continue

        psf_exp = get_psf_exp(
            exp=exp,
            coadd_cen_skypos=coadd_cen_skypos,
            var=medvar,
        )

        assert psf_exp.variance.array[0, 0] == noise_exp.variance.array[0, 0]

        # order must match stackers, wcss, bboxes, warpers
        exps2add = [exp, noise_exp, psf_exp, mfrac_exp]

        try:
            # Use an exception as a easy way to guarantee that if one fails to
            # verify we skip the whole set for coaddition
            warps = get_warps(exps2add, wcss, bboxes, warpers, verify)

            weight = 1/medvar
            add_all(stackers, warps, weight)

        except WarpBoundaryError as err:
            LOG.info('%s', err)
            exp_info[exp_id]['flags'] |= WARP_BOUNDARY

    nkept = len(
        [v for k, v in exp_info.items() if v['flags'] == 0]
    )
    result = {'nkept': nkept, 'exp_info': exp_info}

    if nkept > 0:

        stacker.fill_stacked_masked_image(coadd_exp.maskedImage)
        noise_stacker.fill_stacked_masked_image(coadd_noise_exp.maskedImage)
        psf_stacker.fill_stacked_masked_image(coadd_psf_exp.maskedImage)
        mfrac_stacker.fill_stacked_masked_image(coadd_mfrac_exp.maskedImage)

        LOG.info('making psf')
        psf = extract_coadd_psf(coadd_psf_exp)
        coadd_exp.setPsf(psf)
        coadd_noise_exp.setPsf(psf)

        result.update({
            'coadd_exp': coadd_exp,
            'coadd_noise_exp': coadd_noise_exp,
            'coadd_psf_exp': coadd_psf_exp,
            'coadd_mfrac_exp': coadd_mfrac_exp,
        })

    return result


def add_boundary_bit(exp):
    """
    add a bit to the boundary pixels; if this shows up in the warp we will not
    use it
    """
    exp.mask.addMaskPlane(BOUNDARY_BIT_NAME)
    val = exp.mask.getPlaneBitMask(BOUNDARY_BIT_NAME)

    marr = exp.mask.array
    marr[:BOUNDARY_SIZE, :] |= val
    marr[-BOUNDARY_SIZE:, :] |= val
    marr[:, :BOUNDARY_SIZE] |= val
    marr[:, -BOUNDARY_SIZE:] |= val

    if False:
        vis.show_image_and_mask(exp)


def verify_warp(exp):
    """
    check to see if boundary pixels were included in the xp

    This might happen due to imprecision in the bounding box checks

    Parameters
    ----------
    exp: afw.image.ExposureF
        The exposure to check

    Raises
    ------
    WarpBoundaryError
    """

    tocheck = [BOUNDARY_BIT_NAME, 'NO_DATA']
    all_flags = exp.mask.getPlaneBitMask(tocheck)

    if np.any(exp.mask.array & all_flags != 0):
        # give fine grained feedback on what happened
        message = []
        for flagname in tocheck:
            w = np.where(exp.mask.array & all_flags != 0)
            if w[0].size > 0:
                message += [
                    f'{w[0].size} pixels with {flagname}'
                ]
            message = ' and '.join(message)
            raise WarpBoundaryError(message)

        if False:
            vis.show_image_and_mask(exp)


def make_coadd_exposure(coadd_bbox, coadd_wcs, filter_label):
    """
    make a coadd exposure with extra mask planes for
    rejected, clipped, sensor_edge

    Parameters
    ----------
    coadd_bbox: geom.Box2I
        the bbox for the coadd exposure
    coads_wcs: DM wcs
        The wcs for the coadd exposure
    filter_label: FilterLabel
        Filter label to set

    Returns
    -------
    ExpsureF
    """
    coadd_exp = afw_image.ExposureF(coadd_bbox, coadd_wcs)
    coadd_exp.setFilterLabel(filter_label)

    # these planes are added by DM, add them here for consistency
    coadd_exp.mask.addMaskPlane("REJECTED")
    coadd_exp.mask.addMaskPlane("CLIPPED")

    # From J. Bosch: the PSF is discontinuous in the neighborhood of this pixel
    # because the number of inputs to the coadd changed due to chip boundaries
    coadd_exp.mask.addMaskPlane("SENSOR_EDGE")

    return coadd_exp


def extract_coadd_psf(coadd_psf_exp):
    """
    extract the PSF image, zeroing the image where
    there are "bad" pixels, associated with areas not
    covered by the input psfs

    Parameters
    ----------
    coadd_psf_exp: afw_image.ExposureF
        The psf exposure

    Returns
    -------
    KernelPsf
    """
    psf_image = coadd_psf_exp.image.array

    wbad = np.where(~np.isfinite(psf_image))
    if wbad[0].size == psf_image.size:
        raise ValueError('no good pixels in the psf')

    if wbad[0].size > 0:
        LOG.info('zeroing %d bad psf pixels' % wbad[0].size)
        psf_image[wbad] = 0.0

    psf_image = psf_image.astype(float)
    psf_image *= 1/psf_image.sum()

    return KernelPsf(
        FixedKernel(afw_image.ImageD(psf_image))
    )


def interp_and_get_noise(exp, rng, remove_poisson, max_maskfrac):
    """
    get the exposure (possibly from a deferred handle) and create
    a corresponding noise exposure

    Artifacts are interpolated and the bit INTRP is set

    Parameters
    ----------
    exp: afw_image.ExposureF
        The input exposure.  This is modified in place.
    rng: np.random.RandomState
        Used for generating the noise exposure
    remove_poisson: bool
        If True, remove the poisson noise from the variance estimate
    max_maskfrac: float
        Maximum allowed masked fraction.  Images masked more than
        this will not be included in the coadd.

    Returns
    -------
    noise_exp, medvar, mfrac_exp
        where medvar is the median of the variance for the exposure and
        mfrac_exp is an image of zeros and ones indicating interpolated pixels

        If the image is too heavily masked, None, None, None is returned
    """

    var = exp.variance.array
    weight = 1/var

    noise_exp, medvar = get_noise_exp(
        exp=exp, rng=rng, remove_poisson=remove_poisson,
    )

    flags2interp = exp.mask.getPlaneBitMask(FLAGS2INTERP)
    iimage, inoise, mfrac_msk, maskfrac = interpolate_image_and_noise(
        image=exp.image.array,
        noise=noise_exp.image.array,
        weight=weight,
        bmask=exp.mask.array,
        bad_flags=flags2interp,
        max_maskfrac=max_maskfrac,
    )
    if iimage is None:
        return None, None, None, maskfrac
    else:

        if maskfrac > 0:
            interp_flag = exp.mask.getPlaneBitMask('INTRP')
            exp.mask.array[mfrac_msk] |= interp_flag
            noise_exp.mask.array[mfrac_msk] |= interp_flag

        exp.image.array[:, :] = iimage
        noise_exp.image.array[:, :] = inoise

        mfrac_exp = make_mfrac_exp(mfrac_msk=mfrac_msk, exp=exp)

        add_boundary_bit(exp)
        add_boundary_bit(noise_exp)

        return noise_exp, medvar, mfrac_exp, maskfrac


def make_mfrac_exp(*, mfrac_msk, exp):
    """
    Make the masked fraction exposure.

    Parameter
    ---------
    mfrac_msk : np.ndarray
        A boolean image with True where interpolation was done and False otherwise.
    exp : ExposureF
        The coadd exposure for this `mfrac`.

    Returns
    -------
    mfrac_exp : ExposureF
        The masked fraction exposure.
    """
    ny, nx = mfrac_msk.shape
    mfrac_img = afw_image.MaskedImageF(width=nx, height=ny)
    assert mfrac_img.image.array.shape == (ny, nx)

    mfrac_img.image.array[:, :] = mfrac_msk.astype(float)
    mfrac_img.variance.array[:, :] = 0
    mfrac_img.mask.array[:, :] = exp.mask.array[:, :]

    mfrac_exp = afw_image.ExposureF(mfrac_img)
    mfrac_exp.setPsf(exp.getPsf())
    mfrac_exp.setWcs(exp.getWcs())
    mfrac_exp.setFilterLabel(exp.getFilterLabel())
    mfrac_exp.setDetector(exp.getDetector())

    return mfrac_exp


def add_all(stackers, warps, weight):
    """
    run do_add on all inputs
    """
    for _stacker, _warp in zip(stackers, warps):
        _stacker.add_masked_image(_warp, weight=weight)


def get_warps(exps, wcss, bboxes, warpers, verify):
    """
    get a list of all warps for the input

    Parameters
    ----------
    exps: [ExposureF]
        List of exposures to warp
    wcss: [wcs]
        List of wcs
    bboxes: [lsst.geom.Box2I]
        List of bounding boxes
    warpers: [afw_math.Warper]
        List of warpers

    Returns
    -------
    waprs: [ExposureF]
        List of warped exposures
    """
    warps = []
    for _exp, _wcs, _bbox, _warper, _verify in zip(
        exps, wcss, bboxes, warpers, verify
    ):
        warp = get_warp(_warper, _exp, _wcs, _bbox)
        if _verify:
            verify_warp(warp)
        warps.append(warp)

    return warps


def get_warp(warper, exp, coadd_wcs, coadd_bbox):
    """
    warp the exposure and add it

    Parameters
    ----------
    warper: afw_math.Warper
        The warper
    exp: afw_image.ExposureF
        The exposure to warp and add
    coadd_wcs: DM wcs object
        The target wcs
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system
    """
    wexp = warper.warpExposure(
        coadd_wcs,
        exp,
        destBBox=coadd_bbox,
    )
    return wexp


def make_stacker(coadd_dims):
    """
    make an AccumulatorMeanStack to do online coadding

    Parameters
    ----------
    coadd_dims: tuple/list
        The coadd dimensions

    Returns
    -------
    lsst.meas.algorithms.AccumulatorMeanStack

    Notes
    -----
    bit_mask_value = 0 says no filtering on mask plane
    mask_threshold_dict={} says propagate all mask plane bits to the
        coadd mask plane
    mask_map=[] says do not remap any mask plane bits to new values; negotiable
    no_good_pixels_mask=None says use default NO_DATA mask plane in coadds for
        areas that have no inputs; shouldn't matter since no filtering
    calc_error_from_input_variance=True says use the individual variances planes
        to predict coadd variance
    compute_n_image=False says do not compute number of images input to
        each pixel
    """

    stats_ctrl = afw_math.StatisticsControl()

    # TODO after sprint
    # Eli will fix bug and will set no_good_pixels_mask to None
    return AccumulatorMeanStack(
        shape=coadd_dims,
        bit_mask_value=0,
        mask_threshold_dict={},
        mask_map=[],
        # no_good_pixels_mask=None,
        no_good_pixels_mask=stats_ctrl.getNoGoodPixelsMask(),
        calc_error_from_input_variance=True,
        compute_n_image=False,
    )


def get_noise_exp(exp, rng, remove_poisson):
    """
    get a noise image based on the input exposure

    Parameters
    ----------
    exp: afw.image.ExposureF
        The exposure upon which to base the noise
    rng: np.random.RandomState
        The random number generator for making the noise image
    remove_poisson: bool
        If True, remove the poisson noise from the variance
        estimate.

    Returns
    -------
    noise exposure
    """

    noise_exp = afw_image.ExposureF(exp, deep=True)

    signal = exp.image.array
    variance = exp.variance.array

    use = np.where(np.isfinite(variance) & np.isfinite(signal))

    if remove_poisson:
        # TODO sprint week gain correct separately in each amplifier, currently
        # averaged.  Morgan
        #
        # TODO sprint week getGain may not work for a calexp Morgan
        gains = [
            amp.getGain() for amp in exp.getDetector().getAmplifiers()
        ]
        mean_gain = np.mean(gains)

        corrected_var = variance[use] - signal[use] / mean_gain

        var = np.median(corrected_var)
    else:
        var = np.median(variance[use])

    noise_image = rng.normal(scale=np.sqrt(var), size=signal.shape)

    noise_exp.image.array[:, :] = noise_image
    noise_exp.variance.array[:, :] = var

    return noise_exp, var


def get_psf_exp(
    exp,
    coadd_cen_skypos,
    var,
):
    """
    create a psf exposure to be coadded, rendered at the
    position in the exposure corresponding to the center of the
    coadd

    Parameters
    ----------
    exp: afw_image.ExposureF
        The exposure
    coadd_cen_skypos: SpherePoint
        The sky position of the center of the coadd within its
        bbox
    var: float
        The variance to set in the psf variance map

    Returns
    -------
    psf ExposureF
    """

    wcs = exp.getWcs()
    pos = wcs.skyToPixel(coadd_cen_skypos)

    psf_obj = exp.getPsf()
    psf_image = psf_obj.computeImage(pos).array

    psf_dim = psf_image.shape[0]

    psf_bbox = get_psf_bbox(pos=pos, dim=psf_dim)

    # wcs same as SE exposure
    psf_exp = afw_image.ExposureF(psf_bbox, wcs)
    psf_exp.image.array[:, :] = psf_image
    psf_exp.variance.array[:, :] = var
    psf_exp.mask.array[:, :] = 0

    psf_exp.setFilterLabel(exp.getFilterLabel())
    detector = DetectorWrapper().detector
    psf_exp.setDetector(detector)

    return psf_exp


def get_psf_bbox(pos, dim):
    """
    get a bounding box for the psf at the given position

    Parameters
    ----------
    pos: lsst.geom.Point2D
        The position at which to get the bounding box
    dim: int
        The dimension of the psf, must be odd

    Returns
    -------
    lsst.geom.Box2I

    Notes
    ------
    copied from https://github.com/beckermr/pizza-cutter/blob/
        66b9e443f840798996b659a4f6ce59930681c776/pizza_cutter/des_pizza_cutter/_se_image.py#L708
    """

    assert isinstance(pos, geom.Point2D)
    assert dim % 2 != 0

    # compute the lower left corner of the stamp xmin, ymin
    #
    # we first find the _nearest pixel_ to the input (x, y) and offset by half
    # the stamp size in pixels.
    #
    # assumes the stamp size is odd, which is asserted above and is
    # also built into the asserts below

    x = pos.x
    y = pos.y

    half = (dim - 1) / 2
    x_cen = np.floor(x+0.5)
    y_cen = np.floor(y+0.5)

    # make sure this is true so pixel index math is ok
    assert y_cen - half == int(y_cen - half)
    assert x_cen - half == int(x_cen - half)

    # compute bounds in Piff wcs coords
    xmin = int(x_cen - half)
    ymin = int(y_cen - half)

    return geom.Box2I(
        geom.Point2I(xmin, ymin),
        geom.Extent2I(dim, dim),
    )


def get_coadd_psf_bbox(cen, dim):
    """
    compute the bounding box for the coadd, based on the coadd
    center as an integer position (Point2I) and the dimensions

    Parameters
    ----------
    cen: lsst.geom.Point2I
        The center.  Should be gotten from bbox.getCenter() to
        provide an integer position
    dim: int
        The dimensions of the psf, must be odd

    Returns
    -------
    lsst.geom.Box2I
    """

    assert isinstance(cen, geom.Point2I)
    assert dim % 2 != 0

    xmin = cen.x - (dim - 1)//2
    ymin = cen.y - (dim - 1)//2

    return geom.Box2I(
        geom.Point2I(xmin, ymin),
        geom.Extent2I(dim, dim),
    )


def check_psf_dims(psf_dims):
    """
    ensure psf dimensions are square and odd
    """
    assert psf_dims[0] == psf_dims[1]
    assert psf_dims[0] % 2 != 0
