from numba import njit
import numpy as np
import ngmix

import lsst.afw.math as afw_math
import lsst.afw.image as afw_image
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
from lsst.daf.butler import DeferredDatasetHandle
import lsst.log
import lsst.geom as geom
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel

from . import vis
from .interp import interpolate_image_and_noise
from esutil.pbar import PBar
import logging

LOG = logging.getLogger('descwl_coadd.coadd')

DEFAULT_INTERP = 'lanczos3'
DEFAULT_LOGLEVEL = 'info'

# areas in the image with these flags set will get interpolated note BRIGHT
# must be added to the mask plane by the caller

FLAGS2INTERP = ('BAD', 'CR', 'SAT', 'BRIGHT')

# No EDGE should make it into the coadds. We keep track of nothing else.
# Instead we keep track of the bits of interest in the mfrac array and separate
# array for BRIGHT

FLAGS2CHECK_FOR_COADD = ('EDGE', )

BOUNDARY_BIT_NAME = 'BOUNDARY'
BOUNDARY_SIZE = 3


def make_coadd_obs(
    exps, coadd_wcs, coadd_bbox, psf_dims, rng, remove_poisson,
    loglevel=DEFAULT_LOGLEVEL,
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
    loglevel : str, optional
        The logging level. Default is 'info'.

    Returns
    -------
    CoaddObs (inherits from ngmix.Observation) or None if no exposures
    were deemed valid
    """

    coadd_data = make_coadd(
        exps=exps, coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=rng, remove_poisson=remove_poisson,
        loglevel=loglevel,
    )

    if coadd_data['nkept'] == 0:
        return None

    return CoaddObs(
        coadd_exp=coadd_data["coadd_exp"],
        coadd_noise_exp=coadd_data["coadd_noise_exp"],
        coadd_psf_exp=coadd_data["coadd_psf_exp"],
        coadd_mfrac_exp=coadd_data["coadd_mfrac_exp"],
        ormask=coadd_data['ormask'],
        loglevel=loglevel,
    )


def make_coadd(
    exps, coadd_wcs, coadd_bbox, psf_dims, rng, remove_poisson,
    loglevel=DEFAULT_LOGLEVEL,
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
    loglevel : str, optional
        The logging level. Default is 'info'.

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

    logger = make_logger('coadd', loglevel)
    filter_label = exps[0].getFilterLabel()

    check_psf_dims(psf_dims)

    # sky center of this coadd within bbox
    coadd_cen, coadd_cen_skypos = get_coadd_center(
        coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
    )

    coadd_psf_bbox = get_coadd_psf_bbox(
        x=coadd_cen.x, y=coadd_cen.y, dim=psf_dims[0],
    )
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

    nkept = 0
    logger.info('warping and adding exposures')

    ormask = np.zeros(coadd_dims, dtype='i4')
    ormasks = [ormask, None, None, None]

    for exp_or_ref in PBar(exps):
        exp, noise_exp, medvar, mfrac_exp = get_exp_and_noise(
            exp_or_ref=exp_or_ref, rng=rng, remove_poisson=remove_poisson,
        )
        if exp is None:
            continue

        psf_exp = get_psf_exp(
            exp=exp,
            coadd_cen_skypos=coadd_cen_skypos,
            var=medvar,
        )

        assert psf_exp.variance.array[0, 0] == noise_exp.variance.array[0, 0]

        weight = 1/medvar

        # order must match stackers, wcss, bboxes
        exps2add = [exp, noise_exp, psf_exp, mfrac_exp]

        warps = get_warps(exps2add, wcss, bboxes, warpers)

        check_ok = [
            verify_boundary(_warp)
            for _warp, _verify in zip(warps, verify) if _verify
        ]
        print('check_ok:', check_ok)

        if all(check_ok):
            nkept += 1
            add_all(stackers, warps, ormasks, weight)

    if nkept == 0:
        return {'nkept': nkept}

    stacker.fill_stacked_masked_image(coadd_exp.maskedImage)
    noise_stacker.fill_stacked_masked_image(coadd_noise_exp.maskedImage)
    psf_stacker.fill_stacked_masked_image(coadd_psf_exp.maskedImage)
    mfrac_stacker.fill_stacked_masked_image(coadd_mfrac_exp.maskedImage)

    if not verify_boundary(coadd_exp):
        raise RuntimeError('boundary pixels found in coadd')
    if not verify_boundary(coadd_noise_exp):
        raise RuntimeError('boundary pixels found in noise coadd')

    flag_bright_as_sat_in_coadd(coadd_exp, ormask)
    flag_bright_as_sat_in_coadd(coadd_noise_exp, ormask)

    logger.info('making psf')
    psf = extract_coadd_psf(coadd_psf_exp, logger)
    coadd_exp.setPsf(psf)
    coadd_noise_exp.setPsf(psf)

    return dict(
        nkept=nkept,
        coadd_exp=coadd_exp,
        coadd_noise_exp=coadd_noise_exp,
        coadd_psf_exp=coadd_psf_exp,
        coadd_mfrac_exp=coadd_mfrac_exp,
        ormask=ormask,
    )


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


def verify_boundary(exp):
    """
    check to see if boundary pixels were included in the xp

    This might happen due to imprecision in the bounding box checks

    Parameters
    ----------
    exp: afw.image.ExposureF
        The exposure to check

    Returns
    -------
    True if no boundary pixels were found
    """

    flagval = exp.mask.getPlaneBitMask(BOUNDARY_BIT_NAME)
    if np.any(exp.mask.array & flagval != 0):
        # TODO sprint week keep track of what gets left out
        LOG.info('skipping warp that includes boundary pixels')
        if False:
            vis.show_image_and_mask(exp)
        return False
    else:
        return True


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


def extract_coadd_psf(coadd_psf_exp, logger):
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
        logger.info('zeroing %d bad psf pixels' % wbad[0].size)
        psf_image[wbad] = 0.0

    psf_image = psf_image.astype(float)
    psf_image *= 1/psf_image.sum()

    return KernelPsf(
        FixedKernel(afw_image.ImageD(psf_image))
    )


def get_exp_and_noise(exp_or_ref, rng, remove_poisson):
    """
    get the exposure (possibly from a deferred handle) and create
    a corresponding noise exposure

    TODO move interpolating BRIGHT downstream
    this is a longer term item, not for the sprint week

    Currently adding that plane if it doesn't exist

    Parameters
    ----------
    exp_or_ref: afw_image.ExposureF or DeferredDatasetHandle
        The input exposure, possible deferred

    Returns
    -------
    exp, noise_exp, medvar, mfrac_exp
        where medvar is the median of the variance for the exposure
        and mfrac_exp is an image of zeros and ones indicating interpolated pixels
    """
    if isinstance(exp_or_ref, DeferredDatasetHandle):
        exp = exp_or_ref.get()
    else:
        exp = exp_or_ref

    mdict = exp.mask.getMaskPlaneDict()
    if 'BRIGHT' not in mdict:
        # this adds it globally too
        exp.mask.addMaskPlane("BRIGHT")

    var = exp.variance.array
    weight = 1/var

    # we can now use BRIGHT directly as it is in our mask plane
    # flag_bright_as_sat(exp)

    noise_exp, medvar = get_noise_exp(
        exp=exp, rng=rng, remove_poisson=remove_poisson,
    )

    flags2interp = exp.mask.getPlaneBitMask(FLAGS2INTERP)
    iimage, inoise, mfrac_msk = interpolate_image_and_noise(
        image=exp.image.array,
        noise=noise_exp.image.array,
        weight=weight,
        bmask=exp.mask.array,
        bad_flags=flags2interp,
    )
    if iimage is None:
        return None, None, None, None

    exp.image.array[:, :] = iimage
    noise_exp.image.array[:, :] = inoise

    mfrac_exp = make_mfrac_exp(mfrac_msk=mfrac_msk, exp=exp)

    add_boundary_bit(exp)
    add_boundary_bit(noise_exp)

    return exp, noise_exp, medvar, mfrac_exp


def make_mfrac_exp(*, mfrac_msk, exp):
    """Make the masked fraction exposure.

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


def add_all(stackers, warps, ormasks, weight):
    """
    run do_add on all inputs
    """
    for _stacker, _warp, _ormask in zip(stackers, warps, ormasks):
        do_add(_stacker, _warp, weight, _ormask)


def do_add(stacker, warp, weight, ormask):
    """
    warp the exposure and add it

    Parameters
    ----------
    stacker: AccumulatorMeanStack
        A stacker, type
        lsst.pipe.tasks.accumulatorMeanStack.AccumulatorMeanStack
    warp: afw_image.ExposureF
        The exposure to warp and add
    weight: float
        Weight for this image in the stack
    ormask: array
        This will be or'ed with the warped mask
    """

    # TODO sprint week configure stacker to do this
    if ormask is not None:
        ormask |= warp.mask.array

    stacker.add_masked_image(warp, weight=weight)


def get_warps(exps, wcss, bboxes, warpers):
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
    for _exp, _wcs, _bbox, _warper in zip(exps, wcss, bboxes, warpers):
        warps.append(get_warp(_warper, _exp, _wcs, _bbox))

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
        maxBBox=exp.getBBox(),
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
    """

    stats_ctrl = get_coadd_stats_control()

    mask_map = AssembleCoaddTask.setRejectedMaskMapping(stats_ctrl)

    cefiv = stats_ctrl.getCalcErrorFromInputVariance()

    mask_threshold_dict = AccumulatorMeanStack.stats_ctrl_to_threshold_dict(stats_ctrl)

    return AccumulatorMeanStack(
        shape=coadd_dims,
        bit_mask_value=stats_ctrl.getAndMask(),
        mask_threshold_dict=mask_threshold_dict,
        mask_map=mask_map,
        no_good_pixels_mask=stats_ctrl.getNoGoodPixelsMask(),
        calc_error_from_input_variance=cefiv,
        compute_n_image=False,
    )


def get_coadd_stats_control():
    """
    get a afw_math.StatisticsControl with "and mask" set

    TODO sprint week get Eli's help with this

    Returns
    -------
    afw_math.StatisticsControl
    """

    mask = afw_image.Mask.getPlaneBitMask(FLAGS2CHECK_FOR_COADD)

    stats_ctrl = afw_math.StatisticsControl()
    stats_ctrl.setAndMask(mask)
    # not used by the Accumulator
    # stats_ctrl.setWeighted(True)
    stats_ctrl.setCalcErrorFromInputVariance(True)

    # the mask here is going to be just EDGE and we always
    # want to watch for it. it is a bug if EDGE is included
    # for regular images (not psf)
    mask_prop_thresh = {}
    for flagname in FLAGS2CHECK_FOR_COADD:
        mask_prop_thresh[flagname] = 0.0

    for plane, threshold in mask_prop_thresh.items():
        bit = afw_image.Mask.getMaskPlane(plane)
        stats_ctrl.setMaskPropagationThreshold(bit, threshold)

    return stats_ctrl


def get_dims_from_bbox(bbox):
    """
    get (nrows, ncols) numpy style from bbox

    Parameters
    ----------
    bbox: geom.Box2I
        The bbox

    Returns
    -------
    (nrows, ncols)
    """
    ncols = bbox.getEndX() - bbox.getBeginX()
    nrows = bbox.getEndY() - bbox.getBeginY()

    # dims is C/numpy ordering
    return nrows, ncols


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
        # averaged
        #
        # TODO sprint week getGain may not work for a calexp
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
    TODO sprint week revisit along with get_coadd_psf_bbox

    copied from https://github.com/beckermr/pizza-cutter/blob/
        66b9e443f840798996b659a4f6ce59930681c776/pizza_cutter/des_pizza_cutter/_se_image.py#L708
    """

    # compute the lower left corner of the stamp
    # we find the nearest pixel to the input (x, y)
    # and offset by half the stamp size in pixels
    # assumes the stamp size is odd
    # there is an assert for this below

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
        geom.Point2I(xmin + dim-1, ymin + dim-1),
    )


def get_coadd_psf_bbox(x, y, dim):
    """
    suggested by Matt Becker

    TODO sprint week revisit along with get_psf_bbox
    """
    xpix = int(x)
    ypix = int(y)

    xmin = (xpix - (dim - 1)/2)
    ymin = (ypix - (dim - 1)/2)

    return geom.Box2I(
        geom.Point2I(xmin, ymin),
        geom.Point2I(xmin + dim-1, ymin + dim-1),
    )


def get_coadd_center(coadd_wcs, coadd_bbox):
    """
    get the pixel and sky center of the coadd within the bbox

    Parameters
    -----------
    coadd_wcs: DM wcs
        The wcs for the coadd
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system

    Returns
    -------
    pixcen as Point2D, skycen as SpherePoint
    """
    pixcen = coadd_bbox.getCenter()
    skycen = coadd_wcs.pixelToSky(pixcen)

    return pixcen, skycen


class CoaddObs(ngmix.Observation):
    """
    Class representing a coadd observation

    Note that this class is a subclass of an `ngmix.Observation` and so it has
    all of the usual methods and attributes.

    Parameters
    ----------
    coadd_exp : afw_image.ExposureF
        The coadd exposure
    noise_exp : afw_image.ExposureF
        The coadded noise exposure
    coadd_psf_exp : afw_image.ExposureF
        The psf coadd
    coadd_mfrac_exp : afw_image.ExposureF
        The masked frraction image.
    ormask: array
        The ormask for the coadd
    loglevel : str, optional
        The logging level. Default is 'info'.
    """
    def __init__(
        self, *,
        coadd_exp,
        coadd_noise_exp,
        coadd_psf_exp,
        coadd_mfrac_exp,
        ormask,
        loglevel='info',
    ):

        self.log = make_logger('CoaddObs', loglevel)

        self.coadd_exp = coadd_exp
        self.coadd_psf_exp = coadd_psf_exp
        self.coadd_noise_exp = coadd_noise_exp
        self.coadd_mfrac_exp = coadd_mfrac_exp

        self._finish_init(ormask)

    def show(self):
        """
        show the output coadd in DS9
        """
        self.log.info('showing coadd in ds9')
        vis.show_image_and_mask(self.coadd_exp)

        vis.show_image(self.coadd_psf_exp.image.array, title='psf')
        # this will block
        # vis.show_images(
        #     [
        #         self.image,
        #         self.coadd_exp.mask.array,
        #         self.noise,
        #         self.coadd_noise_exp.mask.array,
        #         self.coadd_psf_exp.image.array,
        #         # self.weight,
        #     ],
        # )

    def _get_jac(self, *, cenx, ceny):
        """
        get jacobian at the coadd image center

        make an ngmix jacobian with specified center specified (this is not the
        position used to evaluate the jacobian)

        Parameters
        ----------
        cenx: float
            Center for the output ngmix jacobian (not place of evaluation)
        ceny: float
            Center for the output ngmix jacobian (not place of evaluation)
        """

        coadd_wcs = self.coadd_exp.getWcs()
        coadd_bbox = self.coadd_exp.getBBox()

        coadd_cen, coadd_cen_skypos = get_coadd_center(
            coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
        )

        dm_jac = coadd_wcs.linearizePixelToSky(coadd_cen, geom.arcseconds)
        matrix = dm_jac.getLinear().getMatrix()

        # note convention differences
        return ngmix.Jacobian(
            x=cenx,
            y=ceny,
            dudx=matrix[1, 1],
            dudy=-matrix[1, 0],
            dvdx=matrix[0, 1],
            dvdy=-matrix[0, 0],
        )

    def _get_coadd_psf_obs(self):
        """
        get the psf observation
        """

        coadd_wcs = self.coadd_exp.getWcs()
        coadd_bbox = self.coadd_exp.getBBox()

        coadd_cen, coadd_cen_skypos = get_coadd_center(
            coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
        )

        psf_obj = self.coadd_exp.getPsf()
        psf_image = psf_obj.computeKernelImage(coadd_cen).array

        psf_cen = (np.array(psf_image.shape)-1.0)/2.0

        psf_jac = self._get_jac(cenx=psf_cen[1], ceny=psf_cen[0])

        psf_err = psf_image.max()*0.0001
        psf_weight = psf_image*0 + 1.0/psf_err**2

        return ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jac,
        )

    def _finish_init(self, ormask):
        """
        finish the init by sending the image etc. to the
        Observation init
        """
        psf_obs = self._get_coadd_psf_obs()  # noqa

        image = self.coadd_exp.image.array
        noise = self.coadd_noise_exp.image.array
        mfrac = self.coadd_mfrac_exp.image.array

        var = self.coadd_exp.variance.array.copy()
        wnf = np.where(~np.isfinite(var))

        if wnf[0].size == image.size:
            raise ValueError('no good variance values')

        if wnf[0].size > 0:
            var[wnf] = -1

        weight = var.copy()
        weight[:, :] = 0.0

        w = np.where(var > 0)
        weight[w] = 1.0/var[w]

        if wnf[0].size > 0:
            # medval = np.sqrt(np.median(var[w]))
            # weight[wbad] = medval
            # TODO: add noise instead based on medval, need to send in rng
            image[wnf] = 0.0
            noise[wnf] = 0.0

        cen = (np.array(image.shape)-1)/2
        jac = self._get_jac(cenx=cen[1], ceny=cen[0])

        super().__init__(
            image=image,
            noise=noise,
            weight=weight,
            bmask=self.coadd_exp.mask.array.copy(),
            ormask=ormask,
            jacobian=jac,
            psf=psf_obs,
            store_pixels=False,
            mfrac=mfrac,
        )

        flags_for_maskfrac = self.coadd_exp.mask.getPlaneBitMask('BRIGHT')
        self.meta['bright_frac'] = get_masked_frac(
            mask=self.ormask,
            flags=flags_for_maskfrac,
        )
        self.meta['mask_frac'] = np.mean(mfrac)


def make_logger(name, loglevel):
    """
    make a logger with the specified loglevel
    """
    logger = lsst.log.getLogger(name)
    logger.setLevel(getattr(lsst.log, loglevel.upper()))
    return logger


def zero_bits(image, noise, mask, flags):
    """
    zero the image and noise where the input flags are set

    Parameters
    ----------
    image: array
        The image to be modified
    noise: array
        The noise image to be modified
    mask: array
        bitmask array to be checked
    flags: int
        An integer representing the bitmask
    """
    w = np.where((mask & flags) != 0)
    if w[0].size > 0:
        image[w] = 0.0
        noise[w] = 0.0


def flag_bright_as_sat(exp):
    """
    flag BRIGHT also as SAT

    TODO remove and do bright object masking downstream
    this is a longer term item, not for the sprint week
    """

    mask = exp.mask.array
    brightval = exp.mask.getPlaneBitMask('BRIGHT')
    satval = exp.mask.getPlaneBitMask('SAT')

    w = np.where((mask & brightval) != 0)
    if w[0].size > 0:
        mask[w] |= satval


def flag_bright_as_sat_in_coadd(exp, ormask):
    """
    wherever BRIGHT is set in the ormask, set
    the BRIGHT and SAT flags in the exposure mask
    SAT prevents detections
    """

    mask = exp.mask
    satval = mask.getPlaneBitMask('SAT')
    brightval = mask.getPlaneBitMask('BRIGHT')

    w = np.where(ormask & brightval != 0)
    if w[0].size > 0:
        mask.array[w] |= satval
        mask.array[w] |= brightval


@njit
def get_masked_frac(mask, flags):
    nrows, ncols = mask.shape

    npixels = mask.size
    nmasked = 0

    for row in range(nrows):
        for col in range(ncols):
            if mask[row, col] & flags != 0:
                nmasked += 1

    return nmasked/npixels


def get_psf_offset(pos):
    """
    the offset where the psf ends up landing with computeImage
    I don't know if this actually works or not for real psfs

    Parameters
    ----------
    pos: geom.Point2D
        The position requested for the reconstruction
    """
    return geom.Point2D(
        x=pos.x - int(pos.x + 0.5),
        y=pos.y - int(pos.y + 0.5),
    )


def check_psf_dims(psf_dims):
    """
    ensure psf dimensions are square and odd
    """
    assert psf_dims[0] == psf_dims[1]
    assert psf_dims[0] % 2 != 0
