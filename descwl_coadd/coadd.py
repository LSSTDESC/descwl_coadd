"""
TODO
    - actually try to run it
    - psf coadds
    - noise coadds; may want to generate the noise images on the fly
      within this code

So I'll just set andMask for EDGE and all other bits get passed on, and I can
check on the fly if there are any EDGE in the coadd region

Reject warps with NO_DATA set; for HSC this can be set for chips near the edge
of the focal plane

check EDGE and NO_DATA are not set in warp mask, eli expects NO_DATA should not be set

BUT, warps are the exact size of the coadd, which for standard patches is
bigger than an image so we can't do any of this yet!

Then check bits for masked fraction
    bad = np.where(mask & BADSTUFF != 0)
    maskfrac = bad[0].size / mask.size
"""
from numba import njit
import numpy as np
import ngmix
from lsst.afw.geom import makeSkyWcs
import lsst.afw.math as afw_math
import lsst.afw.image as afw_image
from lsst.pipe.tasks.accumulatorMeanStack import (
    AccumulatorMeanStack,
    stats_ctrl_to_threshold_dict,
)
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

DEFAULT_INTERP = 'lanczos3'
DEFAULT_LOGLEVEL = 'info'

FLAGS2INTERP = ('BAD', 'CR', 'SAT')


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
    CoaddObs (inherits from ngmix.Observation)
    """

    coadd_exp, coadd_noise_exp, coadd_psf_exp = make_coadd(
        exps=exps, coadd_wcs=coadd_wcs, coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=rng, remove_poisson=remove_poisson,
        loglevel=loglevel,
    )
    if coadd_exp is None:
        return None

    return CoaddObs(
        coadd_exp=coadd_exp,
        coadd_noise_exp=coadd_noise_exp,
        coadd_psf_exp=coadd_psf_exp,
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
    ExposureF for coadd
    """

    logger = make_logger('coadd', loglevel)
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
    coadd_exp = make_coadd_exposure(coadd_bbox, coadd_wcs)
    coadd_noise_exp = make_coadd_exposure(coadd_bbox, coadd_wcs)
    coadd_psf_exp = make_coadd_exposure(coadd_psf_bbox, coadd_psf_wcs)

    mask = afw_image.Mask.getPlaneBitMask('EDGE')
    coadd_dims = coadd_exp.image.array.shape

    stacker = make_stacker(mask=mask, coadd_dims=coadd_dims)
    noise_stacker = make_stacker(mask=mask, coadd_dims=coadd_dims)
    psf_stacker = make_stacker(mask=mask, coadd_dims=psf_dims)

    # can re-use the warper for each coadd type
    warp_config = afw_math.Warper.ConfigClass()
    warp_config.warpingKernelName = DEFAULT_INTERP
    warper = afw_math.Warper.fromConfig(warp_config)

    # will zip these with the exposures to warp and add
    stackers = [stacker, noise_stacker, psf_stacker]
    wcss = [coadd_wcs, coadd_wcs, coadd_psf_wcs]
    bboxes = [coadd_bbox, coadd_bbox, coadd_psf_bbox]

    nuse = 0
    logger.info('warping and adding exposures')

    for exp_or_ref in PBar(exps):
        exp, noise_exp, var = get_exp_and_noise(
            exp_or_ref=exp_or_ref, rng=rng, remove_poisson=remove_poisson,
        )
        if exp is None:
            continue

        psf_exp = get_psf_exp(
            exp=exp,
            coadd_cen_skypos=coadd_cen_skypos,
            var=var,
        )

        assert psf_exp.variance.array[0, 0] == noise_exp.variance.array[0, 0]

        weight = get_exp_weight(exp)

        # order must match stackers, wcss, bboxes
        exps2add = [exp, noise_exp, psf_exp]

        for _stacker, _exp, _wcs, _bbox in zip(stackers, exps2add, wcss, bboxes):
            warp_and_add(
                _stacker, warper, _exp, _wcs, _bbox, weight,
            )
        nuse += 1

    if nuse == 0:
        return None, None, None

    stacker.fill_stacked_masked_image(coadd_exp.maskedImage)
    noise_stacker.fill_stacked_masked_image(coadd_noise_exp.maskedImage)
    psf_stacker.fill_stacked_masked_image(coadd_psf_exp.maskedImage)

    logger.info('making psf')
    psf = extract_coadd_psf(coadd_psf_exp, logger)
    coadd_exp.setPsf(psf)
    coadd_noise_exp.setPsf(psf)

    return coadd_exp, coadd_noise_exp, coadd_psf_exp


def make_coadd_exposure(coadd_bbox, coadd_wcs):
    """
    make a coadd exposure with extra mask planes for
    rejected, clipped, sensor_edge

    Parameters
    ----------
    coadd_bbox: geom.Box2I
        the bbox for the coadd exposure
    coads_wcs: DM wcs
        The wcs for the coadd exposure

    Returns
    -------
    ExpsureF
    """
    coadd_exp = afw_image.ExposureF(coadd_bbox, coadd_wcs)
    coadd_exp.mask.addMaskPlane("REJECTED")
    coadd_exp.mask.addMaskPlane("CLIPPED")
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

    return KernelPsf(
        FixedKernel(
            afw_image.ImageD(psf_image.astype(np.float))
        )
    )


def get_exp_and_noise(exp_or_ref, rng, remove_poisson):
    """
    get the exposure (possibly from a deferred handle) and create
    a corresponding noise exposure

    TODO move interpolating BRIGHT into metadetect or mdet-lsst-sim

    Parameters
    ----------
    exp_or_ref: afw_image.ExposureF or DeferredDatasetHandle
        The input exposure, possible deferred

    Returns
    -------
    exp, noise_exp
    """
    if isinstance(exp_or_ref, DeferredDatasetHandle):
        exp = exp_or_ref.get()
    else:
        exp = exp_or_ref

    var = exp.variance.array
    weight = 1/var

    flag_bright_as_sat(exp)

    noise_exp, var = get_noise_exp(
        exp=exp, rng=rng, remove_poisson=remove_poisson,
    )

    # noise and image will have zeros in EDGE
    zero_bits(
        image=exp.image.array,
        noise=noise_exp.image.array,
        mask=exp.mask.array,
        flags=afw_image.Mask.getPlaneBitMask('EDGE'),
    )

    flags2interp = exp.mask.getPlaneBitMask(FLAGS2INTERP)
    iimage, inoise = interpolate_image_and_noise(
        image=exp.image.array,
        noise=noise_exp.image.array,
        weight=weight,
        bmask=exp.mask.array,
        bad_flags=flags2interp,
    )
    if iimage is None:
        return None, None

    exp.image.array[:, :] = iimage
    noise_exp.image.array[:, :] = inoise

    return exp, noise_exp, var


def warp_and_add(stacker, warper, exp, coadd_wcs, coadd_bbox, weight):
    """
    warp the exposure and add it

    Parameters
    ----------
    stacker: AccumulatorMeanStack
        A stacker, type
        lsst.pipe.tasks.accumulatorMeanStack.AccumulatorMeanStack
    warper: afw_math.Warper
        The warper
    exp: afw_image.ExposureF
        The exposure to warp and add
    coadd_wcs: DM wcs object
        The target wcs
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system
    weight: float
        Weight for this image in the stack
    """
    wexp = warper.warpExposure(
        coadd_wcs,
        exp,
        maxBBox=exp.getBBox(),
        destBBox=coadd_bbox,
    )

    stacker.add_masked_image(wexp, weight=weight)


def make_stacker(mask, coadd_dims):
    """
    make an AccumulatorMeanStack to do online coadding

    Parameters
    ----------
    mask: int
        The mask bits for andMask
    coadd_bbox: geom.Box2I
        The coadd bbox
    """

    stats_ctrl = get_coadd_stats_control(
        mask=afw_image.Mask.getPlaneBitMask('EDGE')
    )

    mask_map = AssembleCoaddTask.setRejectedMaskMapping(stats_ctrl)

    cefiv = stats_ctrl.getCalcErrorFromInputVariance()

    mask_threshold_dict = stats_ctrl_to_threshold_dict(stats_ctrl)
    return AccumulatorMeanStack(
        shape=coadd_dims,
        bit_mask_value=stats_ctrl.getAndMask(),
        mask_threshold_dict=mask_threshold_dict,
        mask_map=mask_map,
        no_good_pixels_mask=stats_ctrl.getNoGoodPixelsMask(),
        calc_error_from_input_variance=cefiv,
        compute_n_image=False,
    )


def get_coadd_stats_control(mask):
    """
    get a afw_math.StatisticsControl with "and mask" set

    Parameters
    ----------
    mask: mask for setAndMask
        Bits for which the pixels will not be added to the coadd.
        e.g. we would not let EDGE pixels get coadded

    Returns
    -------
    afw_math.StatisticsControl
    """
    stats_ctrl = afw_math.StatisticsControl()
    stats_ctrl.setAndMask(mask)
    # not used by the Accumulator
    # stats_ctrl.setWeighted(True)
    stats_ctrl.setCalcErrorFromInputVariance(True)

    # TODO when we make the BRIGHT plane, we will have BRIGHT at 0.0 here but
    # not so for others (e.g. SAT should be 0.1 or whatever)
    # TODO make this part of a configuration

    # we want to always propagate BRIGHT, which is currently translated to
    # SAT
    mask_prop_thresh = {
        'SAT': 0.0,
    }
    for plane, threshold in mask_prop_thresh.items():
        bit = afw_image.Mask.getMaskPlane(plane)
        stats_ctrl.setMaskPropagationThreshold(bit, threshold)

    return stats_ctrl


def get_exp_weight(exp):
    """
    get a afw_math.StatisticsControl with "and mask" set

    Parameters
    ----------
    mask: mask for setAndMask
        Bits for which the pixels will not be added to the coadd.
        e.g. we would not let EDGE pixels get coadded

    Returns
    -------
    afw_math.StatisticsControl
    """
    # Compute variance weight
    stats_ctrl = afw_math.StatisticsControl()
    stats_ctrl.setCalcErrorFromInputVariance(True)
    stat_obj = afw_math.makeStatistics(
        exp.variance,
        exp.mask,
        afw_math.MEANCLIP,
        stats_ctrl,
    )

    mean_var, mean_var_err = stat_obj.getResult(afw_math.MEANCLIP)
    weight = 1.0 / float(mean_var)
    return weight


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

    TODO gain correct separately in each amplifier, currently
    averaged

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
    signal = exp.image.array
    variance = exp.variance.array.copy()

    use = np.where(np.isfinite(variance) & np.isfinite(signal))

    if remove_poisson:
        gains = [
            amp.getGain() for amp in exp.getDetector().getAmplifiers()
        ]
        mean_gain = np.mean(gains)

        corrected_var = variance[use] - signal[use] / mean_gain

        var = np.median(corrected_var)
    else:
        var = np.median(variance[use])

    noise_image = rng.normal(scale=np.sqrt(var), size=signal.shape)

    ny, nx = signal.shape
    nmimage = afw_image.MaskedImageF(width=nx, height=ny)
    assert nmimage.image.array.shape == (ny, nx)

    nmimage.image.array[:, :] = noise_image
    nmimage.variance.array[:, :] = var
    nmimage.mask.array[:, :] = exp.mask.array[:, :]

    noise_exp = afw_image.ExposureF(nmimage)
    noise_exp.setPsf(exp.getPsf())
    noise_exp.setWcs(exp.getWcs())
    noise_exp.setFilterLabel(exp.getFilterLabel())
    noise_exp.setDetector(exp.getDetector())

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
    copied from https://github.com/beckermr/pizza-cutter/blob/66b9e443f840798996b659a4f6ce59930681c776/pizza_cutter/des_pizza_cutter/_se_image.py#L708
    """  # noqa

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
    loglevel : str, optional
        The logging level. Default is 'info'.
    """
    def __init__(
        self, *,
        coadd_exp,
        coadd_noise_exp,
        coadd_psf_exp,
        loglevel='info',
    ):

        self.log = make_logger('CoaddObs', loglevel)

        self.coadd_exp = coadd_exp
        self.coadd_psf_exp = coadd_psf_exp
        self.coadd_noise_exp = coadd_noise_exp

        self._finish_init()

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
        coadd_cen = coadd_wcs.getPixelOrigin()
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
        coadd_cen = coadd_wcs.getPixelOrigin()

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

    def _finish_init(self):
        """
        finish the init by sending the image etc. to the
        Observation init
        """
        psf_obs = self._get_coadd_psf_obs()  # noqa

        image = self.coadd_exp.image.array
        noise = self.coadd_noise_exp.image.array

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
            bmask=np.zeros(image.shape, dtype='i4'),
            ormask=self.coadd_exp.mask.array,
            jacobian=jac,
            psf=psf_obs,
            store_pixels=False,
        )

        flags_for_maskfrac = self.coadd_exp.mask.getPlaneBitMask('BRIGHT')
        self.meta['mask_frac'] = get_masked_frac(
            mask=self.ormask,
            flags=flags_for_maskfrac,
        )


def get_coadd_psf_wcs(coadd_wcs, coadd_bbox, psf_dims):
    """
    create the coadd psf wcs

    Parameters
    ----------
    coadd_wcs: DM wcs
        The coadd wcs
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system
    psf_dims: tuple
        The dimensions of the psf

    Returns
    -------
    A DM SkyWcs
    """
    cy, cx = (np.array(psf_dims)-1)/2
    psf_crpix = geom.Point2D(x=cx, y=cy)

    coadd_pixcen, coadd_skycen = get_coadd_center(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    )
    coadd_cd_matrix = coadd_wcs.getCdMatrix(coadd_pixcen)

    # coadd_skycen = coadd_wcs.getSkyOrigin()
    # coadd_cd_matrix = coadd_wcs.getCdMatrix(coadd_wcs.getPixelOrigin())

    return makeSkyWcs(
        crpix=psf_crpix,
        crval=coadd_skycen,
        cdMatrix=coadd_cd_matrix,
    )


def make_logger(name, loglevel):
    """
    make a logger with the specified loglevel
    """
    logger = lsst.log.getLogger(name)
    logger.setLevel(getattr(lsst.log, loglevel.upper()))
    return logger


def zero_bits(image, noise, mask, flags):
    w = np.where((mask & flags) != 0)
    # w = np.where(mask != 0)
    if w[0].size > 0:
        image[w] = 0.0
        noise[w] = 0.0


def flag_bright_as_sat(exp):
    """
    flag BRIGHT also as SAT so no detections will occur there

    we currently pull the bitmask value from the descwl_shear_sims
    package
    """

    mask = exp.mask.array
    brightval = exp.mask.getPlaneBitMask('BRIGHT')
    satval = exp.mask.getPlaneBitMask('SAT')

    w = np.where((mask & brightval) != 0)
    if w[0].size > 0:
        mask[w] |= satval


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
