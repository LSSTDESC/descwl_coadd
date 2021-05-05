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

Note, warps are the exact size of the coadd, which for standard patches is
bigger than an image so we can't do any of this yet!

Then check bits for masked fraction
    bad = np.where(mask & BADSTUFF != 0)
    maskfrac = bad[0].size / mask.size
"""
import lsst.afw.math as afw_math
import lsst.afw.image as afw_image
from lsst.pipe.tasks.accumulatorMeanStack import (
    AccumulatorMeanStack,
    stats_ctrl_to_threshold_dict,
)
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
from lsst.daf.butler import DeferredDatasetHandle

INTERP = 'lanczos3'


def make_online_coadd(exps, coadd_wcs, coadd_bbox):
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
        The bounding fox for the coadd

    Returns
    -------
    ExposureF for coadd
    """
    coadd_exp = afw_image.ExposureF(coadd_bbox, coadd_wcs)

    coadd_stats_ctrl = get_coadd_stats_control(
        afw_image.Mask.getPlaneBitMask('EDGE')
    )

    stacker = make_online_stacker(
        coadd_stats_ctrl, coadd_exp.image.array.shape,
    )

    warp_config = afw_math.Warper.ConfigClass()
    warp_config.warpingKernelName = INTERP
    warper = afw_math.Warper.fromConfig(warp_config)

    for texp in exps:
        if isinstance(texp, DeferredDatasetHandle):
            exp = texp.get()
        else:
            exp = texp

        weight = get_exp_weight(exp)

        wexp = warper.warpExposure(
            coadd_wcs,
            exp,
            maxBBox=exp.getBBox(),
            destBBox=coadd_bbox,
        )

        stacker.add_masked_image(wexp.image, weight=weight)

    stacker.fill_stacked_masked_image(coadd_exp.image)

    return coadd_exp


def make_online_stacker(stats_ctrl, coadd_dims):
    """
    make an AccumulatorMeanStack to online coadding

    Parameters
    ----------
    stats_ctrl: afw_math.StatisticsControl
        The stats control
    coadd_bbox: geom.Box2I
        The coadd bbox
    """

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
