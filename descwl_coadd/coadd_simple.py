"""
Coadding without warping
"""
import numpy as np
from .coadd import (
    DEFAULT_LOGLEVEL,
    CoaddObs,
    make_logger,
    check_psf_dims,
    get_coadd_center,
    get_coadd_psf_bbox,
    make_coadd_exposure,
    make_stacker,
    get_exp_and_noise,
    get_psf_exp,
    get_exp_weight,
    verify_coadd_edges,
    flag_bright_as_sat_in_coadd,
    extract_coadd_psf,
)
from esutil.pbar import PBar


def make_coadd_obs_nowarp(
    exps, psf_dims, rng, remove_poisson,
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

    coadd_data = make_coadd_nowarp(
        exps=exps,
        psf_dims=psf_dims,
        rng=rng, remove_poisson=remove_poisson,
        loglevel=loglevel,
    )
    if coadd_data is None:
        return None

    return CoaddObs(
        coadd_exp=coadd_data["coadd_exp"],
        coadd_noise_exp=coadd_data["coadd_noise_exp"],
        coadd_psf_exp=coadd_data["coadd_psf_exp"],
        coadd_mfrac_exp=coadd_data["coadd_mfrac_exp"],
        ormask=coadd_data['ormask'],
        loglevel=loglevel,
    )


def make_coadd_nowarp(
    exps, psf_dims, rng, remove_poisson,
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
    check_psf_dims(psf_dims)

    example_exp = exps[0]
    coadd_wcs = example_exp.wcs
    coadd_bbox = example_exp.getBBox()

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
    coadd_mfrac_exp = make_coadd_exposure(coadd_bbox, coadd_wcs)

    coadd_dims = example_exp.image.array.shape
    stacker = make_stacker(coadd_dims=coadd_dims)
    noise_stacker = make_stacker(coadd_dims=coadd_dims)
    psf_stacker = make_stacker(coadd_dims=psf_dims)
    mfrac_stacker = make_stacker(coadd_dims=coadd_dims)

    # will zip these with the exposures to add
    stackers = [stacker, noise_stacker, psf_stacker, mfrac_stacker]
    bboxes = [coadd_bbox, coadd_bbox, coadd_psf_bbox, coadd_bbox]

    # PSF will generally have NO_DATA in warp as we tend to use the same psf
    # stamp size for input and output psf and just zero out wherever there is
    # no data

    verify = [True, True, False, True]

    nuse = 0
    logger.info('adding exposures')

    ormask = np.zeros(coadd_dims, dtype='i4')
    ormasks = [ormask, None, None, None]

    for exp_or_ref in PBar(exps):
        exp, noise_exp, var, mfrac_exp = get_exp_and_noise(
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

        # order must match stackers, bboxes
        exps2add = [exp, noise_exp, psf_exp, mfrac_exp]

        for _stacker, _exp, _bbox, _verify, _ormask in zip(
            stackers, exps2add, bboxes, verify, ormasks
        ):
            add_nowarp(_stacker, _exp, _bbox, weight, _verify, _ormask)

        nuse += 1

    if nuse == 0:
        return None

    stacker.fill_stacked_masked_image(coadd_exp.maskedImage)
    noise_stacker.fill_stacked_masked_image(coadd_noise_exp.maskedImage)
    psf_stacker.fill_stacked_masked_image(coadd_psf_exp.maskedImage)
    mfrac_stacker.fill_stacked_masked_image(coadd_mfrac_exp.maskedImage)

    verify_coadd_edges(coadd_exp)
    verify_coadd_edges(coadd_noise_exp)

    flag_bright_as_sat_in_coadd(coadd_exp, ormask)
    flag_bright_as_sat_in_coadd(coadd_noise_exp, ormask)

    logger.info('making psf')
    psf = extract_coadd_psf(coadd_psf_exp, logger)
    coadd_exp.setPsf(psf)
    coadd_noise_exp.setPsf(psf)

    return dict(
        coadd_exp=coadd_exp,
        coadd_noise_exp=coadd_noise_exp,
        coadd_psf_exp=coadd_psf_exp,
        coadd_mfrac_exp=coadd_mfrac_exp,
        ormask=ormask,
    )


def add_nowarp(stacker, exp, weight, ormask):
    """
    Add the exposure

    Parameters
    ----------
    stacker: AccumulatorMeanStack
        A stacker, type
        lsst.pipe.tasks.accumulatorMeanStack.AccumulatorMeanStack
    exp: afw_image.ExposureF
        The exposure to add
    weight: float
        Weight for this image in the stack
    ormask: array
        This will be or'ed with the warped mask
    """

    if ormask is not None:
        ormask |= exp.mask.array

    stacker.add_masked_image(exp, weight=weight)
