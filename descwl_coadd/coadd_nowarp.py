import logging
from .coadd import (
    CoaddObs,
    check_psf_dims,
    get_coadd_center,
    interp_and_get_noise,
    get_psf_exp,
    get_info_struct,
)
from .defaults import MAX_MASKFRAC

LOG = logging.getLogger('descwl_coadd.coadd_nowarp')


def make_coadd_obs_nowarp(exp, psf_dims, rng, remove_poisson):
    """
    Make a coadd from the input exposures and store in a CoaddObs, which
    inherits from ngmix.Observation. See make_coadd for docs on online
    coadding.

    Parameters
    ----------
    exp: Exposure
        Exposure to adapt to coadd observation form
    psf_dims: tuple
        The dimensions of the psf
    rng: np.random.RandomState
        The random number generator for making noise images
    remove_poisson: bool
        If True, remove the poisson noise from the variance
        estimate.

    Returns
    -------
    CoaddObs, exp_info
        CoaddObs inherits from ngmix.Observation

        exp_info structured array with fields 'exp_id', 'flags', 'maskfrac'
            Flags are set to non zero for skipped exposures
    """

    coadd_data = make_coadd_nowarp(
        exp=exp,
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


def make_coadd_nowarp(exp, psf_dims, rng, remove_poisson):
    """
    make a coadd from the input exposures, working in "online mode",
    adding each exposure separately.  This saves memory when
    the exposures are being read from disk

    Parameters
    ----------
    exp: Exposure
        Either a list of exposures or a list of DeferredDatasetHandle
    psf_dims: tuple
        The dimensions of the psf
    rng: np.random.RandomState
        The random number generator for making noise images
    remove_poisson: bool
        If True, remove the poisson noise from the variance
        estimate.

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

    LOG.info('making coadd obs')

    check_psf_dims(psf_dims)

    noise_exp, medvar, mfrac_exp, maskfrac = interp_and_get_noise(
        exp=exp, rng=rng, remove_poisson=remove_poisson,
        max_maskfrac=MAX_MASKFRAC,
    )
    cen, cen_skypos = get_coadd_center(
        coadd_wcs=exp.getWcs(), coadd_bbox=exp.getBBox(),
    )

    psf_exp = get_psf_exp(
        exp=exp,
        coadd_cen_skypos=cen_skypos,
        var=medvar,
    )

    try:
        exp_id = exp.getId()
    except AttributeError:
        exp_id = rng.randint(0, 2**31)

    exp_info = get_info_struct(1)
    exp_info['exp_id'] = exp_id
    exp_info['maskfrac'] = maskfrac

    return {
        'nkept': 1,
        'exp_info': exp_info,
        'coadd_exp': exp,
        'coadd_noise_exp': noise_exp,
        'coadd_psf_exp': psf_exp,
        'coadd_mfrac_exp': mfrac_exp,
    }
