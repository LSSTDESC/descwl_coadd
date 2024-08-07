import logging
from .coadd_obs import CoaddObs
from .coadd import (
    check_psf_dims,
    get_coadd_center,
    get_noise_exp,
    get_bad_mask,
    make_mfrac_exp,
    get_psf_exp,
    get_info_struct,
    get_default_image_interpolator,
)
from .procflags import HIGH_MASKFRAC
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


def make_coadd_nowarp(exp, psf_dims, rng, remove_poisson, interpolator=None):
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
    interpolator: interpolator object, optional
        An object or function used to interpolate pixels.
        Must be callable as interpolator(exposure)

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

    if interpolator is None:
        interpolator = get_default_image_interpolator()

    check_psf_dims(psf_dims)

    try:
        exp_id = exp.getId()
    except AttributeError:
        exp_id = 0

    exp_info = get_info_struct(1)
    exp_info['exp_id'] = exp_id

    nkept = 0

    bad_msk, maskfrac = get_bad_mask(exp)
    exp_info['maskfrac'] = maskfrac

    if maskfrac < MAX_MASKFRAC:

        noise_exp, medvar = get_noise_exp(
            exp=exp, rng=rng, remove_poisson=remove_poisson,
        )
        mfrac_exp = make_mfrac_exp(mfrac_msk=bad_msk, exp=exp)

        if maskfrac > 0:
            # images modified internally
            interpolator.run(exp)
            interpolator.run(noise_exp)

        cen, cen_skypos = get_coadd_center(
            coadd_wcs=exp.getWcs(), coadd_bbox=exp.getBBox(),
        )

        psf_exp = get_psf_exp(
            exp=exp,
            coadd_cen_skypos=cen_skypos,
            var=medvar,
        )
        nkept += 1
    else:
        exp_info['flags'] |= HIGH_MASKFRAC

    result = {
        'nkept': nkept,
        'exp_info': exp_info,
    }
    if nkept > 0:
        result.update({
            'coadd_exp': exp,
            'coadd_noise_exp': noise_exp,
            'coadd_psf_exp': psf_exp,
            'coadd_mfrac_exp': mfrac_exp,
        })

    return result
