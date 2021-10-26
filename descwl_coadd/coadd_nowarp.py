import logging
from .coadd import (
    DEFAULT_LOGLEVEL,
    CoaddObs,
    check_psf_dims,
    get_coadd_center,
    get_exp_and_noise,
    get_psf_exp,
    flag_bright_as_sat_in_coadd,
)

LOG = logging.getLogger('descwl_coadd.coadd_nowarp')


def make_coadd_obs_nowarp(
    exp, psf_dims, rng, remove_poisson,
    loglevel=DEFAULT_LOGLEVEL,
):
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
    loglevel : str, optional
        The logging level. Default is 'info'.

    Returns
    -------
    CoaddObs (inherits from ngmix.Observation)
    """

    coadd_data = make_coadd_nowarp(
        exp=exp,
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
        loglevel=loglevel,
    )


def make_coadd_nowarp(
    exp, psf_dims, rng, remove_poisson,
    loglevel=DEFAULT_LOGLEVEL,
):
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

    LOG.info('making coadd obs')

    check_psf_dims(psf_dims)

    _, noise_exp, var, mfrac_exp = get_exp_and_noise(
        exp_or_ref=exp, rng=rng, remove_poisson=remove_poisson,
    )
    cen, cen_skypos = get_coadd_center(
        coadd_wcs=exp.getWcs(), coadd_bbox=exp.getBBox(),
    )

    psf_exp = get_psf_exp(
        exp=exp,
        coadd_cen_skypos=cen_skypos,
        var=var,
    )

    flag_bright_as_sat_in_coadd(exp)
    flag_bright_as_sat_in_coadd(noise_exp)

    return dict(
        coadd_exp=exp,
        coadd_noise_exp=noise_exp,
        coadd_psf_exp=psf_exp,
        coadd_mfrac_exp=mfrac_exp,
    )
