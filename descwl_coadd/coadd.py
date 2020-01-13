import ngmix


class Coadd(ngmix.Observation):
    """
    note we cannot do the psf coadding using the stack because we do not
    have simulation psfs in the right format

    Parameters
    ----------
    data: list of observations
        For a single band.  Should have image, weight, noise, wcs attributes,
        as well as get_psf method.  For example see the simple sim from
        descwl_shear_testing
    """
    pass
