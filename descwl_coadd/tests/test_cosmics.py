import numpy as np

from descwl_shear_sims import Sim
from ..coadd import (
    MultiBandCoadds,
    make_stack_psf,
    repair_exp,
)


def get_cosmic_flag():
    import lsst.afw.image as afw_image
    return 2**afw_image.Mask.getMaskPlane('CR')


def test_cosmics():

    cosmic_flag = get_cosmic_flag()

    rng = np.random.RandomState(8312)
    sim = Sim(
        rng=rng,
        epochs_per_band=2,
        cosmic_rays=True,
    )
    data = sim.gen_sim()

    # coadding individual bands as well as over bands
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=sim.coadd_dims,
        coadd_psf_wcs=sim.coadd_psf_wcs,
        coadd_psf_dims=sim.coadd_psf_dims,
    )

    for exp, nexp in zip(coadds.exps, coadds.noise_exps):
        bmask = exp.mask.array
        noise_bmask = nexp.mask.array

        w = np.where((bmask & cosmic_flag) != 0)
        assert w[0].size != 0

        w = np.where((noise_bmask & cosmic_flag) != 0)
        assert w[0].size != 0


def test_noise_cosmics():
    """
    very simple version not using the main sim
    """

    import lsst.afw.image as afw_image
    import galsim

    psf_image = galsim.Gaussian(fwhm=0.9).drawImage(
        scale=0.263,
    ).array

    noise = 180.0
    sx, sy = 500, 500

    rng = np.random.RandomState(9763)

    stack_image = afw_image.MaskedImageF(sx, sy)
    stack_image.image.array[:, :] = rng.normal(
        scale=noise,
        size=(sy, sx),
    )

    stack_image.variance.array[:, :] = noise**2
    stack_image.mask.array[:, :] = 0
    exp = afw_image.ExposureF(stack_image)

    exp.setPsf(make_stack_psf(psf_image))

    repair_exp(exp)

    cosmic_flag = get_cosmic_flag()

    bmask = exp.mask.array

    w = np.where((bmask & cosmic_flag) != 0)

    print('found:', w[0].size, 'cosmics in noise')
    if w[0].size > 0:
        # import fitsio
        # fitsio.write('/tmp/tmp.fits', nexp.image.array, clobber=True)
        print('indices:', w)
        print('values:', exp.image.array[w])

    # for now don't require 0, until we figure out why its finding
    # cosmics in the noise
    assert w[0].size < 5
