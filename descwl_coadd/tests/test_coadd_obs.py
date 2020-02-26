import numpy as np

from descwl_shear_sims import Sim
from ..coadd import MultiBandCoadds


def test_coadd_obs_smoke():
    rng = np.random.RandomState(8312)
    sim = Sim(
        rng=rng,
        epochs_per_band=2,
    )
    data = sim.gen_sim()

    coadd_dims = (sim.coadd_dim,)*2

    # coadding individual bands as well as over bands
    psf_dim = int(sim.psf_dim/np.sqrt(3))
    if psf_dim % 2 == 0:
        psf_dim -= 1
    psf_dims = (psf_dim,)*2

    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
    )

    coadd = coadds.get_coadd()
    assert coadd.coadd_exp.image.array.shape == coadd_dims
    assert coadd.image.shape == coadd_dims
    assert coadd.noise.shape == coadd_dims
    assert coadd.psf.image.shape == psf_dims

    for band in coadds.bands:
        bcoadd = coadds.get_coadd(band=band)
        assert bcoadd.coadd_exp.image.array.shape == coadd_dims
        assert bcoadd.image.shape == coadd_dims
        assert bcoadd.noise.shape == coadd_dims
        assert bcoadd.psf.image.shape == psf_dims

    # not coadding individual bands
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
        byband=False,
    )

    for band in coadds.bands:
        assert band not in coadds.coadds


def test_coadd_obs_weights():
    """
    ensure the psf and noise var are set close, so
    that the relative weight is right
    """
    rng = np.random.RandomState(8312)
    sim = Sim(
        rng=rng,
        epochs_per_band=2,
    )
    data = sim.gen_sim()

    coadd_dims = (sim.coadd_dim,)*2

    # coadding individual bands as well as over bands
    psf_dim = int(sim.psf_dim/np.sqrt(3))
    if psf_dim % 2 == 0:
        psf_dim -= 1
    psf_dims = (psf_dim,)*2

    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
    )

    tol = 1.0e-3

    for exp, pexp, nexp in zip(coadds.exps, coadds.psf_exps, coadds.noise_exps):

        emed = np.median(exp.variance.array)
        pmed = np.median(pexp.variance.array)
        nmed = np.median(nexp.variance.array)

        assert abs(pmed/emed-1) < 1.0e-3
        assert abs(nmed/emed-1) < 1.0e-3
