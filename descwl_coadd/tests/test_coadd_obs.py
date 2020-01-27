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

    # coadding individual bands as well as over bands
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=sim.coadd_dims,
        coadd_psf_wcs=sim.coadd_psf_wcs,
        coadd_psf_dims=sim.coadd_psf_dims,
    )

    coadd = coadds.get_coadd()
    assert coadd.coadd_exp.image.array.shape == sim.coadd_dims
    assert coadd.image.shape == sim.coadd_dims
    assert coadd.noise.shape == sim.coadd_dims
    assert coadd.psf.image.shape == sim.coadd_psf_dims

    for band in coadds.bands:
        bcoadd = coadds.get_coadd(band=band)
        assert bcoadd.coadd_exp.image.array.shape == sim.coadd_dims
        assert bcoadd.image.shape == sim.coadd_dims
        assert bcoadd.noise.shape == sim.coadd_dims

        assert bcoadd.psf.image.shape == sim.coadd_psf_dims

    # not coadding individual bands
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=sim.coadd_dims,
        coadd_psf_wcs=sim.coadd_psf_wcs,
        coadd_psf_dims=sim.coadd_psf_dims,
        byband=False,
    )

    for band in coadds.bands:
        assert band not in coadds.coadds
