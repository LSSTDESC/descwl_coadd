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
    coadd_dims = (sim.coadd_dim, )*2
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim._coadd_wcs,  # TODO:  make getter
        coadd_dims=coadd_dims,
    )

    coadd = coadds.get_coadd()
    assert coadd.coadd_exp.image.array.shape == coadd_dims
    assert coadd.image.shape == coadd_dims

    for band in coadds.bands:
        bcoadd = coadds.get_coadd(band=band)
        assert bcoadd.coadd_exp.image.array.shape == coadd_dims
        assert bcoadd.image.shape == coadd_dims

    # not coadding individual bands
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim._coadd_wcs,  # TODO:  make getter
        coadd_dims=coadd_dims,
        byband=False,
    )

    for band in coadds.bands:
        assert band not in coadds.coadds
