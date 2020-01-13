import numpy as np
import ngmix

from descwl_shear_sims import Sim
from ..coadd import CoaddObs


def test_coadd_obs_smoke():
    rng = np.random.RandomState(34123)
    sim = Sim(
        rng=rng,
        epochs_per_band=2,
    )
    data = sim.gen_sim()

    mbobs = ngmix.MultiBandObsList()
    for band in data:

        coadd_dims = [sim.coadd_dim]*2
        band_coadd_obs = CoaddObs(
            data=data[band],
            coadd_wcs=sim._coadd_wcs,  # TODO:  make getter
            coadd_dims=coadd_dims,
        )

        olist = ngmix.ObsList()
        olist.append(band_coadd_obs)
        mbobs.append(olist)
