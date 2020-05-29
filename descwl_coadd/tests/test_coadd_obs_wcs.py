import os
import pytest
import numpy as np

from descwl_shear_sims import Sim
from ..coadd import MultiBandCoadds


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present')
@pytest.mark.parametrize('wcs_type', ['tan', 'tan-sip'])
def test_coadd_obs_wcs_smoke(wcs_type):
    rng = np.random.RandomState(8312)
    sim = Sim(
        rng=rng,
        bands=["r", "i"],
        epochs_per_band=1,
        wcs_kws={'type': wcs_type},
    )
    data = sim.gen_sim()

    coadd_dims = (sim.coadd_dim,)*2

    # coadding individual bands as well as over bands
    psf_dim = sim.psf_dim
    psf_dims = (psf_dim,)*2

    _ = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
    )
