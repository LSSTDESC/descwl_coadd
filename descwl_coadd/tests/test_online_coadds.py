import numpy as np

from descwl_shear_sims.sim import make_dmsim
from descwl_shear_sims.sim.galaxy_catalogs import FixedGalaxyCatalog
from descwl_shear_sims.sim.psfs import make_psf

from ..online_coadds import make_online_coadd_obs


def test_online_coadds_smoke():
    rng = np.random.RandomState(313)
    coadd_dim = 100
    psf_dim = 51
    buff = 10
    cat = FixedGalaxyCatalog(
        rng=rng, coadd_dim=coadd_dim, buff=buff, layout='grid',
        mag=22, hlr=0.5,
    )
    g1 = 0.02
    g2 = 0.00
    psf = make_psf(psf_type='gauss')
    bands = ['r', 'i', 'z']
    epochs_per_band = 3
    sim_data = make_dmsim(
        rng=rng,
        galaxy_catalog=cat,
        coadd_dim=coadd_dim,
        g1=g1,
        g2=g2,
        psf=psf,
        bands=bands,
        epochs_per_band=epochs_per_band,
    )

    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = [data['exp'] for data in bdata[band]]

        coadd_obs = make_online_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        coadd_dims = (coadd_dim, )*2
        psf_dims = (psf_dim, )*2
        assert coadd_obs.image.shape == coadd_dims
        assert coadd_obs.psf.image.shape == psf_dims
