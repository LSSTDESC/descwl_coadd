import numpy as np

from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_coadd.coadd import make_coadd_obs


def test_cosmics():

    rng = np.random.RandomState(32)

    coadd_dim = 101
    buff = 5
    psf_dim = 51
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="grid",
    )
    psf = make_fixed_psf(psf_type='gauss')

    data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        epochs_per_band=10,  # so we get a CR
        cosmic_rays=True,
    )
    coadd, exp_info = make_coadd_obs(
        exps=data['band_data']['i'],
        coadd_wcs=data['coadd_wcs'],
        coadd_bbox=data['coadd_bbox'],
        psf_dims=data['psf_dims'],
        rng=rng,
        remove_poisson=False,  # no object poisson noise in sims
    )

    assert any(exp_info['maskfrac'] > 0)

    cosmic_flag = coadd.coadd_exp.mask.getPlaneBitMask('CR')

    bmask = coadd.coadd_exp.mask.array
    noise_bmask = coadd.coadd_noise_exp.mask.array

    w = np.where((bmask & cosmic_flag) != 0)
    assert w[0].size != 0

    w = np.where((noise_bmask & cosmic_flag) != 0)
    assert w[0].size != 0
