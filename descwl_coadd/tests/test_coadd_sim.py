import os
import pytest
import numpy as np

from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from ..online_coadds import make_online_coadd_obs


def _make_sim(rng, psf_type, epochs_per_band=3, stars=False, dither=True, rotate=True):
    seed = 431
    rng = np.random.RandomState(seed)
    buff = 5

    coadd_dim = 101
    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="grid",
    )

    if psf_type == "ps":
        se_dim = get_se_dim(coadd_dim=coadd_dim)
        psf = make_ps_psf(rng=rng, dim=se_dim)
    else:
        psf = make_fixed_psf(psf_type=psf_type)

    if stars:
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=100,
        )
    else:
        star_catalog = None

    return make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        epochs_per_band=epochs_per_band,
        g1=0.02,
        g2=0.00,
        bands=['i', 'z'],
        psf=psf,
        dither=dither,
        rotate=dither,
    )


def test_coadd_sim_psgauss_smoke():
    psf_type = 'gauss'
    rng = np.random.RandomState(8312)
    data = _make_sim(rng, psf_type)

    extent = data['coadd_bbox'].getDimensions()
    coadd_dims = (extent.getX(), extent.getY())
    assert data['coadd_dims'] == coadd_dims

    psf_dims = data['psf_dims']

    for band, exps in data['band_data'].items():
        coadd = make_online_coadd_obs(
            exps=exps,
            coadd_wcs=data['coadd_wcs'],
            coadd_bbox=data['coadd_bbox'],
            psf_dims=psf_dims,
            rng=rng,
            remove_poisson=False,
        )

        assert coadd.image.shape == coadd_dims
        assert coadd.noise.shape == coadd_dims
        assert coadd.psf.image.shape == psf_dims


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present'
)
def test_coadd_sim_pspsf_smoke():
    psf_type = 'ps'
    rng = np.random.RandomState(8312)
    data = _make_sim(rng, psf_type, stars=True)

    extent = data['coadd_bbox'].getDimensions()
    coadd_dims = (extent.getX(), extent.getY())
    assert data['coadd_dims'] == coadd_dims

    psf_dims = data['psf_dims']

    for band, exps in data['band_data'].items():
        coadd = make_online_coadd_obs(
            exps=exps,
            coadd_wcs=data['coadd_wcs'],
            coadd_bbox=data['coadd_bbox'],
            psf_dims=psf_dims,
            rng=rng,
            remove_poisson=False,
        )

        assert coadd.image.shape == coadd_dims
        assert coadd.noise.shape == coadd_dims
        assert coadd.psf.image.shape == psf_dims


def test_coadd_sim_noise():
    """
    ensure the psf and noise var are set close, so
    that the relative weight is right
    """
    rng = np.random.RandomState(325)

    # turn off dither/rotate for testing noise
    data = _make_sim(
        rng=rng, psf_type='gauss',
        dither=False, rotate=False,
    )

    psf_dims = data['psf_dims']

    for band, exps in data['band_data'].items():
        coadd = make_online_coadd_obs(
            exps=exps,
            coadd_wcs=data['coadd_wcs'],
            coadd_bbox=data['coadd_bbox'],
            psf_dims=psf_dims,
            rng=rng,
            remove_poisson=False,
        )

        emed = np.min(coadd.coadd_exp.variance.array)
        pmed = np.min(coadd.coadd_psf_exp.variance.array)
        nmed = np.min(coadd.coadd_noise_exp.variance.array)

        assert abs(pmed/emed-1) < 1.0e-3
        assert abs(nmed/emed-1) < 1.0e-3

        bmask = coadd.coadd_noise_exp.mask.array
        w = np.where(bmask == 0)
        nvar = coadd.coadd_noise_exp.image.array[w].var()
        assert abs(nvar / emed - 1) < 0.02
