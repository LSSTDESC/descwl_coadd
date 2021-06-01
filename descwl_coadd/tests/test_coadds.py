import os
import pytest
import numpy as np

from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog

from ..online_coadds import make_online_coadd_obs
from descwl_shear_sims.galaxies import make_galaxy_catalog


def _make_sim(
    rng, psf_type,
    epochs_per_band=3,
    stars=False,
    dither=True,
    rotate=True,
    bands=['i', 'z'],
    coadd_dim=101,
    psf_dim=51,
):
    buff = 5

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="grid",
        gal_config={'mag': 22},
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
        psf_dim=psf_dim,
        epochs_per_band=epochs_per_band,
        g1=0.02,
        g2=0.00,
        bands=bands,
        psf=psf,
        dither=dither,
        rotate=dither,
    )


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_online_coadds_smoke(dither, rotate):
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        dither=dither, rotate=rotate,
    )

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        coadd = make_online_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        coadd_dims = (coadd_dim, )*2
        psf_dims = (psf_dim, )*2
        assert coadd.image.shape == coadd_dims
        assert coadd.psf.image.shape == psf_dims


def test_coadd_sim_psgauss_smoke():
    psf_type = 'gauss'
    rng = np.random.RandomState(8312)
    data = _make_sim(rng=rng, psf_type=psf_type)

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
    data = _make_sim(rng=rng, psf_type=psf_type, stars=True)

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


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_online_coadds_noise(dither, rotate):
    rng = np.random.RandomState(55)

    # noise test currently only works with odd due to pixel offsets
    # used for psf coadding.  Same goes for dithering
    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        dither=dither, rotate=rotate,
    )

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        coadd = make_online_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        coadd_dims = (coadd_dim, )*2
        psf_dims = (psf_dim, )*2
        assert coadd.image.shape == coadd_dims
        assert coadd.psf.image.shape == psf_dims

        if not dither:
            emed = np.median(coadd.coadd_exp.variance.array)
            pmed = np.median(coadd.coadd_psf_exp.variance.array)
            nmed = np.median(coadd.coadd_noise_exp.variance.array)

            assert abs(pmed/emed-1) < 1.0e-3
            assert abs(nmed/emed-1) < 1.0e-3
