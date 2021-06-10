import pytest
import numpy as np

from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog

from ..coadd import make_coadd_obs
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
    bad_columns=False,
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
        bad_columns=bad_columns,
    )


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_smoke(dither, rotate):
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

        coadd = make_coadd_obs(
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

        assert np.all(np.isfinite(coadd.psf.image))
        assert np.all(coadd.mfrac == 0)


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_mfrac(dither, rotate):
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        dither=dither, rotate=rotate,
        bad_columns=True,
    )

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        coadd = make_coadd_obs(
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

        assert np.all(np.isfinite(coadd.psf.image))
        assert not np.all(coadd.mfrac == 0)
        assert np.max(coadd.mfrac) > 0.1
        assert np.mean(coadd.mfrac) < 0.05

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(coadd.mfrac)
            import pdb
            pdb.set_trace()


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_noise(dither, rotate):
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    cen = (coadd_dim - 1)//2
    pcen = (psf_dim - 1)//2

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

        coadd = make_coadd_obs(
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
        assert coadd.noise.shape == coadd_dims
        assert coadd.psf.image.shape == psf_dims

        assert np.all(np.isfinite(coadd.psf.image))
        assert np.all(coadd.mfrac == 0)

        emed = coadd.coadd_exp.variance.array[cen, cen]
        pmed = coadd.coadd_psf_exp.variance.array[pcen, pcen]
        nmed = coadd.coadd_noise_exp.variance.array[cen, cen]

        assert abs(pmed/emed-1) < 1.0e-3
        assert abs(nmed/emed-1) < 1.0e-3
