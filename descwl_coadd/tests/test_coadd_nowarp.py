import os
import pytest
import numpy as np

from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog

from descwl_coadd.coadd_nowarp import make_coadd_obs_nowarp
from descwl_shear_sims.galaxies import make_galaxy_catalog


def _make_sim(
    rng, psf_type,
    epochs_per_band=3,
    stars=False,
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

    se_dim = coadd_dim

    if psf_type == "ps":
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
        se_dim=se_dim,
        psf_dim=psf_dim,
        epochs_per_band=epochs_per_band,
        g1=0.02,
        g2=0.00,
        bands=bands,
        psf=psf,
        bad_columns=bad_columns,
    )


def test_coadd_nowarp_smoke():
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    bands = ['i']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
    )

    exp = sim_data['band_data'][bands[0]][0]

    coadd, exp_info = make_coadd_obs_nowarp(
        exp=exp,
        psf_dims=sim_data['psf_dims'],
        rng=rng,
        remove_poisson=False,  # no object poisson noise in sims
    )

    coadd_dims = (coadd_dim, )*2
    psf_dims = (psf_dim, )*2
    assert coadd.image.shape == coadd_dims
    assert coadd.psf.image.shape == psf_dims

    assert np.all(coadd.image == exp.image.array)
    assert np.all(coadd.coadd_exp.image.array == exp.image.array)

    assert np.all(np.isfinite(coadd.psf.image))
    assert np.all(coadd.mfrac == 0)


def test_coadds_mfrac():
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    bands = ['i']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        bad_columns=True,
    )

    exp = sim_data['band_data'][bands[0]][0]

    coadd, exp_info = make_coadd_obs_nowarp(
        exp=exp,
        psf_dims=sim_data['psf_dims'],
        rng=rng,
        remove_poisson=False,  # no object poisson noise in sims
    )

    assert any(exp_info['maskfrac'] > 0)

    assert not np.all(coadd.mfrac == 0)
    assert np.max(coadd.mfrac) > 0.1
    assert np.mean(coadd.mfrac) < 0.05
    assert np.all(coadd.mfrac >= 0)
    assert np.all(coadd.mfrac <= 1)


def test_coadds_noise():
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    cen = (coadd_dim - 1)//2
    pcen = (psf_dim - 1)//2

    bands = ['i']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
    )

    exp = sim_data['band_data'][bands[0]][0]

    coadd, exp_info = make_coadd_obs_nowarp(
        exp=exp,
        psf_dims=sim_data['psf_dims'],
        rng=rng,
        remove_poisson=False,  # no object poisson noise in sims
    )

    emed = coadd.coadd_exp.variance.array[cen, cen]
    pmed = coadd.coadd_psf_exp.variance.array[pcen, pcen]
    nmed = coadd.coadd_noise_exp.variance.array[cen, cen]

    assert abs(pmed/emed-1) < 1.0e-3
    assert abs(nmed/emed-1) < 1.0e-3


@pytest.mark.skipif('CATSIM_DIR' not in os.environ,
                    reason='CATSIM_DIR not in os.environ')
def test_coadds_set():
    """
    run trials with stars and make sure we get some SAT in the coadd mask
    """
    rng = np.random.RandomState(33)

    coadd_dim = 101
    psf_dim = 51
    band = 'i'
    epochs_per_band = 1

    ntrial = 100

    somesat = False
    for i in range(ntrial):
        sim_data = _make_sim(
            rng=rng, psf_type='gauss', bands=[band],
            epochs_per_band=epochs_per_band,
            coadd_dim=coadd_dim, psf_dim=psf_dim,
            stars=True,
        )

        exp = sim_data['band_data'][band][0]

        coadd, exp_info = make_coadd_obs_nowarp(
            exp=exp,
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        mask = coadd.coadd_exp.mask
        satflag = mask.getPlaneBitMask('SAT')

        wsat = np.where(mask.array & satflag != 0)

        if wsat[0].size > 0:
            somesat = True
            break

    print('i:', i)
    assert somesat
