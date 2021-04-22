import pytest
import numpy as np

from descwl_shear_sims.sim import (
    make_dmsim,
    make_galaxy_catalog,
    StarCatalog,
    make_psf,
    make_ps_psf,
    get_se_dim,
)

from ..coadd import MultiBandCoaddsDM


def _make_sim(rng, psf_type):
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
        psf = make_psf(psf_type=psf_type)

    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=100,
    )

    return make_dmsim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        dither=True,
        rotate=True,
    )


# @pytest.mark.parametrize('psf_type', ['gauss', 'ps'])
@pytest.mark.parametrize('psf_type', ['gauss'])
def test_coadd_obs_smoke(psf_type):
    rng = np.random.RandomState(8312)
    data = _make_sim(rng, psf_type)

    coadd_dims = data['coadd_dims']
    psf_dims = data['psf_dims']

    # coadding individual bands as well as over bands

    coadds = MultiBandCoaddsDM(
        data=data['band_data'],
        coadd_wcs=data['coadd_wcs'],
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
    )

    coadd = coadds.get_coadd()
    assert coadd.coadd_exp.image.array.shape == coadd_dims
    assert coadd.image.shape == coadd_dims
    assert coadd.noise.shape == coadd_dims
    assert coadd.psf.image.shape == psf_dims

    for band in coadds.bands:
        bcoadd = coadds.get_coadd(band=band)
        assert bcoadd.coadd_exp.image.array.shape == coadd_dims
        assert bcoadd.image.shape == coadd_dims
        assert bcoadd.noise.shape == coadd_dims
        assert bcoadd.psf.image.shape == psf_dims

    # not coadding individual bands
    coadds = MultiBandCoaddsDM(
        data=data['band_data'],
        coadd_wcs=data['coadd_wcs'],
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
        byband=False,
    )

    for band in coadds.bands:
        assert band not in coadds.coadds


'''
def test_coadd_obs_weights():
    """
    ensure the psf and noise var are set close, so
    that the relative weight is right
    """
    rng = np.random.RandomState(8312)
    sim = SimpleSim(
        rng=rng,
        epochs_per_band=2,
    )
    data = sim.gen_sim()

    coadd_dims = (sim.coadd_dim,)*2

    # coadding individual bands as well as over bands
    psf_dim = sim.psf_dim
    psf_dims = (psf_dim,)*2

    coadds = MultiBandCoaddsDM(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=coadd_dims,
        psf_dims=psf_dims,
    )

    for exp, pexp, nexp in zip(coadds.exps,
                               coadds.psf_exps,
                               coadds.noise_exps):

        emed = np.median(exp.variance.array)
        pmed = np.median(pexp.variance.array)
        nmed = np.median(nexp.variance.array)

        assert abs(pmed/emed-1) < 1.0e-3
        assert abs(nmed/emed-1) < 1.0e-3
'''
