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


def _make_sim(rng, psf_type, epochs_per_band=3):
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
        epochs_per_band=epochs_per_band,
        g1=0.02,
        g2=0.00,
        bands=['i', 'z'],
        psf=psf,
        dither=True,
        rotate=True,
    )


@pytest.mark.parametrize('psf_type', ['gauss', 'ps'])
def test_coadd_sim_smoke(psf_type):
    rng = np.random.RandomState(8312)
    data = _make_sim(rng, psf_type)

    extent = data['coadd_bbox'].getDimensions()
    coadd_dims = (extent.getX(), extent.getY())
    assert data['coadd_dims'] == coadd_dims

    psf_dims = data['psf_dims']

    # coadding individual bands as well as over bands
    coadds = MultiBandCoaddsDM(
        data=data['band_data'],
        coadd_wcs=data['coadd_wcs'],
        coadd_bbox=data['coadd_bbox'],
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
        coadd_bbox=data['coadd_bbox'],
        psf_dims=psf_dims,
        byband=False,
    )

    for band in coadds.bands:
        assert band not in coadds.coadds


def test_coadd_sim_noise():
    """
    ensure the psf and noise var are set close, so
    that the relative weight is right
    """
    rng = np.random.RandomState(32)
    data = _make_sim(rng=rng, psf_type='gauss')

    psf_dims = data['psf_dims']

    # coadding individual bands as well as over bands
    coadds = MultiBandCoaddsDM(
        data=data['band_data'],
        coadd_wcs=data['coadd_wcs'],
        coadd_bbox=data['coadd_bbox'],
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

        bmask = nexp.mask.array
        w = np.where(bmask == 0)
        nvar = nexp.image.array[w].var()
        assert abs(nvar / emed - 1) < 0.02
