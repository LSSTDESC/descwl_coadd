import sys
import numpy as np

from descwl_shear_sims.sim import make_sim, get_se_dim, get_coadd_center_gs_pos
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.galaxies import make_galaxy_catalog
from descwl_shear_sims.layout import Layout
from descwl_coadd.coadd import make_coadd, get_coadd_psf_at_position
from descwl_coadd.coadd import get_coadd_psf_bbox
import lsst.geom as geom
import galsim
import pytest

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

psf_dim = 51
coadd_dim = 101
pixel_scale = 0.2


class CustomGalaxyCatalog(object):
    """
    Catalog that uses an explicit list of galsim objects and (u, v) shifts.

    Parameters
    ----------
    gal_list : list[galsim.GSObject]
        The galaxies to place.
    uv_shift : list[tuple[float, float]]
        Per-galaxy (u, v) shifts in arcsec, same length as gal_list.
    layout : Layout | None
        Optional; only used to carry coadd bbox/world origin metadata.
        Positions are taken strictly from `uv_shift`.
    """
    def __init__(self, *, gal_list, uv_shift, coadd_dim, pixel_scale,
                 simple_coadd_bbox, layout=None):
        if gal_list is None or len(gal_list) == 0:
            raise ValueError("gal_list must be a non-empty list of galsim objects")
        if uv_shift is None or len(uv_shift) != len(gal_list):
            raise ValueError("uv_shift must be provided and match len(gal_list)")
        self.gal_type = 'fixed'
        self._gal_list = list(gal_list)
        self._shifts = [galsim.PositionD(u, v) for (u, v) in uv_shift]

        buff = 0  # ignored since positions are explicit

        if isinstance(layout, str):
            self.layout = Layout(layout, coadd_dim, buff, pixel_scale,
                                 simple_coadd_bbox=simple_coadd_bbox)
        else:
            assert isinstance(layout, Layout)
            self.layout = layout

    def __len__(self):
        return len(self._gal_list)

    def get_objlist(self, *, survey):
        """
        Returns a dict with the same structure as other catalogs.
        The provided GSObjects are used verbatim (no flux remapping).
        """
        indexes = list(range(len(self._gal_list)))
        return {
            "objlist": list(self._gal_list),
            "shifts": list(self._shifts),
            "redshifts": None,
            "indexes": indexes,
        }


def _make_sim(
    rng,
    psf_type,
    epochs_per_band=3,
    stars=False,
    dither=True,
    rotate=True,
    bands=["i", "z"],
    coadd_dim=101,
    se_dim=None,
    psf_dim=51,
    bad_columns=False,
    psf_variation_factor=None,
    u_shift=0,
    v_shift=0,
):
    buff = 5

    gal = galsim.DeltaFunction(flux=1.0)
    galaxy_catalog = CustomGalaxyCatalog(
        gal_list=[gal],
        uv_shift=[(u_shift, v_shift)],
        coadd_dim=coadd_dim,
        pixel_scale=pixel_scale,
        simple_coadd_bbox=True,
        layout="no_layout",
    )

    if se_dim is None:
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate)

    if psf_type == "ps":
        psf = make_ps_psf(rng=rng, dim=se_dim,
                          variation_factor=psf_variation_factor)
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
        g1=0.00,
        g2=0.00,
        bands=bands,
        psf=psf,
        dither=dither,
        rotate=rotate,
        bad_columns=bad_columns,
        draw_noise=False,
    )


def get_coadd_res(u_shift, v_shift):
    rng = np.random.RandomState(2025)
    bands = ['i']

    sim_data = _make_sim(
        rng=rng,
        psf_type='gauss',
        bands=bands,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        dither=False,
        rotate=False,
        u_shift=u_shift,
        v_shift=v_shift,
    )

    exps = sim_data['band_data']['i']
    coadd_wcs = sim_data['coadd_wcs']
    coadd_bbox = sim_data['coadd_bbox']
    psf_dims = sim_data['psf_dims']

    coadd_dict = make_coadd(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=np.random.RandomState(7),
        remove_poisson=False,
    )

    # center psf
    cen_psf_img = coadd_dict['coadd_psf_exp'].image.array
    cen_psf_img /= cen_psf_img.sum()

    assert cen_psf_img.shape == (psf_dim, psf_dim)
    assert np.isfinite(cen_psf_img).all()
    np.testing.assert_allclose(cen_psf_img.sum(), 1.0, rtol=1e-6, atol=1e-8)

    coadd_wcs = sim_data['coadd_wcs']
    coadd_bbox = sim_data['coadd_bbox']

    coadd_bbox_cen_gs_skypos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    )

    world_pos = coadd_bbox_cen_gs_skypos.deproject(
        u_shift * galsim.arcsec,
        v_shift * galsim.arcsec,
    )

    dm_world_pos = geom.SpherePoint(world_pos.ra / galsim.degrees,
                                    world_pos.dec / galsim.degrees,
                                    geom.degrees)
    new_image_pos = coadd_wcs.skyToPixel(dm_world_pos)

    dm_image_pos = geom.Point2D(new_image_pos.x, new_image_pos.y)

    psf_off = get_coadd_psf_at_position(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        image_pos=dm_image_pos,
        rng=np.random.RandomState(13),
        remove_poisson=False,
    )

    off_img = psf_off.computeKernelImage(dm_image_pos).array
    assert off_img.shape == (psf_dim, psf_dim)
    assert np.isfinite(off_img).all()
    np.testing.assert_allclose(off_img.sum(), 1.0, rtol=1e-6, atol=1e-8)

    return sim_data, coadd_dict, cen_psf_img, off_img, exps


def get_crop_bbox(u_shift, v_shift, coadd_dim, psf_dim, pixel_scale):
    coadd_cen = (coadd_dim - 1) / 2

    cen_int = geom.Point2I(
        int(np.floor(coadd_cen + u_shift / pixel_scale + 0.5)),
        int(np.floor(coadd_cen + v_shift / pixel_scale + 0.5)))

    to_crop = get_coadd_psf_bbox(cen_int, psf_dim)

    return to_crop


np.random.seed(42)
random_shifts = [
    (np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0)) 
    for _ in range(3)
]

test_cases = [
    (0.0, 0.0),
    (10.0 * 0.2, 0.0),
    (10.5 * 0.2, 5.0 * 0.2)
] + random_shifts


@pytest.mark.parametrize("u_shift, v_shift", test_cases)
def test_coadd_off_cen(u_shift, v_shift):
    sim_data, coadd_dict, cen_psf_img, off_img, exps = get_coadd_res(u_shift, v_shift)

    crop_box = get_crop_bbox(
        u_shift=u_shift,
        v_shift=v_shift,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        pixel_scale=sim_data['coadd_wcs'].getPixelScale().asArcseconds(),
    )

    crop_image = coadd_dict["coadd_exp"][crop_box].image.array
    crop_image_norm = crop_image / crop_image.sum()

    if u_shift == 0.0 and v_shift == 0.0:
        # check if the center psf from the old method
        # and new method are the same
        np.testing.assert_allclose(off_img, cen_psf_img, atol=1e-6)

    # check if the off_center psf agree with the input delta object
    np.testing.assert_allclose(off_img, crop_image_norm, atol=1e-6)
