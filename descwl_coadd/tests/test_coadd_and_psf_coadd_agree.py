import sys
import numpy as np
import galsim

from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.surveys import get_survey

from descwl_coadd.coadd import make_coadd_obs
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_centroid(image):
    rows, cols = np.mgrid[:image.shape[0], :image.shape[1]]

    imsum = image.sum()

    rowmean = (rows * image).sum() / imsum
    colmean = (cols * image).sum() / imsum
    return rowmean, colmean


class OneStarCatalog(object):
    def __init__(self, coadd_dim, mag=17):
        self.gal_type = 'fixed'
        self.mag = mag

    def __len__(self):
        return 1

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects], [shifts]
        """

        flux = survey.get_flux(self.mag)
        objlist = [galsim.Gaussian(fwhm=1.0e-4, flux=flux)]

        shifts = [galsim.PositionD(0, 0)]

        return objlist, shifts


def _make_one_star_sim(
    rng, psf_type='gauss',
    epochs_per_band=3,
    dither=True,
    rotate=True,
    bands=['i'],
    coadd_dim=51,
    psf_dim=51,
):

    cat = OneStarCatalog(coadd_dim)

    psf = make_fixed_psf(psf_type=psf_type)

    data = make_sim(
        rng=rng,
        galaxy_catalog=cat,
        coadd_dim=coadd_dim,
        psf_dim=psf_dim,
        epochs_per_band=epochs_per_band,
        g1=0.00,
        g2=0.00,
        bands=bands,
        psf=psf,
        dither=dither,
        rotate=rotate,
    )
    return data, cat


def compare_images(coadd_image, psf_image_rescaled):
    import matplotlib.pyplot as mplt
    fig, axs = mplt.subplots(nrows=2, ncols=2)
    axs[1, 1].axis('off')

    p00 = axs[0, 0].imshow(coadd_image)
    fig.colorbar(p00, ax=axs[0, 0])
    axs[0, 0].set_title('coadded image')

    p01 = axs[0, 1].imshow(psf_image_rescaled)
    fig.colorbar(p01, ax=axs[0, 1])
    axs[0, 1].set_title('coadded psf image')

    p10 = axs[1, 0].imshow(coadd_image - psf_image_rescaled)
    fig.colorbar(p10, ax=axs[1, 0])
    axs[1, 0].set_title('coadd - psf')

    mplt.show()


def test_coadd_psf(show=False):
    """
    test that a coadded high s/n star and the coadded psf are consistent to
    high precision.  Also check that both are well centered on the odd dims psf
    of coadd/psf codad

    This is a crucial test of the conventions used for the wcs, bounding
    boxes, and the dithers
    """
    rng = np.random.RandomState(995)

    ntrial = 10

    for itrial in range(ntrial):
        sim_data, cat = _make_one_star_sim(rng=rng)

        bdata = sim_data['band_data']
        for band, exps in bdata.items():
            exps = bdata[band]

            survey = get_survey(gal_type='fixed', band=band)
            star_flux = survey.get_flux(cat.mag)

            coadd, exp_info = make_coadd_obs(
                exps=exps,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=False,  # no object poisson noise in sims
            )

            assert coadd.image.shape == coadd.psf.image.shape

            star_row, star_col = get_centroid(coadd.image)
            psf_row, psf_col = get_centroid(coadd.psf.image)

            cen = (coadd.image.shape[0] - 1)//2
            tol = 0.015
            assert abs(star_row - cen) < tol
            assert abs(star_col - cen) < tol
            assert abs(psf_row - cen) < tol
            assert abs(psf_col - cen) < tol

            print(f'star cen: {star_row:g} {star_col:g}')
            print(f'psf cen:  {psf_row:g} {psf_col:g}')

            psf_image_rescaled = (
                coadd.psf.image * star_flux / coadd.psf.image.sum()
            )

            abs_diff_im = np.abs(coadd.image - psf_image_rescaled)
            max_diff = abs_diff_im.max()
            im_max = coadd.image.max()
            assert max_diff/im_max < 0.0005

            if show:
                compare_images(
                    coadd_image=coadd.image,
                    psf_image_rescaled=psf_image_rescaled,
                )


if __name__ == '__main__':
    test_coadd_psf(show=True)
