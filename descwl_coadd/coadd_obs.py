import numpy as np
from numba import njit
import ngmix
import logging
import lsst.geom as geom
from lsst.geom import Point2D
from . import vis
from .util import get_coadd_center

LOG = logging.getLogger('CoaddObs')


class CoaddObs(ngmix.Observation):
    """
    Class representing a coadd observation

    Note that this class is a subclass of an `ngmix.Observation` and so it has
    all of the usual methods and attributes.

    Parameters
    ----------
    coadd_exp : afw_image.ExposureF
        The coadd exposure
    noise_exp : afw_image.ExposureF
        The coadded noise exposure
    coadd_psf_exp : afw_image.ExposureF
        The psf coadd
    coadd_mfrac_exp : afw_image.ExposureF
        The masked frraction image.
    """
    def __init__(
        self, *,
        coadd_exp,
        coadd_noise_exp,
        coadd_psf_exp,
        coadd_mfrac_exp,
    ):

        self.coadd_exp = coadd_exp
        self.coadd_psf_exp = coadd_psf_exp
        self.coadd_noise_exp = coadd_noise_exp
        self.coadd_mfrac_exp = coadd_mfrac_exp

        self.coadd_cen_integer, _ = get_coadd_center(
            coadd_wcs=coadd_exp.getWcs(),
            coadd_bbox=coadd_exp.getBBox(),
        )
        self.coadd_cen = Point2D(self.coadd_cen_integer)

        self._finish_init()

    def show(self):
        """
        show the output coadd in DS9
        """
        LOG.info('showing coadd in ds9')
        vis.show_image_and_mask(self.coadd_exp)

        vis.show_image(self.coadd_psf_exp.image.array, title='psf')

    def _get_jac(self, *, cenx, ceny):
        """
        get jacobian at the coadd image center

        make an ngmix jacobian with specified center specified (this is not the
        position used to evaluate the jacobian)

        Parameters
        ----------
        cenx: float
            Center for the output ngmix jacobian (not place of evaluation)
        ceny: float
            Center for the output ngmix jacobian (not place of evaluation)
        """

        coadd_wcs = self.coadd_exp.getWcs()

        dm_jac = coadd_wcs.linearizePixelToSky(self.coadd_cen, geom.arcseconds)
        matrix = dm_jac.getLinear().getMatrix()

        # note convention differences
        return ngmix.Jacobian(
            x=cenx,
            y=ceny,
            dudx=matrix[1, 1],
            dudy=-matrix[1, 0],
            dvdx=matrix[0, 1],
            dvdy=-matrix[0, 0],
        )

    def _get_coadd_psf_obs(self):
        """
        get the psf observation
        """

        psf_obj = self.coadd_exp.getPsf()
        psf_image = psf_obj.computeKernelImage(self.coadd_cen).array

        psf_cen = (np.array(psf_image.shape)-1.0)/2.0

        psf_jac = self._get_jac(cenx=psf_cen[1], ceny=psf_cen[0])

        psf_err = psf_image.max()*0.0001
        psf_weight = psf_image*0 + 1.0/psf_err**2

        return ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jac,
        )

    def _finish_init(self):
        """
        finish the init by sending the image etc. to the
        Observation init
        """
        psf_obs = self._get_coadd_psf_obs()  # noqa

        image = self.coadd_exp.image.array
        noise = self.coadd_noise_exp.image.array
        mfrac = self.coadd_mfrac_exp.image.array

        var = self.coadd_exp.variance.array.copy()
        wnf = np.where(~np.isfinite(var))

        if wnf[0].size == image.size:
            raise ValueError('no good variance values')

        if wnf[0].size > 0:
            var[wnf] = -1

        weight = var.copy()
        weight[:, :] = 0.0

        w = np.where(var > 0)
        weight[w] = 1.0/var[w]

        # TODO
        # gonna leave this commented since I don't remember why
        # I did it
        # if wnf[0].size > 0:
        #     # medval = np.sqrt(np.median(var[w]))
        #     # weight[wbad] = medval
        #     # TODO: add noise instead based on medval, need to send in rng
        #     image[wnf] = 0.0
        #     noise[wnf] = 0.0

        cen = (np.array(image.shape)-1)/2
        jac = self._get_jac(cenx=cen[1], ceny=cen[0])

        ormask = self.coadd_exp.mask.array

        super().__init__(
            image=image,
            noise=noise,
            weight=weight,
            bmask=ormask * 0,
            ormask=ormask,
            jacobian=jac,
            psf=psf_obs,
            store_pixels=False,
            mfrac=mfrac,
        )

        self.meta['mask_frac'] = np.mean(mfrac)


@njit
def get_masked_frac(mask, flags):
    nrows, ncols = mask.shape

    npixels = mask.size
    nmasked = 0

    for row in range(nrows):
        for col in range(ncols):
            if mask[row, col] & flags != 0:
                nmasked += 1

    return nmasked/npixels
