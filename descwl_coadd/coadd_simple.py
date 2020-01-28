"""
A container for coadd observations.
"""
import numpy as np
import ngmix
import galsim


class MultiBandCoaddsSimple(ngmix.Observation):
    """
    Coadd a simple set of perfectly aligned images with constant psf and
    non-varying wcs

    Parameters
    ----------
    data: list of observations
        For a single band.  Should have image, weight, noise, wcs attributes,
        as well as get_psf method.  For example see the simple sim from
        descwl_shear_testing
    """
    def __init__(self, *, data):
        self._data = data
        self._make_coadd()

    def _make_coadd(self):
        import ngmix

        ntot = 0
        wsum = 0.0
        for band in self._data:
            bdata = self._data[band]
            for epoch_ind, se_obs in enumerate(bdata):

                cen = (np.array(se_obs.image.array.shape)-1)/2
                y, x = cen

                wt = np.median(se_obs.weight.array)
                wsum += wt

                if ntot == 0:

                    image = se_obs.image.array.copy()*wt
                    noise = se_obs.noise.array.copy()*wt

                    weight = se_obs.weight.array.copy()

                    psf_image = se_obs.get_psf(x, y, center_psf=True).array
                    psf_err = psf_image.max()*0.0001
                    psf_weight = psf_image*0 + 1.0/psf_err**2
                    psf_cen = (np.array(psf_image.shape)-1.0)/2.0

                    pos = galsim.PositionD(x=x, y=y)
                    wjac = se_obs.wcs.jacobian(image_pos=pos)
                    wscale, wshear, wtheta, wflip = wjac.getDecomposition()
                    jac = ngmix.Jacobian(
                        x=x,
                        y=y,
                        dudx=wjac.dudx,
                        dudy=wjac.dudy,
                        dvdx=wjac.dvdx,
                        dvdy=wjac.dvdy,
                    )

                    psf_jac = ngmix.Jacobian(
                        x=psf_cen[1],
                        y=psf_cen[0],
                        dudx=wjac.dudx,
                        dudy=wjac.dudy,
                        dvdx=wjac.dvdx,
                        dvdy=wjac.dvdy,
                    )

                else:
                    image += se_obs.image.array[:, :]*wt
                    noise += se_obs.noise.array[:, :]*wt
                    weight[:, :] += se_obs.weight.array[:, :]

                ntot += 1

        image *= 1.0/wsum
        noise *= 1.0/wsum

        psf_obs = ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jac,
        )

        super().__init__(
            image=image,
            noise=noise,
            weight=weight,
            bmask=np.zeros(image.shape, dtype='i4'),
            ormask=np.zeros(image.shape, dtype='i4'),
            jacobian=jac,
            psf=psf_obs,
            store_pixels=False,
        )
