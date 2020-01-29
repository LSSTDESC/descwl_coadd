"""
A container for coadd observations.
"""
import numpy as np
import ngmix
import galsim


class MultiBandCoaddsNoStack(ngmix.Observation):
    """
    Rolling our own instead of using the stack

    Parameters
    ----------
    data: list of observations
        For a single band.  Should have image, weight, noise, wcs attributes,
        as well as get_psf method.  For example see the simple sim from
        descwl_shear_testing
    coadd_wcs:
        galsim wcs
    coadd_dims: sequence
        Dimensions of coadd
    wcs_position_offset : float
        The offset to get from pixel-centered, zero-indexed coordinates to
        the coordinates expected by the WCS.  Default 0.0
    """
    def __init__(self, *,
                 data,
                 coadd_wcs,
                 coadd_dims,
                 wcs_position_offset=0.0):
        self.data = data
        self.coadd_wcs = coadd_wcs
        self.coadd_dims = coadd_dims
        self.wcs_position_offset = wcs_position_offset

        self._set_coadd_pixels()
        self._make_coadd()

    def _get_coadd_grid(self):
        ygrid, xgrid = np.mgrid[
            0:self.coadd_dims[0],
            0:self.coadd_dims[1],
        ]
        return ygrid.ravel(), xgrid.ravel()

    def _set_coadd_pixels(self):
        ygrid, xgrid = self._get_coadd_grid()

        # TODO: deal with color
        self._coadd_ra, self._coadd_dec = self.coadd_wcs.xyToradec(
            xgrid + self.wcs_position_offset,
            ygrid + self.wcs_position_offset,
            units=galsim.degrees,
            color=0,
        )

    def _get_coadd_pixels_in_wcs(self, wcs):

        # TODO: deal with color
        x, y = wcs.radecToxy(
            self._coadd_ra,
            self._coadd_dec,
            galsim.degrees,
            color=0,
        )

        x -= self.wcs_position_offset,
        y -= self.wcs_position_offset,
        return y, x

    def _make_coadd(self):
        import ngmix

        ntot = 0
        wsum = 0.0
        for band in self.data:
            bdata = self.data[band]
            for epoch_ind, se_obs in enumerate(bdata):

                wcs = se_obs.wcs

                rows, cols = self._get_coadd_pixels_in_wcs(wcs)
                # rows, cols = self._get_coadd_pixels_in_wcs(self.coadd_wcs)
                print(rows[1000:1050])
                print(cols[1000:1050])
                stop
                """
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
                """

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
