import pytest
import lsst.geom as geom

from descwl_coadd.coadd import get_coadd_psf_bbox


def get_coadd_psf_bbox_old(x, y, dim):
    """
    This old code only did the right thing for odd coadd dimensions

    compute the bounding box for the coadd, based on the coadd
    center as an integer position (Point2I) and the dimensions

    Parameters
    ----------
    x: int
        The integer center in x.  Should be gotten from bbox.getCenter() to
        provide an integer position
    y: int
        The integer center in x.  Should be gotten from bbox.getCenter() to
        provide an integer position
    dim: int
        The dimensions of the psf, must be odd

    Returns
    -------
    lsst.geom.Box2I
    """

    xpix = int(x)
    ypix = int(y)

    xmin = (xpix - (dim - 1)/2)
    ymin = (ypix - (dim - 1)/2)

    return geom.Box2I(
        geom.Point2I(xmin, ymin),
        geom.Point2I(xmin + dim-1, ymin + dim-1),
    )


@pytest.mark.parametrize('dim', (31, 32))
def test_coadd_psf_bbox_smoke(dim):
    psf_dim = 51

    bbox = geom.Box2I(geom.Point2I(54, 54), geom.Extent2I(dim, dim))
    cen = geom.Point2I(bbox.getCenter())

    psf_bbox = get_coadd_psf_bbox(cen=cen, dim=psf_dim)
    assert isinstance(psf_bbox, geom.Box2I)


@pytest.mark.parametrize('dim', (31, 32))
def test_coadd_psf_bbox_backcompat(dim):
    """
    make sure the new code agrees with the old code
    """
    psf_dim = 51

    bbox = geom.Box2I(geom.Point2I(54, 54), geom.Extent2I(dim, dim))
    cen = geom.Point2I(bbox.getCenter())

    bbox = get_coadd_psf_bbox(cen=cen, dim=psf_dim)
    bbox_old = get_coadd_psf_bbox_old(
        x=cen.x, y=cen.y, dim=psf_dim,
    )
    assert bbox == bbox_old
