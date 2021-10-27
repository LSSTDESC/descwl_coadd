def get_coadd_center(coadd_wcs, coadd_bbox):
    """
    get the pixel and sky center of the coadd within the bbox
    The pixel center is forced to be integer

    Parameters
    -----------
    coadd_wcs: DM wcs
        The wcs for the coadd
    coadd_bbox: geom.Box2I
        The bounding box for the coadd within larger wcs system

    Returns
    -------
    pixcen as Point2D, skycen as SpherePoint
    """
    from lsst.geom import Point2D, Point2I

    # force integer location
    pixcen = Point2I(coadd_bbox.getCenter())
    skycen = coadd_wcs.pixelToSky(Point2D(pixcen))

    return pixcen, skycen
