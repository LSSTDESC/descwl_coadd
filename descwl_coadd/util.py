def get_coadd_center(coadd_wcs, coadd_bbox):
    """
    get the pixel and sky center of the coadd within the bbox

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

    # this returns integer positions
    pixcen = coadd_bbox.getCenter()
    skycen = coadd_wcs.pixelToSky(pixcen)

    return pixcen, skycen
