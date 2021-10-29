class WarpBoundaryError(Exception):
    """
    Part of the warp boundary was in the coadd region
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)
