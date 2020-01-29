def show_image(image):
    """
    show an image
    """
    import matplotlib.pyplot as plt
    plt.imshow(
        image,
        interpolation='nearest',
        cmap='gray',
    )
    plt.show()


def show_2images(im1, im2, title=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
    ax = axs[0]
    pos = ax.imshow(
        im1,
        interpolation='nearest',
        cmap='gray',
    )
    fig.colorbar(pos, ax=ax)
    ax.grid(False)

    ax = axs[1]
    pos = ax.imshow(
        im2,
        interpolation='nearest',
        cmap='gray',
    )
    fig.colorbar(pos, ax=ax)
    ax.grid(False)

    if title is not None:
        fig.suptitle(title)

    plt.show()


def show_images(imlist, title=None):
    import matplotlib.pyplot as plt

    grid = Grid(len(imlist))

    fig, axs = plt.subplots(
        nrows=grid.nrow,
        ncols=grid.ncol,
    )

    for i, im in enumerate(imlist):
        row, col = grid(i)
        ax = axs[row, col]
        pos = ax.imshow(
            im,
            interpolation='nearest',
            cmap='gray',
        )
        fig.colorbar(pos, ax=ax)
        ax.grid(False)

    if title is not None:
        fig.suptitle(title)

    plt.show()


class Grid(object):
    """
    represent plots in a grid.  The grid is chosen
    based on the number of plots

    example
    -------
    grid=Grid(n)

    for i in xrange(n):
        row,col = grid(i)

        # equivalently grid.get_rowcol(i)

        plot_table[row,col] = plot(...)
    """
    def __init__(self, nplot):
        self.set_grid(nplot)

    def set_grid(self, nplot):
        """
        set the grid given the number of plots
        """
        from math import sqrt

        self.nplot = nplot

        # first check some special cases
        if nplot == 8:
            self.nrow, self.ncol = 2, 4
        else:

            sq = int(sqrt(nplot))
            if nplot == sq*sq:
                self.nrow, self.ncol = sq, sq
            elif nplot <= sq*(sq+1):
                self.nrow, self.ncol = sq, sq+1
            else:
                self.nrow, self.ncol = sq+1, sq+1

        self.nplot_tot = self.nrow*self.ncol

    def get_rowcol(self, index):
        """
        get the grid position given the number of plots

        move along columns first

        parameters
        ----------
        index: int
            Index in the grid

        example
        -------
        nplot=7
        grid=Grid(nplot)
        arr=biggles.FramedArray(grid.nrow, grid.ncol)

        for i in xrange(nplot):
            row,col=grid.get_rowcol(nplot, i)
            arr[row,col].add( ... )
        """

        imax = self.nplot_tot-1
        if index > imax:
            raise ValueError("index too large %d > %d" % (index, imax))

        row = index//self.ncol
        col = index % self.ncol

        return row, col

    def __call__(self, index):
        return self.get_rowcol(index)
