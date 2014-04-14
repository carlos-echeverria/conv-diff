import conv_diff
import numpy
import krypy
from matplotlib import pyplot, animation

def get_gmres_roots(H):
    n_, n = H.shape

    from scipy.linalg import eig
    theta, _ = eig(H[:n, :].T.conj(), H.T.conj().dot(H))

    # TODO: check for zero thetas
    return 1./theta


def main():
    A, b, x0 = conv_diff.get_conv_diff_ls(10)

    ls = krypy.linsys.LinearSystem(A, b, Ml=None)
    solver = krypy.deflation.DeflatedGmres(ls, x0=x0, store_arnoldi=True)

    H = solver.H
    n_, n = H.shape

    fig, ax = pyplot.subplots()
    line, = ax.plot([], [], '.')

    all_roots = numpy.concatenate([
        get_gmres_roots(H[:i+1, :i]) for i in range(1, n+1)
        ])
    ax.set_xlim(numpy.min(all_roots.real), numpy.max(all_roots.real))
    ax.set_ylim(numpy.min(all_roots.imag), numpy.max(all_roots.imag))
    ax.set_xscale('log')

    def plot_zeros(i):
        print(i)
        roots = get_gmres_roots(H[:i+1, :i])
        line.set_data(roots.real, roots.imag)
        return line,

    ani = animation.FuncAnimation(fig, plot_zeros, numpy.arange(1, n),
            interval=25)
    pyplot.show()



if __name__ == '__main__':
    main()
