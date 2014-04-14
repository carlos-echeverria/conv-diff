import conv_diff
import krypy
from matplotlib import pyplot

def get_gmres_roots(H):
    n_, n = H.shape

    from scipy.linalg import eig
    theta, _ = eig(H[:n, :].T.conj(), H.T.conj().dot(H))

    # TODO: check for zero thetas
    return 1./theta


def main():
    A, b, x0 = conv_diff.get_conv_diff_ls(10)

    ls = krypy.linsys.LinearSystem(A, b)
    solver = krypy.deflation.DeflatedGmres(ls, x0=x0, store_arnoldi=True)

    H = solver.H
    n_, n = H.shape
    for i in range(n, n+1):
        roots = get_gmres_roots(H[:i+1, :i])
        pyplot.plot(roots.real, roots.imag, '.')
        pyplot.show()

    pyplot.semilogy(solver.resnorms)
    pyplot.show()


if __name__ == '__main__':
    main()
