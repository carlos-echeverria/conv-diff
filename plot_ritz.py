import conv_diff
import numpy
import krypy
from matplotlib import pyplot, animation, rc

# use latex in matplotlib
rc('text', usetex=True)

def get_gmres_roots(H):
    n_, n = H.shape

    from scipy.linalg import eig
    theta, _ = eig(H[:n, :].T.conj(), H.T.conj().dot(H))

    # TODO: check for zero thetas
    return 1./theta


def main():
    A, b, x0 = conv_diff.get_conv_diff_ls(10)
    N = A.shape[0]

    from scipy.sparse import spdiags
    D = spdiags(1./A.diagonal(), [0], N, N)

    ls = krypy.linsys.LinearSystem(A, b, Ml=D)
    solver = krypy.deflation.DeflatedGmres(ls, x0=x0, tol=1e-10, store_arnoldi=True)

    H = solver.H
    n_, n = H.shape

    fig, axs = pyplot.subplots(ncols=2, figsize=(1920./200,1080./200), dpi=100)
    fig.subplots_adjust(wspace=0.3)
    line_res, = axs[0].plot([], [])
    line_roots, = axs[1].plot([], [], '.')

    axs[0].set_yscale('log')
    axs[0].set_xlim(0, n)
    axs[0].set_ylim(numpy.min(solver.resnorms), numpy.max(solver.resnorms))
    axs[0].set_title('GMRES residual norm')
    axs[0].set_xlabel('GMRES iteration $i$')
    axs[0].set_ylabel(r'$\frac{\|r_i\|}{\|b\|}$')

    all_roots = numpy.concatenate([
        get_gmres_roots(H[:i+1, :i]) for i in range(1, n+1)
        ])
    axs[1].set_xlim(0.95*numpy.min(all_roots.real), 1.05*numpy.max(all_roots.real))
    axs[1].set_ylim(1.05*numpy.min(all_roots.imag), 1.05*numpy.max(all_roots.imag))
    axs[1].set_xscale('log')
    axs[1].set_title('roots of GMRES polynomial')
    axs[1].set_xlabel('Real part')
    axs[1].set_ylabel('Imaginary part')

    def animate(i):
        print(i)
        line_res.set_data(list(range(0, i+1)), solver.resnorms[:i+1])
        roots = get_gmres_roots(H[:i+1, :i])
        line_roots.set_data(roots.real, roots.imag)
        #return line,

    ani = animation.FuncAnimation(fig, animate, numpy.arange(1, n),
            interval=100)

    #pyplot.show()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, bitrate=4000)
    ani.save('gmres.mp4', writer=writer, dpi=200)



if __name__ == '__main__':
    main()
