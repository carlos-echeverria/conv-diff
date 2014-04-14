import dolfin
from dolfin import dx, inner, nabla_grad, sqrt, near, dot
import krypy
import numpy
from matplotlib import pyplot

dolfin.parameters.linear_algebra_backend = "uBLAS"


def mat_dolfin2sparse(A):
    rows, cols, values = A.data()
    from scipy.sparse import csr_matrix
    return csr_matrix((values, cols, rows))


# element stabilization term for SUPG (streamline diffusion)
class Stabilization(dolfin.Expression):
    def __init__(self, mesh, wind, epsilon):
        self.mesh = mesh
        self.wind = wind
        self.epsilon = epsilon

    def eval_cell(self, values, x, ufc_cell):
        wk, hk = self.get_wind(ufc_cell)

        # element Peclet number
        Pk = wk*hk/(2*self.epsilon)

        if Pk > 1:
            values[:] = hk/(2*wk)*(1-1/Pk)
        else:
            values[:] = 0.

    def get_wind(self, ufc_cell):
        '''|w_k| and h_k as in ElmSW05'''
        cell = dolfin.Cell(self.mesh, ufc_cell.index)

        # compute centroid
        dim = self.mesh.topology().dim()
        centroid = numpy.zeros(dim)
        vertices = cell.get_vertex_coordinates()
        for i in range(dim):
            centroid[i] = numpy.sum(vertices[i::dim])
        centroid /= (vertices.shape[0]/dim)

        # evaluate wind and its norm |w_k|
        wind = numpy.array([self.wind[i](centroid) for i in range(dim)])
        wk = numpy.linalg.norm(wind, 2)

        # compute element length in direction of wind
        # TODO: this has to be tweaked for general wind vectors
        hk = cell.diameter()

        return wk, hk


def get_conv_diff_ls():
    # mesh and function space
    mesh = dolfin.RectangleMesh(-1, -1, 1, 1, 25, 25, 'crossed')
    V = dolfin.FunctionSpace(mesh, 'Lagrange', 1)

    wind = dolfin.Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'))

    # right hand side
    f = dolfin.Constant(0.)

    # diffusivity
    epsilon = 1./200

    # convection field
    delta = Stabilization(mesh, wind, epsilon)

    # define boundary conditions
    class Boundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class BoundaryRight(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1.)
    boundaries = dolfin.FacetFunction('size_t', mesh)
    boundary = Boundary()
    boundary.mark(boundaries, 1)
    boundary2 = BoundaryRight()
    boundary2.mark(boundaries, 2)
    boundary
    bcs = [dolfin.DirichletBC(V, dolfin.Constant(0.), boundaries, 1),
           dolfin.DirichletBC(V, dolfin.Constant(1.), boundaries, 2)
           ]

    # variational formulation
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    def get_conv_diff(u, v, epsilon, wind, stabilize=True):
        a = (
            epsilon*inner(nabla_grad(u), nabla_grad(v))*dx
            + inner(nabla_grad(u), wind)*v*dx
            )
        L = f*v*dx
        if stabilize:
            a += delta*inner(wind, nabla_grad(u))*inner(wind, nabla_grad(v))*dx
            L += delta*f*inner(wind, nabla_grad(v))*dx
        return a, L

    a, L = get_conv_diff(u, v, epsilon, wind)

    A = dolfin.assemble(a)
    b = dolfin.assemble(L)
    u0 = dolfin.Function(V).vector()
    u0.zero()

    [bc.apply(A, b) for bc in bcs]
    [bc.apply(u0) for bc in bcs]

    import scipy.sparse
    A_mat = scipy.sparse.csc_matrix(mat_dolfin2sparse(A))

    return A_mat, b.array(), u0.array()
