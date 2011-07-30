"""Matrices and related entities that can be used in expressions. """

from sympy.core import S, Expr, Symbol, Integer, Add, Mul, Function, Tuple
from sympy.matrices import Matrix

class MatrixExpr(Expr):
    """Matrix that can be used in expressions. """

    is_commutative = False

    def __new__(cls, *args):
        matrix = None

        if len(args) == 1:
            arg = args[0]

            if isinstance(arg, Matrix):
                matrix = arg
            elif isinstance(arg, MatrixExpr):
                return arg

        if matrix is None:
            matrix = Matrix(*args)

        obj = Expr.__new__(cls)
        obj.rows = matrix.rows
        obj.cols = matrix.cols
        obj.mat = tuple(matrix.mat)

        return obj

    @property
    def args(self):
        return (Integer(self.rows), Integer(self.cols), Tuple(*self.mat))

    def _hashable_content(self):
        return self.args

    def as_matrix(self):
        return Matrix.new(self.rows, self.cols, self.mat)

    def __getitem__(self, key):
        result = self.as_matrix().__getitem__(key)

        if isinstance(result, Matrix):
            return result.as_expr()
        else:
            return result

    def transpose(self):
        return self.as_matrix().transpose().as_expr()

    def det(self, method=None):
        return self.as_matrix().det(method=method)

    @property
    def is_square(self):
        return self.rows == self.cols

    @property
    def is_symmetric(self):
        return self.as_matrix().is_symmetric()

class Transpose(Function):
    """Symbolic representation of transposition of a matrix. """

    @classmethod
    def eval(cls, expr):
        r"""
        Evaluation rules for ``Transpose(expr)``.

        1. `(A^T)^T \Rightarrow A`
        2. `A^T \Rightarrow A` where `A` is symmetric
        3. `(c \cdot A)^T \Rightarrow c \cdot A^T`

        """
        if isinstance(expr, Transpose):
            return expr.args[0]
        elif isinstance(expr, MatrixExpr):
            if expr.is_symmetric:
                return expr.args[0]
        elif isinstance(expr, Mul):
            coeff, expr = expr.as_independent(MatrixExpr)

            if coeff is not S.One:
                return coeff*Transpose(expr)

    def _eval_doit(self, **hints):
        expr = self.args[0]

        if isinstance(expr, MatrixExpr):
            return expr.transpose()
        else:
            return self

class Determinant(Function):
    """Symbolic representation of determinant of a matrix. """

    @classmethod
    def eval(cls, expr):
        r"""
        Evaluation rules for ``Determinant(expr)``.

        1. `\det A^T \Rightarrow \det A`
        2. `\det A_{1 \times 1} \Rightarrow a_{1,1}

        """
        if isinstance(expr, Transpose):
            return Determinant(expr.args[0])
        elif isinstance(expr, MatrixExpr):
            if expr.is_square and expr.rows == 1:
                return expr[0, 0]

    def _eval_doit(self, **hints):
        expr = self.args[0]

        if isinstance(expr, MatrixExpr):
            return expr.det()
        else:
            return self

    def _eval_do_laplace_expand(self, orientation='row', index=0):
        """One step computation of determinant of a matrix using Laplace expansion. """
        expr = self.args[0]

        if not isinstance(expr, MatrixExpr) and expr.is_square:
            return None

        matrix = expr.as_matrix()
        n = expr.rows

        def aij_cofactor(i, j):
            minor = matrix.minorMatrix(i, j).as_expr()
            cofactor = matrix[i, j]*Determinant(minor)

            if (i + j) % 2 != 0:
                cofactor = -cofactor

            return cofactor

        result = []

        if orientation == 'rows':
            for i in xrange(0, n):
                result.append(aij_cofactor(i, index))
        elif orientation == 'cols':
            for j in xrange(0, n):
                result.append(aij_cofactor(index, j))
        else:
            raise ValueError("unrecognized value for orientation: %s" % orientation)

        return Add(*result)

def laplace_expand(expr, orientation='rows', index=0):
    r"""
    One step computation of determinant of a matrix using Laplace expansion.

    This function use Laplace expansion formula:

    1. ``orientation='rows'``

       .. math:: \det A = \sum_{i=1}^{n} (-1)^{i + j} a_{i,j} M_{i,j}

    2. ``orientation='cols'``

       .. math:: \det A = \sum_{j=1}^{n} (-1)^{i + j} a_{i,j} M_{i,j}

    `M_{i,j}` stands for determinant of the `(n-1) \times (n-1)` minor matrix,
    obtained by removing `i`-th row and `j`-th column from `A`.

    **Parameters**

    orientation : 'rows' | 'cols' (optional, default: 'rows')
        The orientation in which a matrix is expanded.
    index : int (optional, default: 0)
        Which row or column use as the basis for expansion.

    **Examples**

    >>> from sympy.matrices import Matrix, Determinant, laplace_expand
    >>> from sympy.printing import pprint

    >>> det0 = Determinant(Matrix([[-2, 2, -3], [-1, 1, 3], [2, 0, -1]]))
    >>> pprint(det0, use_unicode=False)
       /[-2  2  -3]\
       |[         ]|
    det|[-1  1  3 ]|
       |[         ]|
       \[2   0  -1]/

    >>> det1= laplace_expand(det0, orientation='rows', index=0)
    >>> pprint(det1, use_unicode=False)
           /[1  3 ]\      /[2  -3]\        /[2  -3]\
    - 2*det|[     ]| + det|[     ]| + 2*det|[     ]|
           \[0  -1]/      \[0  -1]/        \[1  3 ]/

    >>> det2 = laplace_expand(det1, orientation='rows', index=0)
    >>> pprint(det2, use_unicode=False)
    18
    >>> det0.doit()
    18

    """
    return expr.do('laplace_expand', dict(orientation=orientation, index=index))
