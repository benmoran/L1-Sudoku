"""
Linear systems sudoku solver from 
http://www.it.uu.se/katalog/praba420/Sudoku.pdf
"""
 
from math import sqrt
import cvxopt
import cvxopt.solvers
matrix = cvxopt.base.matrix

def eye(N):
    """
    Return the identity matrix of size NxN.
    """
    return matrix([ [0]*i + [1] + [0]*(N-i-1) for i in range(N)], (N,N), 'd')

def ones_v(n):
    """
    Return the column vector of ones of length n.
    """
    return matrix(1, (n,1), 'd')

def zeros_v(n):
    """
    Return the column vector of zeroes of length n.
    """
    return matrix(0, (n,1), 'd')
    
def concathoriz(m1, m2):
    """
Concatenate two matrices horizontally.

>>> print concathoriz(eye(3),eye(3))
[ 1.00e+00  0.00e+00  0.00e+00  1.00e+00  0.00e+00  0.00e+00]
[ 0.00e+00  1.00e+00  0.00e+00  0.00e+00  1.00e+00  0.00e+00]
[ 0.00e+00  0.00e+00  1.00e+00  0.00e+00  0.00e+00  1.00e+00]
<BLANKLINE>
"""
    r1, c1 = m1.size
    r2, c2 = m2.size
    if  r1 != r2:
        raise TypeError('Heights don''t match, %d and %d' % (r1,r2))
    return matrix(list(m1)+list(m2), (r1,c1+c2), 'd')

def concatvert(m1,m2):
    """
Concatenate two matrices horizontally.

"""    
    return concathoriz(m1.trans(), m2.trans()).trans()

def sample():
    """
    Return a sample Problem instance.
    """
    return Problem(
        "...15..7."
        "1.6...82."
        "3..86..4."
        "9..4..567"
        "..47.83.."
        "732..6..4"
        ".4..81..9"
        ".17...2.8"
        ".5..37...")

def sample_hard():
    """
    Return a sample Problem instance.
    """
    return Problem(
        "4.....8.5"
        ".3......."
        "...7....."
        ".2.....6."
        "....8.4.."
        "....1...."
        "...6.3.7."
        "5..2....."
        "1.4......")

class Problem:
    """
    Translates between different representations of a Sudoku problem.
    """
    def __unicode__(self):
        s = ""
        for i in xrange(0,self.N):
            row = self.entries[i*self.N:(i+1)*self.N]
            row = [unicode(e) if e else '_' for e in row]
            s += " ".join(row) + "\n"
        return s

    def __init__(self, entries, N=9):
        ok = False
        try:
            iter(entries)
            if len(entries) == N**2:
                ok = True
        except TypeError:
            ok = False
        if not ok:
            msg = "Entries should be an iterable of length %d for N=%d" 
            raise ValueError(msg % (N**2, N))
            
        self.N = N
        if isinstance(entries, basestring):
            self.entries = [int(x) if x.isdigit() else None for x in entries]
        else:
            self.entries = entries

    def num_entries(self):
        """
        Return the number of completed entries.
        """
        return len([e for e in self.entries if e])

    def matrix(self,
               all_cells=False,
               row_digits=False,
               col_digits=False,
               box_digits=False,
               clues=False):
        """
        Build the problem matrix.
        """
        tests = (all_cells, row_digits, col_digits, box_digits, clues)
        test_all = not any(tests)
        Acols = self.N**3
        
        M = matrix(0, (0,Acols), 'd')
        if all_cells or test_all:
            M = concatvert(M, self.get_all_cells_matrix())
        if row_digits or test_all:
            M = concatvert(M, self.get_row_digits_matrix())
        if col_digits or test_all:
            M = concatvert(M, self.get_col_digits_matrix())
        if box_digits or test_all:
            M = concatvert(M, self.get_box_digits_matrix())
        if clues or test_all:
            M = concatvert(M, self.get_clues_matrix())

        return M
    
    def get_all_cells_matrix(self):
        """
        Return the matrix that checks each cell is filled.
        """
        M = matrix(0,(self.N**2,self.N**3), 'd')
        for i in xrange(self.N**2):
            M[i, self.N*i:self.N*(i+1)] = 1
        return M

    def get_row_digits_matrix(self):
        """
        Return the matrix that checks each row contains
        all digits.
        """
        N = self.N
        M = matrix(0, (N**2,N**3), 'd')
        eyes = reduce(concathoriz, [eye(N)] * N)

        # I I I 0 0 0 0 0 0
        # 0 0 0 I I I 0 0 0
        # 0 0 0 0 0 0 I I I
        
        for i in xrange(N):
            M[ N*i: N*i+N , (N**2)*i : (N**2)*(i+1) ] = eyes
        return M

    def get_col_digits_matrix(self):
        """
        Return the matrix that checks each column contains
        all digits.
        """
        N = self.N
        M = matrix(0, (N**2,N**3), 'd')

        # I 0 0 I 0 0 I 0 0
        # 0 I 0 0 I 0 0 I 0
        # 0 0 I 0 0 I 0 0 I
        
        return reduce(concathoriz, [eye(N**2)] * N)


    def get_box_digits_matrix(self):
        """
        Return the matrix that checks each sqrt(N) box contains
        all digits.
        """
        N = self.N
        M = matrix(0, (N**2,N**3), 'd')
        for ix, cells in enumerate(self.get_box_defs()):
            for cell in cells:
                M[ N*ix: N*ix+N , N*cell : N*(cell+1) ] = eye(N)
        return M

    def get_box_defs(self):
        """
        Return the numbers of cells in the NxN grid
        corresponding to non-overlapping squares of
        size sqrt(N)xsqrt(N)
        """
        N = self.N
        boxsize = int(sqrt(N))
        boxes = []
        for boxnum in xrange(N):
            bx = boxnum/boxsize
            by = boxnum % boxsize
            boxes.append([i+(bx*boxsize)+N*(j+by*boxsize)
                          for i in range(boxsize)
                          for j in range(boxsize)])
        return boxes
        
    def to_indicator_vector(self):
        """
        Build the indicator vector, each N entries
        indicates N possible values for each cell
        """
        Acols = self.N**3
        v = matrix(0, (Acols,1), 'd')
        for ix, e in enumerate(self.entries):
            if e:
                v[ix*self.N + e-1] = 1
        return v

    @classmethod
    def from_indicator_vector(cls, v):
        """
        Convert an indicator vector to a Problem
        instance.
        """
        N3 = len(v)
        N = int(round(pow(N3, 1/3.0)))
        if N**3 != N3:
            raise ValueError("Vector must be cube length but is %d" % N3)
        def getnum(l):
            try:
                return list(l).index(1)+1
            except ValueError:
                return None
        for i, e in enumerate(v):
            v[i] = int(round(e))
        entries = [getnum(v[N*e:N*(e+1)]) for e in range(0,N**2)]
        return Problem(entries,N)

    def get_clues_matrix(self):        
        """
        Get the matrix to enforce that the answer is consistent
        with the clues.
        """
        
        N = self.N
        M = matrix(0, (self.num_entries(),N**3), 'd')

        i = 0
        for ix, e in enumerate(self.entries):
            if e:
                M[i, ix*N + e -1] = 1
                i += 1
        return M

    def get_result(self, **kwargs):
        """
        Get the problem matrix and multiply it by the
        indicator vector.
        """
        M = self.matrix(**kwargs)
        return M * self.to_indicator_vector()


    def solve(self, **kwargs):
        """
        Return a new Problem instance with
        best attempt at solution, using plain L1.
        """
        M = self.matrix()
        ones = ones_v(M.size[0])
        v = solve_plain_l1(M,ones)
        return Problem.from_indicator_vector(v)


def solve_plain_l1(A, y, solver='glpk'):
    """
    Find x with min l1 such that Ax=y,
    using plain L1 minimization
    """
    n = A.size[1]
    
    c0 = ones_v(2*n)
    
    G1 = concathoriz(A,-A)
    G2 = concathoriz(-A,A)
    G3 = -eye(2*n)
    G = reduce(concatvert, [G1,G2,G3])
    hh = reduce(concatvert, [y, -y, zeros_v(2*n)])
    
    u = cvxopt.solvers.lp(c0, G, hh, solver=solver)
    
    v = u['x'][:n]
    
    return v

