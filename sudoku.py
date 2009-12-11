"""
Copyright (c) 2009, Ben Moran
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Ben Moran nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


-- This is a re-implementation of the "Linear systems Sudoku solver" from http://www.it.uu.se/katalog/praba420/Sudoku.pdf
"""
 
from math import sqrt
from cvxmod import problem, minimize, optvar, ones, speye, diag
from cvxmod.atoms import norm1

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
[   1.000    0.000    0.000    1.000    0.000    0.000]
[   0.000    1.000    0.000    0.000    1.000    0.000]
[   0.000    0.000    1.000    0.000    0.000    1.000]
<BLANKLINE>
"""
    r1, c1 = m1.size
    r2, c2 = m2.size
    if  r1 != r2:
        raise TypeError('Heights don''t match, %d and %d' % (r1,r2))
    return matrix(list(m1)+list(m2), (r1,c1+c2), 'd')

def concatvert(m1,m2):
    """
Concatenate two matrices vertically.

>>> print concatvert(eye(3),eye(3))
[   1.000    0.000    0.000]
[   0.000    1.000    0.000]
[   0.000    0.000    1.000]
[   1.000    0.000    0.000]
[   0.000    1.000    0.000]
[   0.000    0.000    1.000]
<BLANKLINE>
"""
    r1, c1 = m1.size
    r2, c2 = m2.size
    if  c1 != c2:
        raise TypeError('Widths don''t match, %d and %d' % (c1, c2))
    
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

def sample_tricky():
    """
    Return a "tricky" Problem instance from the paper.
    """
    return Problem(
        "..3..9.81"
        "...2...6."
        "5...1.7.."
        "89......."
        "..56.12.."
        ".......37"
        "..9..2..8"
        ".7...4..."
        "25.8..6..")


def sample_moderate():
    """
    Return a "moderate" Problem instance from the paper.
    """
    return Problem(
        "..5...7.."
        "93.5.4..."
        "84.....3."
        "6...2.4.."
        "5...9...8"
        "..9.8...1"
        ".5.....7."
        "...3.7.86"
        "..1...9..")

class Problem:
    """
    Translates between different representations of a Sudoku problem.
    """
    def __str__(self):
        return unicode(self)
    
    def __unicode__(self):
        s = ""
        boxsize = self.get_box_size()
        if boxsize:
            inserts = range(self.N - boxsize, 0, -boxsize)
        for i in xrange(0,self.N):
            row = self.entries[i*self.N:(i+1)*self.N]
            row = [unicode(e) if e else '_' for e in row]
            if boxsize:
                for ix in inserts:
                    row.insert(ix, "|")
            s += " ".join(row) + "\n"
            if boxsize and (i+1) in inserts:
                s += "+-".join(["--" * boxsize] * boxsize) + "\n"
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
        if self.get_box_size() and (box_digits or test_all):
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

    def get_box_size(self):
        """
        Return the size of sub-boxes, sqrt(N), if applicable.
        """        
        boxsize = int(sqrt(self.N))
        if boxsize > 1 and self.N == boxsize**2:            
            return boxsize
        return 0

    def get_box_defs(self):
        """
        Return the numbers of cells in the NxN grid
        corresponding to non-overlapping squares of
        size sqrt(N)xsqrt(N)
        """
        N = self.N
        boxsize = self.get_box_size()
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

    def solve(self, solvefunc=None, **kwargs):
        """
        Return a new Problem instance with
        best attempt at solution, using plain L1.
        """
        M = self.matrix()
        ones = ones_v(M.size[0])
        if solvefunc is None:
            #solvefunc = solve_iter_reweighted_l1
            #solvefunc = solve_plain_l1
            solvefunc = solve_plain_l1_cvxmod
            solvefunc = solve_rw_l1_cvxmod
        #v = solve_plain_l1(M,ones)
        v = solvefunc(M,ones)
        result = Problem.from_indicator_vector(v)
        if not all([e==1 for e in result.get_result()]):
            print "Failed"
        else:
            print "OK"
        return result

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


def solve_iter_reweighted_l1(A, y, solver='glpk', iters=4):
    """
    Find x with min l1 such that Ax=y,
    using iteratively reweighted l1 minimization
    """
    #Reweighted l1 approach from Candes Wakin and Boyd Enhancing sparsity by reweighted l1 minimization. J. Fourier Anal. Appl., 14 877-905. and http://www.acm.caltech.edu/~emmanuel/papers/rwl1.pdf:

    ## from http://sites.google.com/site/stephanegchretien/alternatingl1
    n = A.size[1]
    
    c0 = ones_v(2*n)
    
    G1 = concathoriz(A,-A)
    G2 = concathoriz(-A,A)
    G3 = -eye(2*n)
    G = reduce(concatvert, [G1,G2,G3])
    hh = reduce(concatvert, [y, -y, zeros_v(2*n)])

    sol = cvxopt.solvers.lp(c0, G, hh, solver=solver)
    
    doublexlone = sol['x'][:2*n] #.trans()[0]
    xlone = concathoriz(eye(n),-eye(n)) * doublexlone

    xtmp = doublexlone
    for l in range(iters):
        #c1u = (abs(doublexlone)+.1)**-1
        c1u = (abs(xtmp)+.1)**-1 # should it be this?
        sol = cvxopt.solvers.lp(c1u, G, hh)
        solstixdt =  sol['x'][:2*n] #.trans()[0]
        xlone = concathoriz(eye(n),-eye(n)) * solstixdt
        xtmp = solstixdt        

    v = sol['x'][:n]
    
    return v

def solve_plain_l1_cvxmod(A, y):
    x = optvar('x', A.size[1])
    p = problem(minimize(norm1(x)), [A*x == y])
    p.solve(quiet=True, solver='glpk')
    return x.value

def solve_rw_l1_cvxmod(A, y, iters=6):
    W = speye(A.size[1])
    x = optvar('x', A.size[1])
    epsilon = 0.5
    for i in range(iters):
        last_x = matrix(x.value) if x.value else None
        p = problem(minimize(norm1(W*x)), [A*x == y])
        p.solve(quiet=True, cvxoptsolver='glpk')
        ww = abs(x.value) + epsilon
        W = diag(matrix([1/w for w in ww]))
        if last_x:
            err = ( (last_x - x.value).T * (last_x - x.value) )[0]
            if err < 1e-4:
                break
    return x.value

# TODO: IRLS, Alternating L1
