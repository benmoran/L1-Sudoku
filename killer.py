from sudoku import Problem, matrix
from tests import all_ones

import unittest

def get_indices(source, target):
    """
    >>> list(get_indices('bbaba','a'))
    [2, 4]
    """
    pos = source.find(target)
    while pos > -1:
        yield pos
        pos = source.find(target, pos+1)

class KillerProblem(Problem):
    """Solve Killer Sudoku - sums of continous regions,
    rather than individual clues"""

    def __init__(self, regions, totals, N=9):
        ok = False
        try:
            iter(regions)
            if len(regions) == N**2:
                ok = True
        except TypeError:
            ok = False
        if not ok:
            msg = "Regions should be an iterable of length %d for N=%d" 
            raise ValueError(msg % (N**2, N))
        if set(regions) != set(totals.keys()):
            raise ValueError('Each region should have a total provided')

        self.regions = regions
        self.totals = totals
        # todo: check totals are ints?

        Problem.__init__(self, entries="."*(N**2), N=N)

    def get_clues_matrix(self):        
        """
        Get the matrix to enforce that the answer is consistent
        with the clues.
        """

        N = self.N
        M = matrix(0, (len(self.totals),N**3), 'd')

        for ix, (region, total) in enumerate(self.totals.items()):
            values = [n / float(total)  for n in range(1,N+1)]
            for pos in get_indices(self.regions, region):
                M[ix, pos*N:(pos+1)*N] = values
        return M
    
    

class KillerProblemTest(unittest.TestCase):
    """
    Tests for l1sudoku KillerProblem class
    """
    def setUp(self):
        """
        Instantiate a Killer problem (Times 1314, Monday 7th December, 2009)
        """
        totals = {"a":8,
                  "b":5,
                  "c":22,
                  "d":6,
                  "e":13,
                  "f":15,
                  "g":8,
                  "h":16,
                  "i":21,
                  "j":13,
                  "k":7,
                  "l":25,
                  "m":7,
                  "n":12,
                  "o":3,
                  "p":12,
                  "q":12,
                  "r":3,
                  "s":8,
                  "t":17,
                  "u":5,
                  "v":17,
                  "w":11,
                  "x":9,
                  "y":14,
                  "z":13,
                  "A":14,
                  "B":12,
                  "C":12,
                  "D":4,
                  "E":15,
                  "F":17,
                  "G":16,
                  "H":7,
                  "I":6 }

        self.N = 9
        self.killer = KillerProblem(regions=
                                    "aabccddee"
                                    "ffbgchhhh"
                                    "ijjgkklll"
                                    "iimmknnlo"
                                    "ppqrsstto"
                                    "upqrvvwwx"
                                    "uyyzzAABx"
                                    "CDEEEAABF"
                                    "CDGGHHIIF",
                                    totals=totals, N=9)
    def test_init_killer(self):
        try:
            KillerProblem(regions="abcd", totals={'a':1}, N=2)
            self.assertFalse(True, 'Should spot missing totals')
        except ValueError:
            pass
        print KillerProblem(regions="abcd",
                            totals={'a':1,'b':2,'c':3,'d':4},
                            N=2)
        
    def test_clues_matrix(self):
        kp = KillerProblem("aabb",{'a':2,'b':4},2)
        answer = Problem("1122", N=2)
        v = kp.get_clues_matrix() * answer.to_indicator_vector()
        print kp.get_clues_matrix()
        print v
        self.assertTrue(all_ones(v))
        
    def test_killer(self):
        self.killer.solve()
        
