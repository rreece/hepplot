import numpy as np


class DataGenerator:

    def __init__(self, funcs, mixtures):
        n = sum(mixtures)
        mixtures = [ m / n for m in mixtures ]
        self.funcs = funcs
        self.mixtures = mixtures
        assert np.allclose(sum(self.mixtures), 1.0)
        self.random_edges = list()
        _e = 0
        for m in mixtures:
            _e += m
            self.random_edges.append(_e)

    def get_n(self):
        return len(self.mixtures)

    def get_pdf(self, x, i=None):
        if i is not None:
            func = self.funcs[i]
            pdf = func.pdf(x)
        else:
            # define pdfs and normalize them over linspace x
            pdfs = [ func.pdf(x) for func in self.funcs ]
            sums = [ sum(a) for a in pdfs ]
            pdfs = [ a / n for a, n in zip(pdfs, sums) ]
            pdf_tot = sum([ a * m for a, m in zip(pdfs, self.mixtures) ])
            for a in pdfs:
                assert np.allclose(sum(a), 1.0),\
                    'sum(a) = %.f' % sum(a)
            assert np.allclose(sum(pdf_tot), 1.0),\
                    'sum(pdf_tot) = %.f' % sum(pdf_tot)
            pdf = pdf_tot
        return pdf

    def generate(self, size, i=None):
        data = list()
        if i is None and self.get_n() == 1:
            i = 0
        for _ in range(size):
            func = None
            if i is not None:
                func = self.funcs[i]
            else:
                _r = np.random.random()
                for _e, _f in zip(self.random_edges, self.funcs):
                    if _r < _e:
                        func = _f
                        break
            assert func
            _d = func.rvs()
            data.append(_d)
        return data
