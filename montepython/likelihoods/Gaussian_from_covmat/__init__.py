from montepython.likelihood_class import Likelihood_prior
import numpy as np
import scipy.linalg as la

class Gaussian_from_covmat(Likelihood_prior):

    def __init__(self, path, data, command_line):
        # Unusual construction, since the data files are not distributed
        # alongside BK14 (size problems)
        super(Gaussian_from_covmat, self).__init__(path, data, command_line)

        self.covmat = np.loadtxt(data.path['root'] + '/' + self.covmat_file)
        self.bestfit = np.loadtxt(data.path['root'] + '/' + self.bestfit_file)

        with open(data.path['root'] + '/' + self.covmat_file) as fid:
            line = fid.readline()[1:]
            covmat_varnames = [varname.strip() for varname in line.split(',')]

        with open(data.path['root'] + '/' + self.bestfit_file) as fid:
            line = fid.readline()[1:]
            bestfit_varnames = [varname.strip() for varname in line.split(',')]

        # TEST CONSISTENCY BETWEEN NAMES AND SHAPES
        assert(self.bestfit.shape[0] == self.covmat.shape[0] == self.covmat.shape[1])
        assert(self.bestfit.shape[0] == len(covmat_varnames))
        assert(covmat_varnames == bestfit_varnames)

        # Remove derived parameters from likelihood
        self.varnames = [varname for varname in bestfit_varnames if varname not in self.derived_parameters]
        var_indices = [index for index, varname in enumerate(covmat_varnames) if varname in self.varnames]
        self.covmat = self.covmat[:, var_indices][var_indices, :]
        self.bestfit = self.bestfit[var_indices]
        self.covmat_inverse = la.inv(self.covmat)

    def loglkl(self, cosmo, data):

        q_minus_bestfit = -self.bestfit.copy()
        for index, varname in enumerate(self.varnames):
            q_minus_bestfit[index] += data.mcmc_parameters[varname]['current']*data.mcmc_parameters[varname]['scale']

        loglkl = -0.5*np.dot(q_minus_bestfit, np.dot(self.covmat_inverse, q_minus_bestfit))
        return loglkl
