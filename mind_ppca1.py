# Functions for construction of PPCA models for use in the mind algorithm

# Inclusions
import numpy as np
import numba as nb
import scipy

class ppcaModel:
    """
    ppcaModel is an object containing a full ppca model for a dataset
    inputs to ppcaModel are the data, a method and parameter for determining the number of signal dimensions, and optional weights for performing weighted PCA 
    to save memory, ppcaModels don't save the original dataset
    in the future, this may include a hash of the original dataset for comparison
    """
    def __init__(self, data, method='minVariance', dimPrm=0.95, weights=None):
        # input argument checks are performed in ppca
        # add parameters to the ppcaModel object
        self.method = method
        self.dimPrm = dimPrm
        self.weights = weights

        self.v,self.w,self.nv,self.u,self.cv,self.icv,self.logDet,self.logLikelihood = ppcaFull(data, method, dimPrm, weights)
        self.q = len(self.w)
        self.exp = np.sum(self.w) / (np.sum(self.w) + self.nv*(data.shape[1]-self.q))
        
    def processData(self, data):
        # take data and convert it to scores using the learned PPCA model
        assert data.shape[1]==self.v.shape[0], "data must have same dimensions of PPCA model"
        drData = (data - self.u) @ self.v
        return drData

def ppcaFull(data, method='minVariance', dimPrm=0.95, weights=None):
    v,w,nv,u = ppca(data, method, dimPrm, weights)
    cv,icv,logDet,logLikelihood = ppcaValues(data-u, v, w, nv)
    return v,w,nv,u,cv,icv,logDet,logLikelihood
    
# Primary PPCA Function
def ppca(data, method='minVariance', dimPrm=0.95, weights=None):
    # Probabilistic PPCA
    # reference: Tipping, M. E. & Bishop, C. M. Probabilistic principal component analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 61, 611–622 (1999).
    # implements weighted, probabilistic PPCA using either a minimum variance criterion or a number of components criterion
    # if method='minVariance', then PPCA returns as many eigenvector/eigenvalue pairs are required to explain (100*dimPrm)% of the variance
    # if method='numComponents', then PPCA returns dimPrm eigenvector/eigenvalue pairs
    # data is an (observations x dimensions) array
    assert data.ndim==2, "ppca must receive a matrix as data"
    N,D = data.shape # N=numObservations, D=numDimensions
    assert (method=='minVariance') or (method=='numComponents'), "Method must be set to either 'minVariance' or 'numComponents'"
    if method=='minVariance': assert (dimPrm>0) and (dimPrm<=1), "if using minVariance method, dimPrm must be in the set (0,1]"
    if method=='numComponents': assert (dimPrm>0) and (dimPrm<=D) and (isinstance(dimPrm,int) or np.issubdtype(dimPrm,np.integer)), \
        "if using numcomponents method, dimPrm must be a positive integer less than or equal to the number of dimensions in data"
    if (method=='minVariance') and (dimPrm==1):
        method='numComponents'
        dimPrm = D
    
    # Handle optional weight input
    if weights is None:
        useEig = False # allow method to be selected by shape of data
        weights = np.ones(N) # simplify code (1s vector same as not using weights)
    else:
        useEig = True # If weights provided, then we have to do eigendecomposition
        assert weights.ndim == 1, "Weights must be a 1D vector"
        assert len(weights) == N, "Weights must have the same number of elements as the number of rows of data (# of observations)"
    
    # Get MLE of mean and center data
    udata = np.average(data,axis=0,weights=weights) # MLE of mean
    cdata = data - udata # use centered data for computations

    # Pick method based on computational speed
    if useEig or (N > D):
        # do eigendecomposition
        covData = np.cov(cdata.T, bias=True, aweights=weights)
        w,v = scipy.linalg.eigh(covData)
        w[w<=0]=0 # don't allow false zeros or negatives (covariance matrices are symmetric positive semidefinite)
        idx = np.argsort(-w) # return index of descending sort
        w = w[idx] # sort eigenvalues from biggest to smallest
        v = v[:,idx] # sort eigenvectors from biggest to smallest

    else:
        # do svd instead
        _,s,v = np.linalg.svd(cdata)
        v = v.T
        w = s**2 / N # convert singular values to eigenvalues
    
    # Select dimensions of ppca model
    if method=='minVariance':
        varExplained = np.cumsum(w / np.sum(w))
        q = int(np.where(varExplained >= dimPrm)[0][0])+1
    elif method=='numComponents':
        q = dimPrm
    else: raise ValueError("assertions on method didn't work at the top of function!")
    
    # Return first q eigenvectors, eigenvalues, mle of noise variance, and mle of weighted mean
    if q<D: nv = np.mean(w[q:])
    else: nv = 0
    v = v[:,:q]
    w = w[:q]
        
    return v,w,nv,udata


# Supporting PPCA function for computing covariance, inverse covariance, log-determinant, and log-likelihood
def ppcaValues(cdata, v, w, nv):
    # Compute associated values for PPCA model (see def ppca() for information)
    assert cdata.ndim==2, "centered data must be a matrix"
    N,D = cdata.shape # num observations, num dimensions
    q = len(w) # number of signal dimensions in PPCA model
    assert v.shape[0]==D, "Eigenvectors must have same dimension as centered data"
    assert v.shape[1]==q, "Must have same number of eigenvectors and eigenvalues"
    assert nv <= np.min(w), "noise variance must be less than or equal to lowest eigenvalue (this is a sanity check, if thrown, something weird and unexpected happened"
    cv = nv * np.identity(D) + (v @ (np.diag(w) - nv*np.identity(q)) @ v.T)
    icv = np.linalg.inv(cv)
    logDet = 2*np.log(np.prod(np.diag(np.linalg.cholesky(cv))))
    if logDet>0.9999*np.finfo(cv.dtype).max:
        # Log switch to handle large values
        logDet = 2*np.sum(np.log(np.diag(np.linalg.cholesky(cv))))
    logLikelihood = -N*D/2*np.log(2*np.pi) - N/2*logDet - np.sum((cdata @ icv) * cdata)/2
        
    return cv, icv, logDet, logLikelihood
    



