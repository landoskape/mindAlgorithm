import numpy as np
import scipy
from tqdm.notebook import tqdm, trange

class mind:
    """
    This is a master object that performs the full mind algorithm on a dataset. It contains all relevant functions (I'll probably break it down into subclasses in the future...)
    """
    def __init__(self,data,userOptions={}):
        """
        Initializer for mind object. 
        Provide data (numNeurons x numTimepoints)
        Various options can be set using the userOptions dictionary. The defaults are indicated in this function
        """
        if data.ndim != 2:
            raise ValueError("data must be a matrix")
        
        # Default options for entire MIND algorithm
        options = {
            # PCA Dimensionality Reduction Parameters
            "doPcaReduction":True,
            "scoreVariance":0.95,
                        
            # PPCA Forest Parameters
            "nEnsemble":100, 
            "nDir":2, 
            "nLeaf":40, 
            "nQuant":10,
            "minVariance":0.95,
            }
        
        if userOptions.items() > options.items():
                raise ValueError("userOptions contains key not found in object options")
        
        options.update(userOptions)
        
        # Load everything into MIND object
        self.data = data
        
        # PCA Dim Red Parameters
        self.doPcaReduction = options["doPcaReduction"]
        self.scoreVariance = options["scoreVariance"]
        
        # PPCA Forest Parameters
        self.nEnsemble = options["nEnsemble"]
        self.nDir = options["nDir"]
        self.nLeaf = options["nLeaf"]
        self.nQuant = options["nQuant"]
        self.minVariance = options["minVariance"]
        
    def pcaReduce(self):
        likelihood,uML,covML,nvML,w,v = ppca(self.data.T, minVariance=self.scoreVariance)
        self.likelihood = likelihood
        self.uML = uML
        self.covML = covML
        self.nvML = nvML
        self.w = w
        self.v = v
        self.scores = self.v.T @ self.data


# Import functions to be used by MIND algorithm
def ppca(data, minVariance=0.95):
    # probabilistic ppca - using description in Methods Section 1.7 of the following: https://www.biorxiv.org/content/10.1101/418939v2.full.pdf
    # data is a (observations x dimensions) array 
    # minVariance defines minimum fraction of variance required for fitting the PPCA model
    
    if data.ndim != 2:
        raise ValueError("data must be a matrix")
    
    # Return ML estimate of mean
    uML = np.mean(data,axis=0)
    cdata = data - uML # use centered data for computations
    
    # Pick method based on computational speed (logic inherited from Low/Lewallen, haven't tested yet!)
    N,D = data.shape
    if N > D:
        # do eigendecomposition
        covData = np.cov(cdata.T, bias=True)
        w,v = scipy.linalg.eigh(covData)
        w[w<=np.finfo(float).eps]=np.finfo(float).eps # don't allow weird tiny numbers (or negatives, it's a symmetric positive semidefinite matrix)
        idx = np.argsort(-w) # return index of descending sort
        w = w[idx] # sort eigenvalues
        v = v[:,idx] # sort eigenvectors
        s = np.sqrt(N*w) # singular values
        
    else:
        # do svd instead
        _,s,v = np.linalg.svd(cdata)
        v = v.T
        w = s**2 / N # eigenvalues
    
    varExplained = np.cumsum(w / np.sum(w))
    q = int(np.where(varExplained >= minVariance)[0][0])
    
    # Return ML estimate of noise variance
    nvML = np.mean(w[q:]) 
    
    # Keep q eigenvalues & eigenvectors
    w = w[:q]
    v = v[:,:q]
    
    # Compute ML estimate of covariance
    covML = nvML*np.identity(D) + (v @ (np.diag(w) - nvML*np.identity(q)) @ v.T)
    invCovML = np.linalg.inv(covML)
    
    # Return likelihood
    smartLogDet = 2*np.log(np.prod(np.diag(np.linalg.cholesky(covML))))
    likelihood = -N*D/2*np.log(2*np.pi) - N/2*smartLogDet - (1/2)*np.sum(np.array([cdata[n,:] @ invCovML @ cdata[n,:].T for n in range(N)]))

    return likelihood,uML,covML,nvML,w,v