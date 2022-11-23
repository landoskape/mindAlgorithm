# code to implement the mind algorithm from the following paper:
# paper: https://www.biorxiv.org/content/10.1101/418939v2.full.pdf

# Object Oriented Version
# -- here, I'm porting the mindAlgorithm1_Freeze code towards an object oriented version --

# inclusions
import numpy as np
import numba as nb
import scipy
import time
import mind_ppca1 as mppca
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.manifold import MDS as skMDS
from sklearn_extra.cluster import KMedoids
from tqdm.notebook import tqdm, trange
from ipywidgets import IntProgress
from IPython.display import display, clear_output

# top level model
class mindModel:
    
    def __init__(self, userOpts={}):
        # -- Construct options dictionary for algorithm --
        self.opts = {}
        # Options for data management
        self.opts['useScores'] = True # whether to use PPCA for dimreduction of data 
        self.opts['dr_method'] = 'minVariance' # which method to use for dimreduction (either 'minVariance' or 'numComponents')
        self.opts['dr_dimPrm'] = 0.95 # how much variance to preserve when reducing dimensionality
        # Options for random forest
        self.opts['nDir'] = 2 # how many directions to choose from at each split
        self.opts['nLeaf'] = 50 # minimum number of datapoints in each leaf 
        self.opts['nQuant'] = 8 # number of quantiles to choose threshold from
        self.opts['rf_method'] = 'minVariance' # which method to use for PPCA models in random forest (either 'minVariance' or 'numComponents')
        self.opts['rf_dimPrm'] = 0.95 # desired fraction of variance to explain with PPCA models (or number of components to keep)
        # Options for landmark points
        self.opts['numScafPoints'] = 2000 # number of landmark points to create
        self.opts['numScafSeeds'] = 1 # number of landmark seeds to initiate algorithm with
        self.opts['scaffoldAlgorithm'] = 'greedy' # algorithm (can be 'greedy' or 'kmedoids'), k-medoids takes far too long and is unnecessary
        # Options for MDS
        self.opts['mdsSammonIterations'] = 800 # until this is coded with conjugate gradient descent, just use # of landmark points...
        self.opts['mdsSammonAlpha'] = 0.5
        
        # Update default opts with user requests
        assert userOpts.keys() <= self.opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(self.opts.keys())}"
        self.opts.update(userOpts)
        
        # -- Initialize variables --
        # data variables
        self.data = [] 
        self.numObs = [] # number of observations
        self.numDims = [] # number of dimensions in data
        self.drActive = self.opts['useScores']
        self.drMethod = self.opts['dr_method']
        self.drDimPrm = self.opts['dr_dimPrm']
        self.drPPCA = []
        self.drDims = [] # num dims used for reduction 
        self.drData = [] # data after dimensionality reduction (scores of PCA)
        # forest variables
        self.numTrees = 0
        self.nDir = self.opts['nDir']
        self.nLeaf = self.opts['nLeaf']
        self.nQuant = self.opts['nQuant']
        self.rf_method = self.opts['rf_method']
        self.rf_dimPrm = self.opts['rf_dimPrm']
        self.forest = []
        self.forestSummarized = False
        self.treeSummary = []
        self.ppcaMeans = []
        self.ppcaCovs = []
        self.ppcaInvCovs = []
        self.ppcaLogDets = []
        # scaffold variables
        self.numScafPoints = self.opts['numScafPoints']
        self.numScafSeeds = self.opts['numScafSeeds']
        self.scaffoldAlgorithm = self.opts['scaffoldAlgorithm']
        self.scafIdx = []
        self.scafData = []
        self.scafGridProb = []
        self.scafTrProb = []
        self.scafLocalDist = []
        self.scafGlobalDist = [] # distance b/w landmark points before symmetrizing
        self.scafDist = [] # symmetric distance b/w landmark points        
        
        # MDS variables
        self.scafManifold = [] # coordinates of data in manifold space
        self.scafManifoldDim = [] # dimensions for each scafManifold
        self.mdsSammonIterations = self.opts['mdsSammonIterations']
        self.mdsSammonAlpha = self.opts['mdsSammonAlpha']
        
        # Forward/Reverse Mapping Variables
        self.mappingForwardK = []
        self.mappingForwardLambda = []
        self.mappingBackwardK = []
        self.mappingBackwardLambda = []
        self.mappingOptionsK = np.arange(2,21)
        self.mappingOptionsLambda = np.logspace(-6,1,30)
        self.mappingForwardOptDim = []
        self.mappingBackwardOptDim = []
        self.mappingForwardOptionsError = []
        self.mappingBackwardOptionsError = []
        
    def addData(self, data):
        self.data = data
        self.numObs,self.numDims = data.shape
        if self.drActive:
            self.drPPCA = mppca.ppcaModel(data, self.drMethod, self.drDimPrm)
            self.drData = (data - self.drPPCA.u) @ self.drPPCA.v
            self.drDims = self.drPPCA.q
        else:
            # Put full dataset and full #dims in drDims/drData to reduce verbosity
            self.drData = data
            self.drDims = self.numDims
    
    # -------------------------
    # -- Forest Construction --
    # -------------------------
    def constructForest(self,numTrees,simpleUpdates=False):
        cdata = self.drData[:-1,:]
        sdata = self.drData[1:,:]
        
        # Use simple update switch because tqdm doesn't work on my laptop
        if simpleUpdates: tstart = time.time()
            
        progressBar = tqdm(range(numTrees),disable=simpleUpdates)
        for tt in progressBar:
            if simpleUpdates: 
                eta = (time.time() - tstart) * numTrees/(tt+1)
                if tt==0: eta = np.nan
                print(f"Fitting PPCA Tree {tt+1}/{numTrees}, eta: {eta:.1f} seconds...")
                
            progressBar.set_description(f'Fitting PPCA Tree {tt+1}/{numTrees}')
            self.forest.append(self.splitForestNode(cdata,sdata))
            self.numTrees += 1
            if simpleUpdates and tt>0: eta=numTrees/(tt+1)*(time.time()-tstart)
            
    def addTrees(self,numTrees,simpleUpdates=False):
        cdata = self.drData[:-1,:]
        sdata = self.drData[1:,:]
        
        if simpleUpdates: tstart = time.time()
            
        progressBar = tqdm(range(numTrees),disable=simpleUpdates)
        for newTree in progressBar:
            if simpleUpdates:
                eta = (time.time() - tstart) * numTrees/(newTree+1)
                if newTree==0: eta = np.nan
                print(f"Fitting PPCA Tree {newTree+1}/{numTrees}, eta: {eta:.1f} seconds...")
            
            progressBar.set_description(f'Adding PPCA Tree {newTree+1}/{numTrees}')
            self.forest.append(self.splitForestNode(cdata,sdata))
            self.numTrees += 1

    def splitForestNode(self, cdata, sdata, rootNode=True, ppcaModel=None):
        # recursive function to construct a decision tree
        # use hyperparameters to generate candidate split directions and thresholds
        # choose decision plane by optimizing likelihood (using a mean-identity approximation)
        # construct left/right nodes and rerun splitForestNode on it
        # path2node is a list of ints describing decision path towards this particular node
        # --------- it starts as empty, and adds a 0 for every left choice and a 1 for every right choice
        
        # Define node dictionary to be returned
        node = {}

        # important variables for function
        N = cdata.shape[0] # number of observations
        D = self.drDims # number of dimensions of data
        
        if rootNode: assert N >= 2*self.nLeaf, "splitForestNode received a root node, but contains too few data points to split!"
        if N < 2*self.opts['nLeaf']:
            # If it has too few points to be split, it is a terminal node. 
            assert isinstance(ppcaModel,mppca.ppcaModel), "Terminal node identified (N < 2*nLeaf), but no PPCA model provided"
            # Add PPCA model, identify as terminal node, then exit recursive function
            node['terminalNode'] = True
            node['ppca'] = ppcaModel
            return node

        # If this node is being split, find the best split direction
        node['terminalNode'] = False
        hpNormal, hpThreshold, leftPPCA, rightPPCA, ldata, lsdata, rdata, rsdata = self.optimizeHyperplane(cdata, sdata)

        # Add hyperplane to node dictionary
        node['hpNormal'] = hpNormal
        node['hpThreshold'] = hpThreshold
        
        # Return left and right leaves of node (recursively create tree, this function will call itself until it returns a terminal node)
        node['left'] = self.splitForestNode(ldata, lsdata, rootNode=False, ppcaModel=leftPPCA)
        node['right'] = self.splitForestNode(rdata, rsdata, rootNode=False, ppcaModel=rightPPCA)
        return node

    def optimizeHyperplane(self, cdata, sdata):
        # choosing a hyperplane to create a decision boundary
        # we aim to model the successor states of our left and right leaf nodes with a multivariate gaussian.
        # outputs hyperplane separating data (as a dictionary) and left/right current and successor data
        assert cdata.ndim == 2, "cdata must be a matrix"
        assert sdata.ndim == 2, "sdata must be a matrix"
        assert cdata.shape == sdata.shape, "cdata and sdata must have same shape"
        N = cdata.shape[0] # number of observations
        D = self.drDims # number of dimensions
        assert N >= 2*self.nLeaf, "optimizeHyperplane must receive data with at least 2*nLeaf!"

        # Prepare splitting procedure (use quantile speedup trick for decision thresholds)
        quantPoints = np.linspace(0,1,self.nQuant+2)[1:-1]
        if (N < 4*self.nLeaf) and (0.5 not in quantPoints): 
            # make sure we include the median when down to small datasets
            quantPoints = np.concatenate((quantPoints,[0.5]))
        
        # Generate candidate hyperplane directions on unit hypersphere
        hypDirs = np.zeros((self.nDir,D)) # do it this way to avoid annoyance, it's very much not the bottleneck of the pipeline 
        while np.any(np.sum(hypDirs,axis=0) == 0): hypDirs = np.random.normal(0,1,(self.nDir,D))
        hypDirs = hypDirs / np.sqrt(np.sum(hypDirs**2,axis=1)).reshape(-1,1)
        assert np.allclose(np.linalg.norm(hypDirs,axis=1),1), "hypDirs don't all have norm==1!"

        # Preallocate variables
        hypThresholds = np.zeros(self.nDir)
        lCandidatePPCA = []
        rCandidatePPCA = []
        llCandidate = np.zeros((self.nDir,2))
        for ndir in range(self.nDir):
            cProjection = cdata @ hypDirs[ndir,:]
            cQuantiles = np.quantile(cProjection, quantPoints)
            ssError = []
            for cThreshold in cQuantiles:
                # Do isotropic gaussian model first
                idxLeft = np.where(cProjection <= cThreshold)[0]
                idxRight = np.where(cProjection > cThreshold)[0]
                # Instead of choosing quantiles intelligently, just skip this threshold if it produces too few datapoints in one of the children
                if len(idxLeft)<self.nLeaf or len(idxRight)<self.nLeaf:
                    ssError.append(np.inf)
                    continue
                cMeanLeft = np.mean(sdata[idxLeft,:],axis=0,keepdims=True)
                cMeanRight = np.mean(sdata[idxRight,:],axis=0,keepdims=True)
                cDevLeft = np.sum((sdata[idxLeft,:] - cMeanLeft)**2)
                cDevRight = np.sum((sdata[idxRight,:] - cMeanRight)**2)
                ssError.append(cDevLeft+cDevRight)

            # Then, for the best isotropic fit, compute a full ppca model
            idxBestThreshold = np.argmin(ssError)
            idxLeft = np.where(cProjection <= cQuantiles[idxBestThreshold])[0]
            idxRight = np.where(cProjection > cQuantiles[idxBestThreshold])[0]

            hypThresholds[ndir] = cQuantiles[idxBestThreshold]
            lCandidatePPCA.append(mppca.ppcaModel(sdata[idxLeft,:],self.rf_method,self.rf_dimPrm))
            rCandidatePPCA.append(mppca.ppcaModel(sdata[idxRight,:],self.rf_method,self.rf_dimPrm))
            llCandidate[ndir,0] = lCandidatePPCA[-1].logLikelihood
            llCandidate[ndir,1] = rCandidatePPCA[-1].logLikelihood

        # Find optimal direction, return indices for left and right data
        totalLikelihood = np.sum(llCandidate,axis=1)
        idxHyperplane = np.argmax(totalLikelihood)
        bestProjection = cdata @ hypDirs[idxHyperplane,:] # ATL I removed the transpose on cdata...
        idxLeft = np.where(bestProjection <= hypThresholds[idxHyperplane])[0]
        idxRight = np.where(bestProjection > hypThresholds[idxHyperplane])[0]

        # Save optimal hyperplane values
        hpNormal = hypDirs[idxHyperplane,:]
        hpThreshold = hypThresholds[idxHyperplane]
        # Save left node parameters to dictionary
        leftPPCA = lCandidatePPCA[idxHyperplane]
        rightPPCA = rCandidatePPCA[idxHyperplane]
        
        return hpNormal, hpThreshold, leftPPCA, rightPPCA, cdata[idxLeft,:], sdata[idxLeft,:], cdata[idxRight,:], sdata[idxRight,:]
    
    def checkContainsPPCA(self,node):
        # Take in (node) dictionary, check if it contains a valid PPCA model
        containsPPCA = False
        if ('ppca' in node.keys()) and isinstance(node['ppca'],mppca.ppcaModel): containsPPCA=True
        # if 'ppca' not in node.keys(): return # if there's no ppca key, then there's no ppca model...
        # if isinstance(node['ppca'],mppca.ppcaModel): containsPPCA=True
        return containsPPCA
    
    def summarizeForest(self, removeOriginal=False):
        """
        This function supports optimization by summarizing a random PPCA forest into easily accessible arrays. 
        The original forest of hyperplanes will remain the same (don't know how to optimize), but at the terminal node, the PPCA model will be replaced with a predefined index.
        Data will be passed through the reduced forest and each datapoint will be associated to an index. 
        Then, the data will be passed through the appropriate PPCA model using it's associated index.

        removeOriginal (default=False) removes the PPCA model from the original tree structures. They usually take up little memory, so I think it's not necessary to remove them. 
        """
        if removeOriginal: print("NOTE: In summarizeForest, removeOriginal was set to True, but it hasn't been coded yet!!!")

        for tt, tree in enumerate(self.forest):
            treeStructure = self.returnTreeStructure(tree) # Measure tree structure (all possible paths to terminal nodes)
            numPaths = len(treeStructure) 
            cPpcaMeans = np.zeros((numPaths,self.drDims))
            cPpcaCovs = np.zeros((numPaths,self.drDims,self.drDims))
            cPpcaInvCovs = np.zeros((numPaths,self.drDims,self.drDims))
            cPpcaLogDets = np.zeros(numPaths)
            for pp in range(numPaths):
                cPathKeys = self.returnDecPath(treeStructure[pp]) # For each path, create a list of strings describing path
                self.updateNestedTree(tree, cPathKeys, 'pathIdx', pp) # Add a pathIdx value corresponding to this path
                cPPCA = self.returnNestedTree(tree, cPathKeys, 'ppca') # return PPCA object
                assert isinstance(cPPCA, mppca.ppcaModel), "summarizeForest found an item under a 'ppca' key that is not a valid ppca model" 
                cPpcaMeans[pp,:] = cPPCA.u
                cPpcaCovs[pp,:,:] = cPPCA.cv
                cPpcaInvCovs[pp,:,:] = cPPCA.icv
                cPpcaLogDets[pp] = cPPCA.logDet
                
            # Add ppca model for each path to the top of the forest, along with the tree structure
            self.treeSummary.append(treeStructure)
            self.ppcaMeans.append(cPpcaMeans)
            self.ppcaCovs.append(cPpcaCovs)
            self.ppcaInvCovs.append(cPpcaInvCovs)
            self.ppcaLogDets.append(cPpcaLogDets)

    def returnTreeStructure(self,tree):
        # Returns tree structure - in form of list of lists, indicating all possible paths within tree
        # 0:left, 1:right
        treeIdx = 0
        treeStructure = []
        treeStructureInherited = []
        # initialize with all left 
        cTreePath = []
        cNode = tree
        while True:
            if cNode['terminalNode']: break
            cNode = cNode['left']
            cTreePath.append(0)
        treeStructure.append(cTreePath)
        # continue
        while True:
            # The code is setup so the last path is all right choices...
            if all([c==1 for c in treeStructure[-1]]): break
            # Otherwise, reset current node, initialize cTreePath, and extend path
            cNode = tree
            leftSwitch = False
            cTreePath = []
            while True:
                if cNode['terminalNode']: break # leave if we've reached the end

                if leftSwitch:
                    cNode = cNode['left']
                    cTreePath.append(0)

                elif len(cTreePath) < len(treeStructure[-1])-1:
                    # If we're more than 2 away from end of previous, copy it's next decision
                    #if treeStructure[-1][len(cTreePath)+1]==0:
                    if all([t==1 for t in treeStructure[-1][1+len(cTreePath):]]):
                        cNode = cNode['right']
                        leftSwitch = True
                        cTreePath.append(1)
                    else:
                        if treeStructure[-1][len(cTreePath)]==0:
                            cNode = cNode['left']
                            cTreePath.append(0)
                        else:
                            cNode = cNode['right']
                            cTreePath.append(1)
                elif len(cTreePath) == len(treeStructure[-1])-1 and (cTreePath==treeStructure[-1][:len(cTreePath)]):
                    # If we're one away from previous and all else is the same, go right
                    cNode = cNode['right']
                    cTreePath.append(1)
                elif len(cTreePath) >= len(treeStructure[-1])-1:
                    # If we're past the previous one, just go left
                    cNode = cNode['left']
                    cTreePath.append(0)
                else:
                    raise ValueError("there was an error in the logic... sorry")
            treeStructure.append(cTreePath)
        return treeStructure


    def returnDecPath(self,decValues):
        decPath = [{0:'left', 1:'right'}[cdecision] for cdecision in decValues]
        return decPath

    def updateNestedTree(self, tree, decPath, key, value):
        # used to dynamically set value of nested dictionary (here, tree=dictioanry, decPath=list of keys,key=final key, value=value to change or set)
        for dec in decPath:
            tree = tree.setdefault(dec)
        tree[key] = value

    def returnNestedTree(self, tree, decPath, key):
        # used to dynamically return value of nested dictionary (here, tree=dictioanry, decPath=list of keys, key=final key)
        for dec in decPath:
            tree = tree.setdefault(dec)
        return tree.setdefault(key)
    
    # ----------------------
    # -- handle landmarks --
    # ----------------------
    def constructScaffold(self):
        # Greedy algorithm: http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf
        l = np.zeros(self.numScafPoints,dtype=int)
        l[:self.numScafSeeds] = np.random.choice(self.numObs,self.numScafSeeds,replace=False)
        m = np.min(scipy.spatial.distance.cdist(self.drData[l[:self.numScafSeeds],:],self.drData,metric='Euclidean'),axis=0)
        for ii in range(self.numScafSeeds,self.numScafPoints):
            # Set next landmark to datapoint furthest from other datapoints
            l[ii] = np.argmax(m)
            m = np.minimum(m, np.sum((self.drData[l[ii],:] - self.drData)**2,axis=1))
        self.scafIdx = l
        self.scafData = self.drData[l,:]

        self.scafGridProb = self.smartGridProbability(self.scafData)
        self.scafTrProb = self.scafGridProb / np.sum(self.scafGridProb,axis=1).reshape(-1,1)
        assert np.allclose(np.sum(self.scafTrProb,axis=1),1), "sum(trProb,axis=1) does not all equal(close-to) 1!"
        self.scafLocalDist = probabilityToDistance(self.scafTrProb)
        self.scafGlobalDist = scipy.sparse.csgraph.johnson(self.scafLocalDist)
        sgdMinimum = np.minimum(self.scafGlobalDist, self.scafGlobalDist.T) # take minimum transition distance for each pairwise transition 
        sgdInfinite = np.logical_or(self.scafGlobalDist==np.inf, self.scafGlobalDist.T==np.inf) # True if either transition distance is infinite
        self.scafDist = (self.scafGlobalDist + self.scafGlobalDist.T)/2 # Will be infinite if either transition distance is infinite
        self.scafDist[sgdInfinite] = sgdMinimum[sgdInfinite] # If one side is infinite, use minimum transition distance
        assert np.all(self.scafDist >= 0), "Some of the scaffold distances are negative. This probably means that smartGridProbability() is broken."
        assert np.all(self.scafDist != np.inf), "Some of the scaffold distance are infinite. This means that Johnson's algorithm couldn't find a path between at least 1 pair of points in either direction. This sometimes happens when temporal resolution is too high, so datapoints are too close together in state space."
        
    # ------------------------------
    # -- multidimensional scaling --
    # ------------------------------
    def performMDS(self, dims, recompute=False, returnCoord=False, verbose=True):
        # dims=the number of dims used for mapping numScafPoints onto a new manifold
        # this function performs MDS on scaffold points for a given dimension
        # if it exists already, it'll just pull them from storage
        # if it doesn't exist, it'll recompute them
        # coordinates only returned out of function if requested explicitly
        # if recompute=False, returnCoord=False, and coordinates have already been computed, then this function has no effect so it aborts before doing anything! --
        # verbose determines whether to use a progress bar for the iterations of sammon mapping
        
        assert dims>=1 and (isinstance(dims,int) or np.issubdtype(dims, np.integer)), "dims must be positive integer"
        
        # If already computed, no recompute, and returning, then return existing coordinates
        if (dims in self.scafManifoldDim) and (not recompute) and returnCoord:
            dimidx = self.scafManifoldDim.index(dims)
            return self.scafManifold[dimidx]
        
        # If already computed, no recompute, and not returning, return empty
        if (dims in self.scafManifoldDim) and (not recompute) and (not returnCoord):
            print(f"Coordinates already computed for dim={dims}. Exiting function.")
            return None
        
        # For (dims not in self.scafManifoldDim) or (recompute), we need to compute the mapping as follows
        
        # Start by initializing MDS coordinates with the closed from eigendecomposition solution 
        D2 = self.scafDist**2
        C = np.identity(self.numScafPoints) - (1/self.numScafPoints)*np.ones((self.numScafPoints,self.numScafPoints))
        B = (-1/2) * (C @ D2 @ C)
        
        # Do eigendecomposition on B
        w,v = scipy.linalg.eigh(B)
        idx = np.argsort(-w) # return index of descending sort
        w = w[idx[:dims]] # sort eigenvalues, only keep requested ones
        v = v[:,idx[:dims]] # sort eigenvectors, only keep requested ones
        
        # Initial MDS Coordinates:
        mdsCoord = v @ np.diag(np.sqrt(w))
        if dims==1: mdsCoord = mdsCoord.reshape(-1,1)
        
        # Use sammon mapping to refine MDS coordinates
        self.error = -1*np.ones(self.mdsSammonIterations)
        distvec = squareform(self.scafDist)
        
        errorConstant = 1/np.sum(distvec)
        
        progressBar = tqdm(range(self.mdsSammonIterations),disable=(not verbose))
        for it in progressBar:
            progressBar.set_description(f"Performing sammon mapping: Iteration {it+1}/{self.mdsSammonIterations}")
            
            cdistvec = pdist(mdsCoord)
            cdistmat = squareform(cdistvec)
            
            gradient = getGradient(self.scafDist, cdistmat, mdsCoord, errorConstant)
            dblgrad = getDoubleGradient(self.scafDist, cdistmat, mdsCoord, errorConstant)
            mdsCoord = mdsCoord - self.mdsSammonAlpha * gradient / np.linalg.norm(dblgrad)
            cdistvec = pdist(mdsCoord)
            self.error[it] = sammonError(distvec, cdistvec)        
        
        assert mdsCoord.ndim==2, "mdsCoord.ndim != 2" # This just for debugging the case where dims==1
        
        # Center coordinates
        if dims>1:
            v = mppca.ppca(mdsCoord, method='numComponents', dimPrm=dims)[0] # return all eigenvectors
            mdsCoord = (mdsCoord - np.mean(mdsCoord,axis=0)) @ v
        else:
            mdsCoord = mdsCoord - np.mean(mdsCoord)
            
        if dims in self.scafManifoldDim:
            dimidx = self.scafManifoldDim.index(dims)
            self.scafManifold[dimidx] = mdsCoord
        else:
            self.scafManifold.append(mdsCoord)
            self.scafManifoldDim.append(dims)
        
        if returnCoord:
            return mdsCoord # if requested, return output
    
    # ---------------------------------------------
    # function library: forward and reverse mapping
    # ---------------------------------------------
    def mapping(self, data, scaffold, target, k, lam):
        # data is an N x D1 array of datapoints to acquire weights for
        # scaffold is an M x D1 array of scaffold points to approximate with
        # target is an M x D2 array of target points associated with each scaffold point on a different manifold
        if data.ndim == 1: data = data.reshape(1,-1)
        ND,DD = data.shape
        MS,DS = scaffold.shape
        MT,DT = target.shape
        assert DD==DS, "data and scaffold must have same number of dimensions"
        assert MS==MT, "scaffold and target must contain the same number of datapoints"
        idx = np.argsort(cdist(data, scaffold), axis=1)[:,:k]
        datamap = np.empty((ND,DT))
        for nd in range(ND):
            cgvec = data[nd,:].reshape(1,-1) - scaffold[idx[nd,:],:]
            cw = mapWeights(cgvec,k,lam)
            datamap[nd,:] = np.dot(cw,target[idx[nd,:],:])
        return datamap
    
    def forwardMapping(self, data):
        if self.scafManifold is None:
            raise ValueError("Forward mapping can only be used if the scaffold is already mapped to a manifold. Use performMDS with updateObject=True to do so.")
        if self.mappingForwardK is None:
            raise ValueError("Forward mapping can only be used if the forward k parameter has been set. Use optimizeForwardMapping to do so (or provide it manually).")
        if self.mappingForwardLambda is None:
            raise ValueError("Forward mapping can only be used if the forward lambda parameter has been set. Use optimizeForwardMapping to do so (or provide it manually).")
        if self.mappingForwardOptDim != self.scafManifold.shape[1]:
            print("Alert: forward mapping parameters optimized for different manifold dimensionality!")
        if data.ndim==1: data=data.reshape(1,-1)
        assert data.shape[1]==self.drDims, "for forward mapping, data must have same dimensions as scaffold points (same as ppca models of data)"
        return self.mapping(data, self.scafData, self.scafManifold, self.mappingForwardK, self.mappingForwardLambda)
    
    def backwardMapping(self, data):
        if self.scafManifold is None:
            raise ValueError("Backward mapping can only be used if the scaffold is already mapped to a manifold. Use performMDS with updateObject=True to do so.")
        if self.mappingForwardK is None:
            raise ValueError("Backward mapping can only be used if the backward k parameter has been set. Use optimizeBackwardMapping to do so (or provide it manually).")
        if self.mappingForwardLambda is None:
            raise ValueError("Backward mapping can only be used if the backward lambda parameter has been set. Use optimizeBackwardMapping to do so (or provide it manually).") 
        if self.mappingBackwardOptDim != self.scafManifold.shape[1]:
            print("Alert: backward mapping parameters optimized for different manifold dimensionality!")
        if data.ndim==1: data=data.reshape(1,-1)
        assert data.shape[1]==self.scafManifold.shape[1], "for backward mapping, data must have same dimensions as most recently set MDS coordinates of the scaffold points" 
        return self.mapping(data, self.scafManifold, self.scafData, self.mappingBackwardK, self.mappingBackwardLambda)

    def optimizeForwardMapping(self, dims2use, numIterations=5, frac2test=0.1, kOptions=None, lamOptions=None):
        # optimize forward mapping using euclidean distance between scaffoldData and scaffoldManifold using a desired dimensionality
        # can choose how many iterations to do, what fraction of scaffold points to test, and the options for k and lambda
        if kOptions is None: kOptions = self.mappingOptionsK
        if lamOptions is None: lamOptions = self.mappingOptionsLambda
        NK = len(kOptions)
        NL = len(lamOptions)
        numTest = int(self.numScafPoints * frac2test)
        edgeFlags = np.zeros((4,len(dims2use)))
        
        tStart = time.time()
        for dimidx, dim in enumerate(dims2use):
            elapsedTime = time.time() - tStart
            if dimidx>0: eta=elapsedTime * len(dims2use)/dimidx
            else: eta=np.nan
            print(f"Optimizing forward mapping for dim={dim}, completed {dimidx}/{len(dims2use)} in {(time.time()-tStart):.1f} seconds, eta of finish: {(eta):.1f}")
            if dim in self.mappingForwardOptDim:
                # If this dimensionality has already been optimized, replace the values
                replace = True
                dimIdxInObject = self.mappingForwardOptDim.index(dim)
            else: 
                # Otherwise, append to end of optimization values
                replace = False
            
            if len(self.scafManifold)>0 and (dim==self.scafManifold.shape[1]):
                # If stored manfiold coordinates are same as requested dimension, use them
                print(f"Using previously computed manifold coordinates that match current dimensionality...")
                scafManifold = self.scafManifold
            else:
                # Otherwise, generate coordinates without saving
                print(f"Generating manifold coordinates...")
                scafManifold = self.performMDS(dim, updateObject=False, returnCoord=True, verbose=False)
            
            # Generate display variables for tracking progress...
            iterationBar = IntProgress(min=0, max=numIterations-1)
            kLambdaOptionsBar = IntProgress(min=0, max=(NK-1)*(NL-1))
            display(iterationBar)
            display(kLambdaOptionsBar)
            
            netError = np.zeros((numIterations,NK,NL))
            for it in range(numIterations):
                iterationBar.value=it
                iterationBar.description=f"Iteration {it+1}/{numIterations}"
                idxMap = np.random.choice(self.numScafPoints, numTest, replace=False)
                idxScaf = np.delete(np.arange(self.numScafPoints), idxMap)
                for ki in range(NK):
                    for li in range(NL):
                        kLambdaOptionsBar.value=ki*NL+li
                        kLambdaOptionsBar.description=f"K/Lambda: {li+1}/{NL*NK}"
                        testMap = self.mapping(self.scafData[idxMap,:], self.scafData[idxScaf,:], scafManifold[idxScaf,:], kOptions[ki], lamOptions[li])
                        netError[it,ki,li] = np.sqrt(np.sum((scafManifold[idxMap,:] - testMap)**2))
            
            totalError = np.mean(netError,axis=0)
            idxBest = np.unravel_index(np.argmin(totalError),totalError.shape)
            if idxBest[0]==0: edgeFlags[0,dimidx] = 1
            if idxBest[0]==NK-1: edgeFlags[1,dimidx] = 1
            if idxBest[1]==0: edgeFlags[2,dimidx] = 1
            if idxBest[1]==NL-1: edgeFlags[3,dimidx] = 1
            if replace:
                self.mappingForwardK[dimIdxInObject] = kOptions[idxBest[0]]
                self.mappingForwardLambda[dimIdxInObject] = lamOptions[idxBest[1]]
                self.mappingForwardOptDim[dimIdxInObject] = dim
                self.mappingForwardOptionsError[dimIdxInObject] = totalError
            else:
                self.mappingForwardK.append(kOptions[idxBest[0]])
                self.mappingForwardLambda.append(lamOptions[idxBest[1]])
                self.mappingForwardOptDim.append(dim)
                self.mappingForwardOptionsError.append(totalError)
            clear_output(wait=True)
        print(f"Optimizing forward mapping finished for dims={dims2use} in {(time.time()-tStart):.1f} seconds.")
        if np.any(edgeFlags[0,:]): print(f"Alert: the smallest k (k={np.min(kOptions)}) was selected for dim={dims2use[edgeFlags[0,:]==1]}. Consider expanding search.")
        if np.any(edgeFlags[1,:]): print(f"Alert: the largest k (k={np.max(kOptions)}) was selected for dim={dims2use[edgeFlags[1,:]==1]}. Consider expanding search.")
        if np.any(edgeFlags[2,:]): print(f"Alert: the smallest lambda ({np.min(lamOptions)}) was selected for dim={dims2use[edgeFlags[2,:]==1]}. Consider expanding search.")
        if np.any(edgeFlags[3,:]): print(f"Alert: the largest lambda ({np.max(lamOptions)}) was selected for dim={dims2use[edgeFlags[3,:]==1]}. Consider expanding search.")
    
    
    def optimizeBackwardMapping(self, dims2use, numIterations=5, frac2test=0.1, kOptions=None, lamOptions=None):
        # optimize backward mapping using euclidean distance between scaffoldData and scaffoldManifold using a desired dimensionality
        # can choose how many iterations to do, what fraction of scaffold points to test, and the options for k and lambda
        if kOptions is None: kOptions = self.mappingOptionsK
        if lamOptions is None: lamOptions = self.mappingOptionsLambda
        NK = len(kOptions)
        NL = len(lamOptions)
        numTest = int(self.numScafPoints * frac2test)
        edgeFlags = np.zeros((4,len(dims2use)))
        
        tStart = time.time()
        for dimidx, dim in enumerate(dims2use):
            elapsedTime = time.time() - tStart
            if dimidx>0: eta=elapsedTime * len(dims2use)/dimidx
            else: eta=np.nan
            print(f"Optimizing backward mapping for dim={dim}, completed {dimidx}/{len(dims2use)} in {(time.time()-tStart):.1f} seconds, eta of finish: {(eta):.1f}")
            if dim in self.mappingBackwardOptDim:
                # If this dimensionality has already been optimized, replace the values
                replace = True
                dimIdxInObject = self.mappingBackwardOptDim.index(dim)
            else: 
                # Otherwise, append to end of optimization values
                replace = False
            
            if len(self.scafManifold)>0 and (dim==self.scafManifold.shape[1]):
                # If stored manfiold coordinates are same as requested dimension, use them
                print(f"Using previously computed manifold coordinates that match current dimensionality...")
                scafManifold = self.scafManifold
            else:
                # Otherwise, generate coordinates without saving
                print(f"Generating manifold coordinates...")
                scafManifold = self.performMDS(dim, updateObject=False, returnCoord=True, verbose=False)
            
            # Generate display variables for tracking progress...
            iterationBar = IntProgress(min=0, max=numIterations-1)
            kLambdaOptionsBar = IntProgress(min=0, max=(NK-1)*(NL-1))
            display(iterationBar)
            display(kLambdaOptionsBar)
            
            netError = np.zeros((numIterations,NK,NL))
            for it in range(numIterations):
                iterationBar.value=it
                iterationBar.description=f"Iteration {it+1}/{numIterations}"
                idxMap = np.random.choice(self.numScafPoints, numTest, replace=False)
                idxScaf = np.delete(np.arange(self.numScafPoints), idxMap)
                for ki in range(NK):
                    for li in range(NL):
                        kLambdaOptionsBar.value=ki*NL+li
                        kLambdaOptionsBar.description=f"K/Lambda: {li+1}/{NL*NK}"
                        testMap = self.mapping(scafManifold[idxMap,:], scafManifold[idxScaf,:], self.scafData[idxScaf,:], kOptions[ki], lamOptions[li])
                        netError[it,ki,li] = np.sqrt(np.sum((self.scafData[idxMap,:] - testMap)**2))
            
            totalError = np.mean(netError,axis=0)
            idxBest = np.unravel_index(np.argmin(totalError),totalError.shape)
            if idxBest[0]==0: edgeFlags[0,dimidx] = 1
            if idxBest[0]==NK-1: edgeFlags[1,dimidx] = 1
            if idxBest[1]==0: edgeFlags[2,dimidx] = 1
            if idxBest[1]==NL-1: edgeFlags[3,dimidx] = 1
            if replace:
                self.mappingBackwardK[dimIdxInObject] = kOptions[idxBest[0]]
                self.mappingBackwardLambda[dimIdxInObject] = lamOptions[idxBest[1]]
                self.mappingBackwardOptDim[dimIdxInObject] = dim
                self.mappingBackwardOptionsError[dimIdxInObject] = totalError
            else:
                self.mappingBackwardK.append(kOptions[idxBest[0]])
                self.mappingBackwardLambda.append(lamOptions[idxBest[1]])
                self.mappingBackwardOptDim.append(dim)
                self.mappingBackwardOptionsError.append(totalError)
            clear_output(wait=True)
        print(f"Optimizing backward mapping finished for dims={dims2use} in {(time.time()-tStart):.1f} seconds.")
        if np.any(edgeFlags[0,:]): print(f"Alert: the smallest k (k={np.min(kOptions)}) was selected for dim={dims2use[edgeFlags[0,:]==1]}. Consider expanding search.")
        if np.any(edgeFlags[1,:]): print(f"Alert: the largest k (k={np.max(kOptions)}) was selected for dim={dims2use[edgeFlags[1,:]==1]}. Consider expanding search.")
        if np.any(edgeFlags[2,:]): print(f"Alert: the smallest lambda ({np.min(lamOptions)}) was selected for dim={dims2use[edgeFlags[2,:]==1]}. Consider expanding search.")
        if np.any(edgeFlags[3,:]): print(f"Alert: the largest lambda ({np.max(lamOptions)}) was selected for dim={dims2use[edgeFlags[3,:]==1]}. Consider expanding search.")
        
    def getMappingParameters(self, dim, direction='forward'):
        assert direction=='forward' or direction=='backward', "direction must be set to either 'forward' or 'backward'"
        if direction=='forward':
            assert dim in self.mappingForwardOptDim, f"Forward mapping has not been optimized for D={dim} yet."
            idxForward = self.mappingForwardOptDim.index(dim)
            k = self.mappingForwardK[idxForward]
            lam = self.mappingForwardLambda[idxForward]
        if direction=='backward':
            assert dim in self.mappingBackwardOptDim, f"Backward mapping has not been optimized for D={dim} yet."
            idxBackward = self.mappingBackwardOptDim.index(dim)
            k = self.mappingBackwardK[idxBackward]
            lam = self.mappingBackwardLambda[idxBackward]
        return k, lam
            
    # -----------------------
    # -- forest processing --
    # -----------------------
    def smartGridProbability(self, data):
        # Compute probability of transition from each point in data to each point in data using PPCA models in forest
        # Automatically use log switch for computing probability due to numerical stability
        # Uses summarized tree structure 
        assert data.ndim==2, "Data must be a matrix"
        N,D = data.shape
        assert D==self.drDims, "Data must have same dimensionality as ppcaModels in random forest" 
        
        probability = -1*np.ones((self.numTrees,N,N))
        progressBar = tqdm(range(self.numTrees))
        for tt in progressBar:
            progressBar.set_description(f"Measuring probability in tree {tt+1}/{self.numTrees} -- comparing {N} points to {N} points.")
            tPathIdx = self.returnPathIndexLoop(data, tt)
            tPpcaMean = self.ppcaMeans[tt][tPathIdx,:]
            tPpcaInvCov = self.ppcaInvCovs[tt][tPathIdx,:,:]
            tPpcaLogDet = self.ppcaLogDets[tt][tPathIdx]
            for n in range(N):
                cudata = data - tPpcaMean[n,:]
                cexparg = np.sum((cudata @ tPpcaInvCov[n,:,:]) * cudata,axis=1)
                cloglikelihood = -D/2*np.log(2*np.pi) - 1/2*tPpcaLogDet[n] - cexparg/2
                probability[tt,n,:] = np.exp(cloglikelihood)
        return np.median(probability,axis=0)

    def smartForestLikelihood(self, cdata, sdata):
        # Compute probability of transition from all points in cdata to all points in sdata
        # both are NxD arrays of N observations and D dimensions, where D = self.drDims
        assert cdata.ndim==2, "cdata must be a matrix"
        assert sdata.ndim==2, "sdata must be a matrix"
        assert cdata.shape == sdata.shape, "cdata and sdata must have same shape"
        N,D = cdata.shape
        assert D==self.drDims, "dimensions of cdata&sdata must be same as the ppcaModels in randomForest"
        
        probability = np.empty((self.numTrees,N))
        for tt in range(self.numTrees):
            # For each tree, start by returning path index of each datapoint
            tPathIdx = self.returnPathIndexLoop(cdata, tt)
            probability[tt,:] = self.fastLikelihood(sdata, self.ppcaMeans[tt][tPathIdx,:], self.ppcaInvCovs[tt][tPathIdx,:,:], self.ppcaLogDets[tt][tPathIdx], D)
        return np.median(probability,axis=0)

    def fastLikelihood(self, data, u, iS, logDet, D): 
        udata = data - u
        exparg = np.einsum('dm,dmn,dn->d',udata, iS, udata)
        logLikelihood = -D/2*np.log(2*np.pi) - logDet/2 - exparg/2
        return np.exp(logLikelihood)
    
    def returnPathIndexLoop(self, cdata, treeIdx):
        assert cdata.ndim==2, "cdata must be a matrix"
        N,D = cdata.shape
        assert D==self.drDims, "dimensions of cdata must be same as the ppcaModels in randomForest"
        pathIndices = np.empty(N,dtype=int)
        for n in range(N):
            pathIndices[n] = self.returnPathIndex(cdata[n,:],self.forest[treeIdx])
        return pathIndices

    def returnPathIndex(self, cdata, tree):
        if not tree['terminalNode']: 
            # Determine appropriate leaf and go to next level
            cproj = cdata @ tree['hpNormal']
            if cproj <= tree['hpThreshold']:
                return self.returnPathIndex(cdata,tree['left'])
            else:
                return self.returnPathIndex(cdata,tree['right'])
        else:
            assert "pathIdx" in tree, "TerminalNode found without pathIdx key. This means the tree hasn't been summarized."
            return tree['pathIdx']

    # -- finish --


# --------------------------------------------------
# -------------- supporting functions --------------
# --------------------------------------------------
def reconstructionError(data, reconstruction):
    assert data.ndim==2, "data must be a matrix"
    assert reconstruction.ndim==2, "reconstruction must be a matrix"
    assert data.shape==reconstruction.shape, "data and reconstruction must have same shape"
    udata = np.mean(data,axis=1)
    devFromMean = np.mean(np.sum((data - udata.reshape(-1,1))**2,axis=1))
    devFromRecon = np.mean(np.sum((data - reconstruction)**2,axis=1))
    return 1-devFromRecon/devFromMean
    
def probabilityToDistance(probVector,minCutoff=0):#np.finfo(np.float64).eps):
    # Convert probability to distance (with dist = sqrt(-log(p)))
    # But avoiding nonsensical computation on p=0 (set this distance to infinity)
    # Version 0
    # distVector = np.where(probVector > minCutoff, probVector, np.inf)
    # np.log(distVector, out=distVector, where=distVector<np.inf)
    # np.sqrt(-distVector, out=distVector, where=distVector<np.inf)
    # return np.abs(distVector)
    
    # Version 1
    assert minCutoff>=0, "minCutoff must be at least 0"
    distVector = np.empty_like(probVector)
    idxAboveCutoff = probVector > minCutoff
    distVector[idxAboveCutoff==False] = np.inf
    distVector[idxAboveCutoff] = np.sqrt(-np.log(probVector[idxAboveCutoff]))
    return distVector

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64), nogil=True, parallel=True)
def getGradient(distmat, cdistmat, coord, errorConstant):
    # Fast method for acquiring gradient in sammonMapping
    gradient = np.zeros_like(coord)
    for n in nb.prange(coord.shape[0]):
        for d in range(coord.shape[1]):
            for j in range(coord.shape[0]):
                if j!=n:
                    gradient[n,d] += -2/errorConstant * (distmat[n,j] - cdistmat[n,j])/(distmat[n,j]*cdistmat[n,j])*(coord[n,d] - coord[j,d])
    return gradient

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64), nogil=True, parallel=True)
def getDoubleGradient(distmat, cdistmat, coord, errorConstant):
    # Fast method for acquiring double gradient in sammonMapping
    doubleGradient = np.zeros_like(coord)
    for n in nb.prange(coord.shape[0]):
        for d in range(coord.shape[1]):
            for j in range(coord.shape[0]):
                if j!=n:
                    doubleGradient[n,d] += -2/errorConstant * (1/(distmat[n,j]*cdistmat[n,j]))*((distmat[n,j]-cdistmat[n,j]) - \
                        (coord[n,d] - coord[j,d])**2/cdistmat[n,j]*(1+(distmat[n,j]-cdistmat[n,j])/cdistmat[n,j]))
    return doubleGradient

@nb.njit(nb.float64(nb.float64[:],nb.float64[:]), nogil=True, parallel=True)
def sammonError(distvec, cdistvec):
    return np.sum((distvec - cdistvec)**2/distvec) / np.sum(distvec)

@nb.njit(fastmath=True)
def mapWeights(gvec, k, lam):
    gmat = gvec @ gvec.T
    w = np.linalg.inv(gmat + lam*np.identity(k)) @ np.ones(k)
    w = w / np.sum(w)
    return w



