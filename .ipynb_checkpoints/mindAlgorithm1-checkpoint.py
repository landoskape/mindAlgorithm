# code to implement the mind algorithm from the following paper:
# paper: https://www.biorxiv.org/content/10.1101/418939v2.full.pdf

###############################################################################
# Discussion of goals and ideas (in Andrew speak rather than methods speak)
# Creating a Forest PPCA model
# - We start with all the data as a root node
# ------- Note: they preprocess with PCA, using the same 95% trick to reduce computation time (then project everything back at the end...)
# - At each node, we subdivide data into left/right child nodes using a decision hyperplane that maximizes net likelihood of the PPCA models. For each of these children, we fit a PPCA model.    
# ------- choosing a hyperplane: we start with nDir possible splitting directions on a unit sphere (serving to decorrelate ensemble)
# -------                        then we choose a split threshold such that "the mean successor state for each child node best predicts the actual successor states, as measured by the squared error"
# -------                        once a threshold has been chosen, full PPCA models are fit to each leaf. 
# -------                        finally, we compare the total PPCA likelihood for each nDir, evaluated at the best threshold, and choose the direction with the highest likelihood
# -------                        --> we continue this process until each leaf node has no fewer than nLeaf data points
# ------- choosing nDir and nLeaf: they chose nLeaf to be 40. They did it in an ad-hoc manner. (nLeaf must be large enough to fit a PPCA model at least...)
# -------                          they chose nDir to be 2... same thing, ad-hoc 
# - Ensembles: 
# ------- they use an ensemble approach where they fit S trees, and then combine the results of each tree by using the median probability of the output for each network transition
#
# optimizeHyperplane:
# -----    as input, this function requires a data array of the current neural activity along with an aligned successor array of the activity in the following timestep 
# -----    first, we split the data array in two, using one of the proposed hyperplane directions, then pick the best decision threshold for that hyperplane using an isotropic gaussian model
# -----    -----, note that this is constrained by the minimum number of datapoints in each leaf, set by the variable nLeaf (therefore, a split is not required if the number of datapoints is <2*nLeaf
# -----    -----, can either do this with efficient running sums (Torgo, 1999), or by choosing 10-20 candidates (divided along quantiles, set by nQuant) and choosing the best one 
# -----    second, once a threshold is chosen for each direction, we fit a full ppca model for the relevant successor states
# -----    (we do this for each of the proposed hyperplane directions, where the number of proposals is set by the variable nDir)
# -----    third, we sum the likelihoods of the PPCA models for each pair of left and right leafs
# -----    finally, we pick the hyperplane with the highest sum likelihood for the PPCA models of left and right leafs 
# -----    --------------------------
# -----    the output of the function is the following:
# -----    - hypDir,hypThreshold: the direction and threshold defining a decision hyperplane
# -----    - uLeft,covLeft,uRight,covRight: the mean, covariance, and noise of the PPCA model for left and right leaf nodes
# -----    - ----- NOTE: we don't really need the eigenvalues and eigenvectors of the PPCA model because they aren't required to measure the probability of a data point
# -----    - leftLikelihood, rightLikelihood: the likelihood of the PPCA model for the left and right leaf in the model
# -----    - the number of datapoints in each leaf

# inclusions
import numpy as np
import scipy
from sklearn_extra.cluster import KMedoids
from tqdm.notebook import tqdm, trange

# --------------------------------------------
# function library: random forest construction
# --------------------------------------------
def constructForest(currentData, successorData, numTrees=5, nDir=2, nLeaf=40, nQuant=10, keepSummaryData=False):
    forest = {}
    forest['numTrees'] = numTrees
    forest['nDir'] = nDir
    forest['nLeaf'] = nLeaf
    forest['nQuant'] = nQuant
    forest['tree'] = []
    forest['nDims'] = currentData.shape[0] # dimensionality of data
    
    progressBar = tqdm(range(numTrees))
    for tt in progressBar:
        progressBar.set_description(f'Fitting PPCA Tree {tt+1}/{numTrees}')
        if keepSummaryData:
            forest['tree'].append(splitForestNodeWithSummary(currentData,successorData,ppcaModel={},nDir=nDir,nLeaf=nLeaf,nQuant=nQuant))
        else:
            forest['tree'].append(splitForestNode(currentData,successorData,ppcaModel={},nDir=nDir,nLeaf=nLeaf,nQuant=nQuant))
    
    return forest

def addTrees(currentData, successorData, forest, numTrees):
    if currentData.shape[0] != forest['nDims']:
        raise ValueError("Current data doesn't have same number of dimensions as other data in previous trees")
        
    forest['numTrees'] = forest['numTrees'] + numTrees
    
    progressBar = tqdm(range(numTrees))
    for newTree in progressBar:
        progressBar.set_description(f'Adding PPCA Tree {newTree+1}/{numTrees}')
        forest['tree'].append(splitForestNode(currentData,successorData,ppcaModel={},nDir=forest['nDir'],nLeaf=forest['nLeaf'],nQuant=forest['nQuant']))
    return forest

def splitForestNode(currentData, successorData, ppcaModel={}, nDir=2, nLeaf=40, nQuant=10):
    # split a node in the dataset
    # Nodes are defined as:
    # - an ndarray of currentData and successorData (same shape and aligned)
    # - a dictionary with the PPCA model for the current & successor data
    # - ---- - if the dictionary is empty, assume it's the root node. if it is also a terminal node (by nLeaf), then generate an error!
    # Return node dictionary
    
    # Define node dictionary to be returned
    node = {}
    
    # if ppcaModel is empty, then this is the root node
    rootNode = (ppcaModel=={})
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # If this can't be split according to parameters, then return node!
    if D < 2*nLeaf:
        # If it has too few points to be split, and no PPCA model is provided, this function assumes it's a root node. But these are incompatible!
        print(D)
        if rootNode:
            raise ValueError("Too few data points to split, but no PPCA model provided (it is a terminal and a root node)")
        # If it has too few points to be split, but has a PPCA model, then it is a terminal node. Return PPCA model, identify it as a terminal node, and stop.
        if not checkValidPPCA(ppcaModel):
            raise ValueError("Terminal node identified, but no PPCA model provided")
        # Add PPCA model to node, then return
        node['terminalNode'] = True
        node['mean'] = ppcaModel['mean']
        node['covariance'] = ppcaModel['covariance']
        node['invcov'] = ppcaModel['invcov']
        node['likelihood'] = ppcaModel['likelihood']
        return node
    
    # If this node is being split, find the best split direction
    node['terminalNode'] = False
    hyperplane, leftNode, rightNode, leftData, leftSuccessor, rightData, rightSuccessor = optimizeHyperplane(currentData, successorData, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    
    # Add hyperplane to node dictionary
    node['hyperplane'] = hyperplane
    
    # Return left and right leaves of node (recursively create tree, this function will call itself until it returns a terminal node)
    node['left'] = splitForestNode(leftData, leftSuccessor, ppcaModel=leftNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    node['right'] = splitForestNode(rightData, rightSuccessor, ppcaModel=rightNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    return node

def optimizeHyperplane(currentData, successorData, nDir=2, nLeaf=40, nQuant=10):
    # choosing a hyperplane to create a decision boundary
    # we aim to model the successor states of our left and right leaf nodes with a multivariate gaussian.
    # outputs hyperplane separating data (as a dictionary) and left/right current and successor data
    
    if currentData.ndim != 2 or successorData.ndim != 2:
        raise ValueError("data must be a matrix")
    
    if currentData.shape != successorData.shape:
        raise ValueError("current data and successor data must have same shape")
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # Confirm that this data can be split (shouldn't even make it here if it can't, but always good to check!)
    if D < 2*nLeaf:
        raise ValueError("optimizeHyperplane received data with too few datapoints to split!!")
    
    # Prepare splitting procedure (use quantile speedup trick for decision thresholds)
    minQuant = (nLeaf+2)/D
    maxQuant = 1 - minQuant
    quantPoints = np.linspace(minQuant,maxQuant,nQuant-1)
    
    # Generate candidate hyperplane directions on unit hypersphere
    hypDirs = np.zeros((N,nDir)) # do it this way to avoid annoyance, it's very much not the bottleneck of the pipeline 
    while np.any(np.sum(hypDirs,axis=0) == 0): hypDirs = np.random.normal(0,1,(N,nDir))
    hypDirs = hypDirs / np.sqrt(np.sum(hypDirs**2,axis=0))
    
    # Preallocate variables
    hypThresholds = np.zeros(nDir)
    llCandidate = np.zeros((nDir,2))
    uCandidate = np.zeros((nDir,N,2))
    covCandidate = np.zeros((nDir,N,N,2))
    for ndir in range(nDir):
        cProjection = currentData.T @ hypDirs[:,ndir]
        cQuantiles = np.quantile(cProjection, quantPoints)
        ssError = []
        for cThreshold in cQuantiles:
            # Do isotropic gaussian model first
            idxLeft = np.where(cProjection <= cThreshold)[0]
            idxRight = np.where(cProjection > cThreshold)[0]
            cMeanLeft = np.mean(successorData[:,idxLeft],axis=1,keepdims=True)
            cMeanRight = np.mean(successorData[:,idxRight],axis=1,keepdims=True)
            cDevLeft = np.sum((successorData[:,idxLeft] - cMeanLeft)**2)
            cDevRight = np.sum((successorData[:,idxRight] - cMeanRight)**2)
            ssError.append(cDevLeft+cDevRight)
            if (len(idxLeft) < nLeaf) or (len(idxRight) < nLeaf):
                print(f"Left: {len(idxLeft)}, Right: {len(idxRight)}")
                raise ValueError("Quantization wrong in optimizeHyperplane, one leaf ended up with too few datapoints")
        
        # Then, for the best isotropic fit, compute a full ppca model
        idxBestThreshold = np.argmin(ssError)
        idxLeft = np.where(cProjection <= cQuantiles[idxBestThreshold])[0]
        idxRight = np.where(cProjection > cQuantiles[idxBestThreshold])[0]
        
        hypThresholds[ndir] = cQuantiles[idxBestThreshold]
        llCandidate[ndir,0],uCandidate[ndir,:,0],covCandidate[ndir,:,:,0] = ppca(successorData[:,idxLeft].T)[0:3]
        llCandidate[ndir,1],uCandidate[ndir,:,1],covCandidate[ndir,:,:,1] = ppca(successorData[:,idxRight].T)[0:3]
    
    # Find optimal direction, return indices for left and right data
    totalLikelihood = np.sum(llCandidate,axis=1)
    idxHyperplane = np.argmax(totalLikelihood)
    bestProjection = currentData.T @ hypDirs[:,idxHyperplane]
    idxLeft = np.where(bestProjection <= hypThresholds[idxHyperplane])[0]
    idxRight = np.where(bestProjection > hypThresholds[idxHyperplane])[0]
    
    # Save optimal hyperplane to dictionary
    hyperplane = {}
    hyperplane['direction'] = hypDirs[:,idxHyperplane]
    hyperplane['threshold'] = hypThresholds[idxHyperplane]
    # Save left node parameters to dictionary
    leftNode = {}
    leftNode['mean'] = uCandidate[idxHyperplane,:,0]
    leftNode['covariance'] = covCandidate[idxHyperplane,:,:,0]
    leftNode['likelihood'] = llCandidate[idxHyperplane,0]
    leftNode['invcov'] = np.linalg.inv(covCandidate[idxHyperplane,:,:,0])
    # Save right node parameters to dictionary
    rightNode = {}
    rightNode['mean'] = uCandidate[idxHyperplane,:,1]
    rightNode['covariance'] = covCandidate[idxHyperplane,:,:,1]
    rightNode['likelihood'] = llCandidate[idxHyperplane,1]
    rightNode['invcov'] = np.linalg.inv(covCandidate[idxHyperplane,:,:,1])
    
    print(f"Left length: {len(idxLeft)}, Right length: {len(idxRight)}")
    return hyperplane, leftNode, rightNode, currentData[:,idxLeft], successorData[:,idxLeft], currentData[:,idxRight], successorData[:,idxRight]


def checkValidPPCA(node):
    # Take in dictionary, check if it contains a valid PPCA model
    # First check if keys exist
    if ('mean' in node) and ('covariance' in node) and ('likelihood' in node):# and ('invcov' in node):
        # Then check if they have the right dimensions 
        if (node['mean'].ndim == 1) and (node['covariance'].ndim == 2) and (node['likelihood'].ndim==0): # and (node['invcov'].ndim==2) and (node['covariance'].shape == node['invcov'].shape):
            # Then check if the shapes match
            if node['mean'].shape[0] == node['covariance'].shape[0] == node['covariance'].shape[1]:
                # Only then, return True
                return True
    # Otherwise, this isn't a valid PPCA model
    return False

def summarizeForest(forest, removeOriginal=False):
    """
    This function supports optimization by summarizing a random PPCA forest into easily accessible arrays. 
    The original forest of hyperplanes will remain the same (don't know how to optimize), but at the terminal node, the PPCA model will be replaced with a predefined index.
    Data will be passed through the reduced forest and each datapoint will be associated to an index. 
    Then, the data will be passed through the appropriate PPCA model using it's associated index.
    
    removeOriginal (default=False) removes the PPCA model from the original tree structures. They usually take up little memory, so I think it's not necessary to remove them. 
    """
    if removeOriginal: print("NOTE: removeOriginal was set to True, but it hasn't been coded yet!!!")
    
    numTrees = len(forest['tree'])
    forest['treeSummary'] = []
    forest['ppcaMeans'] = []
    forest['ppcaCovs'] = []
    forest['ppcaInvCovs'] = []
    
    for tt, tree in enumerate(forest['tree']):
        treeStructure = returnTreeStructure(tree) # Measure tree structure (all possible paths to terminal nodes)
        numPaths = len(treeStructure) 
        cPpcaMeans = np.zeros((forest['nDims'],numPaths))
        cPpcaCovs = np.zeros((forest['nDims'],forest['nDims'],numPaths))
        cPpcaInvCovs = np.zeros((forest['nDims'],forest['nDims'],numPaths))
        for pp in range(numPaths):
            cPathKeys = returnDecPath(treeStructure[pp]) # For each path, create a list of strings describing path
            updateNestedTree(tree, cPathKeys, 'pathIdx', pp) # Add a pathIdx value corresponding to this path
            cPpcaMeans[:,pp] = returnNestedTree(tree, cPathKeys, 'mean') # Return the mean, covariance, and inverse covariance for this path along the tree
            cPpcaCovs[:,:,pp] = returnNestedTree(tree, cPathKeys, 'covariance')
            cPpcaInvCovs[:,:,pp] = returnNestedTree(tree, cPathKeys, 'invcovariance')
            
        # Add ppca model for each path to the top of the forest, along with the tree structure
        forest['treeSummary'].append(treeStructure)
        forest['ppcaMeans'].append(cPpcaMeans)
        forest['ppcaCovs'].append(cPpcaCovs)
        forest['ppcaInvCovs'].append(cPpcaInvCovs)

def returnTreeStructure(tree):
    # Returns tree structure - in form of list of lists, indicating all possible paths within tree
    # 0:left, 1:right
    treeIdx = 0
    treeStructure = []
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

def returnDecPath(decValues):
    decPath = [{0:'left', 1:'right'}[cdecision] for cdecision in decValues]
    return decPath
    
def updateNestedTree(tree, decPath, key, value):
    # used to dynamically set value of nested dictionary (here, tree=dictioanry, decPath=list of keys,key=final key, value=value to change or set)
    for dec in decPath:
        tree = tree.setdefault(dec)
    tree[key] = value
    
def returnNestedTree(tree, decPath, key):
    # used to dynamically return value of nested dictionary (here, tree=dictioanry, decPath=list of keys, key=final key)
    for dec in decPath:
        tree = tree.setdefault(dec)
    return tree.setdefault(key)

# ----------------------------------------------
# function library: measure probability with forest (using summarized trees!!!)
# ----------------------------------------------
def smartGridProbability(data1, forest, data2=None):
    # Compute probability of transition from each point in data1 to each point in data2 using PPCA models in forest
    # If data2 is None, compute probability of transitioning between each pair in data1 
    # Automatically use log switch for computing probability due to numerical stability
    # Uses summarized tree structure 
    if data2 == None: data2 = data1 # if no second data provided, compare data1 to itself
    if (data1.ndim != 2) or (data2.ndim != 2): raise ValueError("Data1 & Data2 must be matrices")
    D1,N1 = data1.shape
    D2,N2 = data2.shape
    if D1 != D2: raise ValueError("Data1 and Data2 have different dimensions")
    
    numTrees = forest['numTrees']
    probability = -1*np.ones((N1,N2,numTrees))
    return None

def smartProbability(cdata, sdata, tree):
    return None

def returnPathIndexLoop(cdata, tree):
    ND = cdata.shape[1]
    pathIndices = np.empty(ND,dtype=int)
    for nd in range(ND):
        pathIndices[nd] = returnPathIndex(cdata[:,nd],tree)
    return pathIndices
        
def returnPathIndex(cdata, tree):
    if not tree['terminalNode']: 
        # Determine appropriate leaf and go to next level
        cproj = tree['hyperplane']['direction'].reshape(1,-1) @ cdata.reshape(-1,1)
        if cproj <= tree['hyperplane']['threshold']:
            return returnPathIndex(cdata,tree['left'])
        else:
            return returnPathIndex(cdata,tree['right'])
    else:
        if not "pathIdx" in tree:
            raise ValueError("TerminalNode found, but didn't contain an index to this tree path. Summarize tree first!")
        return tree['pathIdx']
    
    
# ----------------------------------------------
# function library: measure probability with forest (using non-summarized trees)
# ----------------------------------------------
def gridProbability(data1, forest, method='log', data2=None):
    # --- note --- can speed this up with numba significantly:
    # ---- the trick is pulling out PPCA models with indices for data points, then finding the relevant index for each landmark point. 
    # ---- then, I can do the likelihood computations in numba without accessing the forest dictionary
    
    # compute probability for each point, compared to each other point
    if data2 == None: data2 = data1 # if no second data provided, compare data1 to itself
    if (data1.ndim != 2) or (data2.ndim != 2): raise ValueError("Data1 & Data2 must be matrices")
    D1,N1 = data1.shape
    D2,N2 = data2.shape
    if D1 != D2: raise ValueError("Data1 and Data2 have different dimensions")
    
    # Compute probability of each datapoint in 1 transitioning to each datapoint in data2
    # Initialize as -1 because we're computing probability... -1 indicates failure
    numTrees = forest['numTrees']
    crossProbability = -1 * np.ones((N1,N2,numTrees))
    progressBar = tqdm(range(numTrees))
    for tt in progressBar:
        progressBar.set_description(f"Measuring probability in tree {tt+1}/{numTrees} -- comparing {N1} points to {N2} points.")
        for n1 in range(N1):
            for n2 in range(N2):
                crossProbability[n1,n2,tt] = computeProbability(data1[:,n1],data2[:,n2],forest['tree'][tt],method=method)

    return np.median(crossProbability,axis=2)
    
    
def forestProbability(currentData, successorData, forest, method='log'):
    # compute probability for each transition using median of output from the forest 
    if currentData.ndim != 2 or successorData.ndim != 2:
        raise ValueError("data must be a matrix")
    
    if currentData.shape != successorData.shape:
        raise ValueError("current data and successor data must have same shape")
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    NT = forest['numTrees']
    
    probability = -1*np.ones((D,NT))
    progressBar = tqdm(range(NT))
    for tt in progressBar:
        progressBar.set_description(f'Measuring probability from tree {tt+1}/{NT}')
        for dd in range(D):
            probability[dd,tt] = computeProbability(currentData[:,dd],successorData[:,dd],forest['tree'][tt],method=method)
    
    return np.median(probability,axis=1)

def getProbability(currentData, successorData, tree, method='log'):
    # for each point in current data, output the probability of observing it's associated successor data by navigating tree and measuring likelihood
    if currentData.ndim != 2 or successorData.ndim != 2:
        raise ValueError("data must be a matrix")
    
    if currentData.shape != successorData.shape:
        raise ValueError("current data and successor data must have same shape")
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    probability = -1*np.ones(D)
    for dd in range(D):
        probability[dd] = computeProbability(currentData[:,dd],successorData[:,dd],tree,method=method)
    
    return probability
    
def computeProbability(cdata,sdata,tree,method='log'):
    # Compute probability of sdata from cdata using PPCA model in tree
    # method is a switch which controls how the likelihood is computed (logarithm is better numerically usually)
    if not tree['terminalNode']: 
        # Determine appropriate leaf and go to next level
        cproj = tree['hyperplane']['direction'].reshape(1,-1) @ cdata.reshape(-1,1)
        if cproj <= tree['hyperplane']['threshold']:
            return computeProbability(cdata,sdata, tree['left'])
        else:
            return computeProbability(cdata,sdata, tree['right'])
    else:
        # Otherwise, confirm we have a PPCA model and return probability
        if not checkValidPPCA(tree):
            raise ValueError("TerminalNode found, but didn't have a valid PPCA model")
        return likelihood(sdata, tree['mean'], tree['covariance'], method=method) # add tree['invcov'],  back

def likelihood(data, u, S, method='log'): # add iS back to this 
    D = data.shape[0]
    udata = data - u
    iS = np.linalg.inv(S)
    expArgument = -(1/2) * (udata.T @ iS @ udata)
    if method=='log':
        # Measure log likelihood first
        smartLogDet = 2*np.log(np.prod(np.diag(np.linalg.cholesky(S))))
        logLikelihood = -D/2*np.log(2*np.pi) - 1/2*smartLogDet + expArgument
        # Then convert back to likelihood
        likelihood = np.exp(logLikelihood)
    elif method=='exp':
        likelihood = np.exp(expArgument) / np.sqrt((2*np.pi)**D * np.linalg.det(S))
    else:
        raise ValueError("Did not recognize method, must be 'log' or 'exp'")
    
    return likelihood


# --------------------------------------------
# function library: ppca model
# --------------------------------------------
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


# --------------------------------------------
# function library: data management
# --------------------------------------------
def returnLandmarkPoints(data, numLandmark=2000, numSeed=10, algorithm='greedy'):
    # Can easily speed this up with Numba
    # -- All I need to do is vectorize the minimum computation across jj's.... 
    
    # K-Medioids Algorithm (takes too long for big datasets!)
    if algorithm=='kmedioids':
        kmedoids = KMedoids(n_clusters=numLandmark).fit(data.T)
        return kmedoids.cluster_centers_.T, kmedoids.medoid_indices_
    
    # Greedy algorithm: http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf
    if algorithm=='greedy':
        N = data.shape[1]
        l = np.zeros(numLandmark,dtype=int)
        l[:numSeed] = np.random.choice(N,numSeed,replace=False)
        m = np.zeros(N)
        for jj in range(N):
            # Determine how close each datapoint is to randomly selected seed points
            m[jj] = np.min(np.sum((data[:,l[:numSeed]] - data[:,jj].reshape(-1,1))**2),axis=0)
        
        progressBar = tqdm(range(numSeed,numLandmark))
        for ii in progressBar:
            progressBar.set_description(f"Selecting {ii+1}'th/{numLandmark} landmark point")
            # Set next landmark to datapoint furthest from other datapoints
            l[ii] = np.argmax(m)
            for jj in range(N):
                # Then reset distances to minimum distance if it is closer to recently selected landmark
                m[jj] = np.minimum(m[jj], np.sum((data[:,l[ii]] - data[:,jj])**2))
        return data[:,l],l

    raise ValueError("algorithm didn't match any of the available options...")

    
    
# --------------------------------------------
# function library: debugging type functions (not efficient, but useful for checking the algorithm after running it) 
# --------------------------------------------
def splitForestNodeWithSummary(currentData, successorData, ppcaModel={}, nDir=2, nLeaf=40, nQuant=10):
    # split a node in the dataset
    # Nodes are defined as:
    # - an ndarray of currentData and successorData (same shape and aligned)
    # - a dictionary with the PPCA model for the current & successor data
    # - ---- - if the dictionary is empty, assume it's the root node. if it is also a terminal node (by nLeaf), then generate an error!
    # Return node dictionary
    
    # Define node dictionary to be returned
    node = {}
    
    # if ppcaModel is empty, then this is the root node
    rootNode = (ppcaModel=={})
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # If this can't be split according to parameters, then return node!
    if D < 2*nLeaf:
        # If it has too few points to be split, and no PPCA model is provided, this function assumes it's a root node. But these are incompatible!
        if rootNode:
            raise ValueError("Too few data points to split, but no PPCA model provided (it is a terminal and a root node)")
        # If it has too few points to be split, but has a PPCA model, then it is a terminal node. Return PPCA model, identify it as a terminal node, and stop.
        if not checkValidPPCA(ppcaModel):
            raise ValueError("Terminal node identified, but no PPCA model provided")
        node['terminalNode'] = True
        node['mean'] = ppcaModel['mean']
        node['covariance'] = ppcaModel['covariance']
        node['invcov'] = ppcaModel['invcov']
        node['likelihood'] = ppcaModel['likelihood']
        return node
    
    # If this node is being split, find the best split direction
    node['terminalNode'] = False
    hyperplane, leftNode, rightNode, leftData, leftSuccessor, rightData, rightSuccessor, meanModelError, ppcaLikelihoodQuants = optimizeHyperplaneWithSummary(currentData, successorData, nDir, nLeaf, nQuant)
    
    # Add hyperplane to node dictionary
    node['hyperplane'] = hyperplane
    node['meanModelError'] = meanModelError
    node['ppcaLikelihoodQuants'] = ppcaLikelihoodQuants
    
    # Return left and right leaves of node (recursively create tree, this function will call itself until it returns a terminal node)
    node['left'] = splitForestNodeWithSummary(leftData, leftSuccessor, ppcaModel=leftNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    node['right'] = splitForestNodeWithSummary(rightData, rightSuccessor, ppcaModel=rightNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    return node


def optimizeHyperplaneWithSummary(currentData, successorData, nDir=2, nLeaf=40, nQuant=4):
    # choosing a hyperplane to create a decision boundary
    # we aim to model the successor states of our left and right leaf nodes with a multivariate gaussian.
    # outputs hyperplane separating data (as a dictionary) and left/right current and successor data
    
    if currentData.ndim != 2 or successorData.ndim != 2:
        raise ValueError("data must be a matrix")
    
    if currentData.shape != successorData.shape:
        raise ValueError("current data and successor data must have same shape")
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # Confirm that this data can be split (shouldn't even make it here if it can't, but always good to check!)
    if D < 2*nLeaf:
        raise ValueError("optimizeHyperplane received data with too few datapoints to split!!")
    
    # Prepare splitting procedure (use quantile speedup trick for decision thresholds)
    numDatapointsPerQuant = D/nQuant
    quantOffset = int(np.ceil(nLeaf / (D/nQuant)))
    quantPoints = np.linspace(0,1,nQuant+1)[quantOffset:-quantOffset]
    if len(quantPoints)==0:
        # If there are two few points, just slice the data in half
        quantPoints = 0.5
    elif quantOffset>1:
        # If the offset is greater than 1, still query nQuant points! 
        quantPoints = np.linspace(quantPoints[0],quantPoints[-1],nQuant) 

    # Generate candidate hyperplane directions on unit hypersphere
    hypDirs = np.zeros((N,nDir)) # do it this way to avoid annoyance, it's very much not the bottleneck of the pipeline 
    while np.any(np.sum(hypDirs,axis=0) == 0): hypDirs = np.random.normal(0,1,(N,nDir))
    hypDirs = hypDirs / np.sqrt(np.sum(hypDirs**2,axis=0))
    
    # Preallocate variables
    hypThresholds = np.zeros(nDir)
    llCandidate = np.zeros((nDir,2))
    uCandidate = np.zeros((nDir,N,2))
    covCandidate = np.zeros((nDir,N,N,2))
    meanModelError = np.zeros((nDir, len(quantPoints)))
    ppcaLikelihoodQuants = np.zeros((nDir, len(quantPoints)))
    for ndir in range(nDir):
        cProjection = currentData.T @ hypDirs[:,ndir]
        cQuantiles = np.quantile(cProjection, quantPoints)
        ssError = []
        for ii,cThreshold in enumerate(cQuantiles):
            # Do isotropic gaussian model first
            idxLeft = np.where(cProjection <= cThreshold)[0]
            idxRight = np.where(cProjection > cThreshold)[0]
            cMeanLeft = np.mean(successorData[:,idxLeft],axis=1,keepdims=True)
            cMeanRight = np.mean(successorData[:,idxRight],axis=1,keepdims=True)
            cDevLeft = np.sum((successorData[:,idxLeft] - cMeanLeft)**2)
            cDevRight = np.sum((successorData[:,idxRight] - cMeanRight)**2)
            ssError.append(cDevLeft+cDevRight)
            meanModelError[ndir,ii] = cDevLeft + cDevRight
            ppcaLikelihoodQuants[ndir,ii] += ppca(successorData[:,idxLeft].T)[0]
            ppcaLikelihoodQuants[ndir,ii] += ppca(successorData[:,idxRight].T)[0]
        
        # Then, for the best isotropic fit, compute a full ppca model
        idxBestThreshold = np.argmin(ssError)
        idxLeft = np.where(cProjection <= cQuantiles[idxBestThreshold])[0]
        idxRight = np.where(cProjection > cQuantiles[idxBestThreshold])[0]
        
        hypThresholds[ndir] = cQuantiles[idxBestThreshold]
        llCandidate[ndir,0],uCandidate[ndir,:,0],covCandidate[ndir,:,:,0] = ppca(successorData[:,idxLeft].T)[0:3]
        llCandidate[ndir,1],uCandidate[ndir,:,1],covCandidate[ndir,:,:,1] = ppca(successorData[:,idxRight].T)[0:3]
    
    # Find optimal direction, return indices for left and right data
    totalLikelihood = np.sum(llCandidate,axis=1)
    idxHyperplane = np.argmax(totalLikelihood)
    bestProjection = currentData.T @ hypDirs[:,idxHyperplane]
    idxLeft = np.where(bestProjection <= cThreshold)[0]
    idxRight = np.where(bestProjection > cThreshold)[0]
    
    # Save optimal hyperplane to dictionary
    hyperplane = {}
    hyperplane['direction'] = hypDirs[:,idxHyperplane]
    hyperplane['threshold'] = hypThresholds[idxHyperplane]
    # Save left node parameters to dictionary
    leftNode = {}
    leftNode['mean'] = uCandidate[idxHyperplane,:,0]
    leftNode['covariance'] = covCandidate[idxHyperplane,:,:,0]
    leftNode['likelihood'] = llCandidate[idxHyperplane,0]
    leftNode['invcov'] = np.linalg.inv(covCandidate[idxHyperplane,:,:,0])
    # Save right node parameters to dictionary
    rightNode = {}
    rightNode['mean'] = uCandidate[idxHyperplane,:,1]
    rightNode['covariance'] = covCandidate[idxHyperplane,:,:,1]
    rightNode['likelihood'] = llCandidate[idxHyperplane,1]
    rightNode['invcov'] = np.linalg.inv(covCandidate[idxHyperplane,:,:,1])
    
    
    # -- just added index !!! --
    return hyperplane, leftNode, rightNode, currentData[:,idxLeft], successorData[:,idxLeft], currentData[:,idxRight], successorData[:,idxRight], meanModelError[idxHyperplane,:], ppcaLikelihoodQuants[idxHyperplane,:]