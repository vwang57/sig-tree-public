"""
Class :class:`SignificanceTree` implements significance trees, both binary and continuous
Class :class:`SignificanceForest` implements significance forests
"""

import math
import random
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
import warnings

# aliases to maintain backwards compatibility
_MODE_ALIASES = {
    "binary": "binary",
    "continuous": "continuous",
    "evidence": "binary",
    "weightedevidence": "continuous",
}

def _normalize_mode(mode):
    try:
        normalized_mode = _MODE_ALIASES[mode]
    except KeyError:
        raise ValueError(
            'mode must be "binary" or "continuous". '
            'The legacy names "evidence" and "weightedevidence" '
            "are also supported."
        ) from None

    if mode in {"evidence", "weightedevidence"}:
        warnings.warn(
            f'mode="{mode}" is deprecated; '
            f'use mode="{normalized_mode}" instead.',
            DeprecationWarning,
            stacklevel=3,
        )

    return normalized_mode

def _tstat(y, w, variant="pseudooutcome"):
    """Get t-stat from assigning everyone to treatment
    """
    target_leaf_map = (
        np.zeros((len(y), 1))
        if variant == "leafwise"
        else None
    )
    return (score_policies(
        np.array([[1] * len(y)]).reshape(-1, 1),
        y, w,
        target_leaf_map=target_leaf_map,
        variant=variant)[0])

def score_policies(target_policies, y, w, *,
                   target_leaf_map=None, subset=False, variant="pseudooutcome"):
    """ Takes as input a n x n_policies matrix where each column is a target treatment policy
    on the n sample units. y is the observed outcome of each sample unit,
    w is the treatment status in the data,
    leaf_identities are the assignments of samples to leaves

    For each assignment policy a, we compute policy values ay.
    Then we estimate the average treatment effect on the policy values, i.e. E[ay(1) - ay(0)] for potential outcomes y(1), y(0).
    This is equivalent to the expected lift of the policy, notated u(a) in the main paper.
    We then return the t-statistic for each assignment policy.

    The variant parameter governs which method we use to compute the treatment effect (relevant when w is not None). There are three options:
    1. pseudooutcome: use IPW-style pseudo-outcomes
    2. ate: compute a simple difference in means between treated and control units
    3. leafwise: compute ate and variance within each target leaf, then average over all the target leaves to get final results

    If subset=True, we only focus on the subset of sample units that would be treated under the target policy (a = 1).
    Data on other sample units is then discarded in the computation.

    NB: this function performs a global computation over the entire assignment policy, hence is used for binary evidence-based assignment.
    Continuous assignments are instead dealt with by score_splits()
    """

    policy_values = target_policies * y.reshape(-1, 1)

    if subset:
        policy_values[target_policies == 0] = np.nan
    if w is not None:
        W = w.reshape(-1, 1)
        if subset:
            W = (target_policies * W).astype(float)
            W[target_policies == 0] = np.nan
        if variant == "pseudooutcome":
            policy_values = policy_values * \
                (W / np.nanmean(W, axis=0, keepdims=True) -
                 (1 - W) / np.nanmean(1 - W, axis=0, keepdims=True))
        elif variant == "ate":
            # Take simple difference in means:
            # average policy value over treated units, minus average policy value over control units

            treated = (w == 1)
            control = (w == 0)

            treated_values = policy_values[treated, :]
            control_values = policy_values[control, :]

            n_treated = np.sum(~np.isnan(treated_values), axis=0)
            n_control = np.sum(~np.isnan(control_values), axis=0)

            ate = np.nanmean(treated_values, axis=0) - np.nanmean(control_values, axis=0)

            se = (np.nanstd(treated_values, axis=0, ddof=1) ** 2 / n_treated
                  + np.nanstd(control_values, axis=0, ddof=1) ** 2 / n_control) ** 0.5

            return ate / se
        elif variant == "leafwise":
            scores = []
            leafids = np.unique(target_leaf_map)
            basemask = (target_policies == 1) if subset else np.full(target_policies.shape, True)
            leafwisemean = []
            leafwisevar = []
            leafwisecounts = []
            leafwisesize = []
            for leaf in leafids:
                leaf_membership = (target_leaf_map == leaf)
                mask = basemask & leaf_membership
                # leafwcount has two columns that respectively count control and treatment units (for units specified by the mask)
                leafwcount = np.array([mask.T @ (1 - w), mask.T @ w]).T
                leafwmean = np.array([mask.T @ (y * (1 - w)), mask.T @ (y * w)]).T / leafwcount
                leafwvar = np.array([mask.T @ ((y**2) * (1 - w)),
                                     mask.T @ ((y**2) * w)]).T / (leafwcount - 1) \
                            - (leafwmean)**2 * (leafwcount / (leafwcount - 1))
                leaf_observation_counts = np.sum(leaf_membership, axis=0)
                leafpolicy = np.divide(
                    np.sum(target_policies * leaf_membership, axis=0),
                    leaf_observation_counts,
                    out=np.zeros(target_policies.shape[1], dtype=float),
                    where=leaf_observation_counts > 0,
                )
                leafsize = np.sum(leafwcount, axis=1)
                leafwisemean.append(leafsize * leafpolicy *
                                    (leafwmean[:, 1] - leafwmean[:, 0]))    # get ate in the leaf
                leafwisevar.append((leafsize**2) * leafpolicy *
                                   np.sum(leafwvar / leafwcount, axis=1))
                leafwisecounts.append(np.min(leafwcount, axis=1))   # we take the minimum value to check if the leaf has 0 treatment or control units
                leafwisesize.append(leafsize)

            total = np.sum(basemask, axis=0)
            leafwisemean = np.nansum(leafwisemean, axis=0) / total
            leafwisestd = np.sqrt(np.nansum(leafwisevar, axis=0)) / total
            scores = leafwisemean / leafwisestd
            scores[np.any((np.array(leafwisecounts) == 0) & (np.array(leafwisesize) > 0), axis=0)] = -np.inf

            return scores

    return (np.count_nonzero(~np.isnan(policy_values), axis=0))**(.5) * np.nanmean(policy_values, axis=0) / \
        np.nanstd(policy_values, axis=0, ddof=1)

def score_splits(valid_side, node_Y, node_W, *,
                 variant="pseudooutcome"):
    """ Score candidate splits, for use with the local splitting procedure of continuous tree
    valid_side is a n x n_splits matrix, where n is the number of sample units and n_splits is the number of valid split points being considered
    Each cell in valid_side is a binary indicator which equals 1 if the sample unit is on the left side of the split point (and 0 if on the right)
    node_Y is the observed outcome of each sample unit,
    node_W is the treatment status for sample units in the leaf.
    Returns scores and weights

    The variant parameter determines how we compute the treatment effect within the leaf. It can take two values:
    1. "ate": calculate treatment effect by a simple difference-in-means between treatment and control units
    2. "pseudooutcome" (default): calculate treatment effect using IPW-style pseudo-outcomes

    NB: binary evidence-based assignment policies are dealt with separately by score_policies(), because they need to computed over the whole global policy
    whereas score_splits just handles a local computation over the sample units in node_Y
    """

    # the following matrices have 2 rows and as many columns as there are valid splits (i.e. same number of columns as valid_side)
    ates = np.zeros((2, valid_side.shape[1]))
    weights = np.zeros((2, valid_side.shape[1]))
    values = np.zeros((2, valid_side.shape[1]))
    samplesizes = np.zeros((2, valid_side.shape[1]))

    for thisside in [True, False]:
        # in the above for loop, True indicates left side; False indicates right side
        # so we'll first analyse sample units falling in the left child node from the split
        # afterwards we repeat the analysis for the right child node

        if node_W is not None:
            # repeat node_W as many times as there are valid splits (i.e. columns of valid_side)
            # and put them side-by-side into a new matrix W (of dimensions n x n_split)
            W = np.tile(node_W.astype(float), (1, valid_side.shape[1]))
            # discard data for sample units which are not on the side of the split we are currently considering
            W[valid_side != thisside] = np.nan

        if node_W is not None and variant == "ate":
            Y1 = np.tile(node_Y, (1, valid_side.shape[1]))
            Y0 = np.tile(node_Y, (1, valid_side.shape[1]))
            # W could potentially be continuous assignment weights
            # hence want to make sure we divide into treatment and control bins (W1 and W0)
            W1 = (W > .5)
            W0 = (W < .5)
            Y1[~W1] = np.nan    #Y1 only retains data on outcomes for treated units
            Y0[~W0] = np.nan    #Y0 only retains data on outcomes for control units
            ate = np.nanmean(Y1, axis=0) - np.nanmean(Y0, axis=0)
            se = (np.nanstd(Y1, axis=0)**2 / np.sum(W1, axis=0) +
                  np.nanstd(Y0, axis=0)**2 / np.sum(W0, axis=0))**.5
        else:
            if node_W is None:
                Y = np.tile(node_Y, (1, valid_side.shape[1]))
                Y[valid_side != thisside] = np.nan
            elif variant == "pseudooutcome":
                # transform each Y into a IPW-style pseudo-outcome
                # replicate the pseudo-outcome vector for each valid split
                # then put them together into a new matrix Y (of dimensions n x n_split)
                Y = np.tile(node_Y, (1, valid_side.shape[1])) * \
                    (W / np.nanmean(W, axis=0, keepdims=True) -
                     (1 - W) / np.nanmean(1 - W, axis=0, keepdims=True))

            ate = np.nanmean(Y, axis=0)
            se = np.nanstd(Y, axis=0) / \
                np.sum(valid_side == thisside, axis=0)**.5
            if np.nanstd(Y, axis=0).all() == 0:
                print("no variation left in leaf")

        # suppose thisside = True (so we're currently considering the left child node)
        # then 1 - int(thisside) = 0
        # so the relevant information for the left child node gets stored in the first row of the below matrices
        # if we're instead considering thisside = False (right child node), the relevant information gets stored in the second row
        # the below matrices then contain one column for each valid split point

        ates[1 - int(thisside), :] = ate
        weights[1 - int(thisside), :] = np.maximum(ate, 0) / se**2  # assignment weights for continuous assignment policy
        values[1 - int(thisside), :] = np.maximum(ate, 0)**2 / se**2    # score criterion for continuous assignment policy
        samplesizes[1 - int(thisside), :] = np.sum(valid_side == thisside, axis=0)
        # Could be simplified by only storing ATEs and SEs

    return values[1, :] + values[0, :], weights

def score_node(node_Y, node_W,
               variant="pseudooutcome"):
    """
    Calculate the score and output value for a node before it is split,
    for use with continuous significance tree

    The score uses the same criterion as score_splits(), allowing the
    children score to be compared with the score of the parent node.
    """

    node_size = node_Y.shape[0]

    if node_W is None:
        Y = node_Y.astype(float)
        ate = np.nanmean(Y)
        se = np.nanstd(Y) / np.sqrt(node_size)

    elif variant == "ate":
        treated = (node_W > 0.5)
        control = (node_W < 0.5)

        Y1 = np.where(treated, node_Y, np.nan)
        Y0 = np.where(control, node_Y, np.nan)

        ate = np.nanmean(Y1) - np.nanmean(Y0)

        se = np.sqrt(
            np.nanstd(Y1)**2 / np.sum(treated)
            + np.nanstd(Y0)**2 / np.sum(control)
        )

    elif variant == "pseudooutcome":
        Y = node_Y * (
            node_W / np.nanmean(node_W)
            - (1 - node_W) / np.nanmean(1 - node_W)
        )

        ate = np.nanmean(Y)
        se = np.nanstd(Y) / np.sqrt(node_size)

    else:
        raise ValueError(
            'For local scoring, variant must be "ate" or "pseudooutcome".'
        )

    score = np.maximum(ate, 0)**2 / se**2
    node_value = np.maximum(ate, 0) / se**2

    return score, node_value

def _get_valid_splits(side, node_X, node_W,
                      min_leaf_size, min_treated_leaf, min_untreated_leaf, balance_tol):
    # get number of rows (sample units) in node_X
    node_size = node_X.shape[0]
    # calculate the number of samples on the left child for each proposed split
    # side contains binary indicators which equal 1 if a given sample unit is on the left side of a split
    size_left = np.sum(side, axis=0)

    valid_split = (size_left >= min_leaf_size)
    # (node_size - size_left) is the number of sample units on the right side of a split
    valid_split &= (node_size - size_left >= min_leaf_size)

    # node_W contains treatment indicators for all sample units in that node
    # check that we're happy with the number and ratio of treated/untreated units in each child node
    if node_W is not None:
        ntreated = node_W.sum()
        nuntreated = (1 - node_W).sum()
        ntreated_left = (side * node_W).sum(axis=0)
        nuntreated_left = (side * (1 - node_W)).sum(axis=0)
        ntreated_right = ((1 - side) * node_W).sum(axis=0)
        nuntreated_right = ((1 - side) * (1 - node_W)).sum(axis=0)
        valid_split &= (ntreated_left >= min_treated_leaf)
        valid_split &= (nuntreated_left >= min_untreated_leaf)
        valid_split &= (ntreated_right >= min_treated_leaf)
        valid_split &= (nuntreated_right >= min_untreated_leaf)
        ratio = ntreated / (ntreated + nuntreated)
        ratio_left = ntreated_left / (ntreated_left + nuntreated_left)
        ratio_right = ntreated_right / (ntreated_right + nuntreated_right)
        valid_split &= (ratio_left >= (1 - balance_tol) * ratio)
        valid_split &= ((1 - balance_tol) * ratio_left <= ratio)
        valid_split &= (ratio_right >= (1 - balance_tol) * ratio)
        valid_split &= ((1 - balance_tol) * ratio_right <= ratio)

    return valid_split

class Node:
    """Building block of :class:`CausalTree` class.

    Parameters
    ----------
    sample_inds : array-like, shape (n, )
        Indices defining the sample that the split criterion will be computed on.

    estimate_inds : array-like, shape (n, )
        Indices defining the sample used for calculating balance criteria.

    """

    def __init__(self, sample_inds, treatment, depth, node_id, parent_id):
        self.split_sample_inds = sample_inds
        self.treatment = treatment
        self.depth = depth
        self.node_id = node_id
        self.parent_id = parent_id
        self.feature = -1   #-1 indicates a leaf node
        self.threshold = np.inf
        self.left = None
        self.right = None

    def find_tree_node(self, value):
        """
        Recursively find and return the leaf node of the causal tree that corresponds
        to the input feature vector.

        Parameters
        ----------
        value : array-like, shape (d_x,)
            Feature vector whose leaf node we want to find.
        """
        if self.feature == -1:
            return self
        elif value[self.feature] <= self.threshold:
            return self.left.find_tree_node(value)
        else:
            return self.right.find_tree_node(value)


class SignificanceTree:
    """Base class for growing a Significance Tree.

    Parameters
    ----------
    min_leaf_size : integer, optional (default=10)
        The minimum number of samples in a leaf.

    max_depth : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree.

    n_proposals_per_feat :  int, optional (default=10)
        Number of split proposals to be considered for each continuous variable.

    min_impurity_decrease : float
        The minimum increase in score that is required for a split to be activated

    min_treated_leaf : int,
        Minimum number of treated people on a node

    min_untreated_leaf : int
        Minimum number of untreated people on a node
        
    max_features : int, "sqrt" or None (default=None)
        Number of features to use when constructing this tree.
        If a subset of the total number of features, the subset will be randomly selected.
        If int, then consider no more than max_features
        If sqrt, consider sqrt(n_features), rounded down
        If None, use all features

    feature_selection_level : "node" or "tree" (default="node")
        Level at which to carry out random subset feature selection, if relevant
        If "node", randomly select a new feature subset at every split
        If "tree", randomly select a feature subset once for the whole tree

    balance_tol : float in [0, 1]
        A split is considered only if the children nodes that are created preserve the
        ratio of treated/total_samples of the parent node to within a (1 - tol) factor.

    variant : one of {'pseudooutcome', 'ate', 'leafwise'} (default='pseudooutcome')
        What finite sample estimate to use as an unbiased estimate for the policy effect

    subset : bool (default=False)
        Whether to calculate p-value of outcome only of the treated population under the new policy

    mode : one of {'binary', 'continuous'} (default = 'binary')
        Whether to compute a binary or continuous significance tree.
        The legacy aliases 'evidence' for 'binary' and 'weightedevidence' for 'continuous' are also supported

    random_state : int, optional (default=None)
        Seed for random number generation

    """

    def __init__(self,
                 min_leaf_size=10,
                 max_depth=10,
                 n_proposals_per_feat=10,
                 min_impurity_decrease=0,
                 min_treated_leaf=5,
                 min_untreated_leaf=5,
                 max_features=None,
                 feature_selection_level="node",
                 balance_tol=1,
                 variant='pseudooutcome',
                 subset=False,
                 mode="binary",
                 random_state=None):
        # Causal tree parameters
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.n_proposals_per_feat = n_proposals_per_feat
        self.min_impurity_decrease = min_impurity_decrease
        self.min_treated_leaf = min_treated_leaf
        self.min_untreated_leaf = min_untreated_leaf
        self.max_features = max_features
        self.feature_selection_level = feature_selection_level
        self.balance_tol = balance_tol
        self.variant = variant
        self.subset = subset
        self.mode = mode
        self._mode = _normalize_mode(mode)
        self.random_state = random_state
        # Tree structure
        self.tree = None

    def fit(self, X, y, w=None, effect_direction="positive"):
        """
        Recursively build a significance tree.

        Parameters
        ----------
        X : array-like, shape (n, d_x)
            Feature vector.

        y : array-like, shape (n, d_y)
            Outcomes.

        w : array-like, shape (n,)
            Treatment assignment (binary).

        effect_direction : one of {'positive', 'negative'} (default = 'positive')
            The algorithm is set up to search for positive effects by default.
            If instead we know the effect direction is negative, the algorithm will flip treatment and control before proceeding.
            If we're unsure we can ask the algorithm to propose a direction (based on training data only);
            rather than using the fit() method, use the separate function fit_propose_direction()
        """
        self.effect_direction = effect_direction
        if self.effect_direction == "negative" and w is not None:
            w = 1 - w

        self.y_ = y.copy()
        self.w_ = w.copy() if w is not None else None

        # compute number of features to use for each split
        if isinstance(self.max_features, int):
            if X.shape[1] > self.max_features:
                n_features = self.max_features
            else:
                n_features = X.shape[1] #cap n_features at the amount of features actually available
        elif self.max_features == "sqrt":
            n_features = math.floor(math.sqrt(X.shape[1]))
        elif self.max_features == None:
            n_features = X.shape[1]
        else:
            raise ValueError("Incorrect value for max_features: should be an int or \"sqrt\" or None")

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        if self.feature_selection_level == "tree":
            self.selected_features = np.random.choice(X.shape[1], size=n_features, replace=False)
            X = X[:, self.selected_features]

        # build the root node
        if self._mode == "binary":
            tstat_best = _tstat(y, w, self.variant)

            if np.isnan(tstat_best):
                raise ValueError(
                    "The root t-statistic is not finite. Check that outcomes are "
                    "finite, both treatment groups are present, and the "
                    "pseudo-outcomes have nonzero variance."
                )
            else:
                root_value = (tstat_best >= 0)
                best_score = max(tstat_best, 0.0)

            current_policy = np.full(
                X.shape[0],
                root_value,
                dtype=float,
            )

        else:
            root_Y = y.reshape(-1, 1)
            root_W = w.reshape(-1, 1) if w is not None else None

            best_score, root_value = score_node(
                root_Y,
                root_W,
                variant=self.variant,
            )

            current_policy = None

        self.tree = Node(
            sample_inds=np.arange(y.shape[0]),
            treatment=root_value,
            depth=0,
            node_id=0,
            parent_id=-1,
        )

        # node list stores the nodes that are yet to be split
        leaf_list = [self.tree]
        leaf_map = np.zeros(y.shape[0]) # map each sample unit to a leaf. Initialise by mapping them all to 0 (root node)

        node_cnt = 1  # current count of number of total nodes created
        helper_node_list = []
        found_split = True
        while found_split:

            found_split = False
            if self._mode != "binary":
                best_gain = -np.inf

            while len(leaf_list) > 0:
                node = leaf_list.pop()
                helper_node_list.append(node)

                if node.depth < self.max_depth:

                    # Create local sample set
                    # split_sample_inds indicates which part of the sample we want to split on
                    # for this part of the sample, we extract covariates and treatment status into node_X and node_W respectively
                    node_X = X[node.split_sample_inds]
                    node_W = w[node.split_sample_inds].reshape(-1, 1) if w is not None else None

                    # a split is determined by a feature and a threshold
                    # for now we assume either binary or continuous features and draw random features to split on.
                    # The threshold proposal generation could potentially be improved. Currently we go over all
                    # features. If a feature is binary we split at .5. If continuous we generate a set of
                    # percentile splits, where the number of percentile splits is controlled by n_proposals_per_feat

                    # some preliminary calculations that will be useful to generate candidate splits
                    isbinary = np.isin(X, [0, 1]).all(axis=0)  # return array of booleans indicating whether each column is a binary covariate or not
                    binary_features = np.arange(X.shape[1])[isbinary]  # return array containing indices of columns in X that were binary
                    cont_features = np.arange(X.shape[1])[~isbinary]

                    if self.feature_selection_level == "node" and n_features < X.shape[1]:
                        # randomly select a certain number of binary features to keep for forming the proposals (i.e. features we'll actually split on)

                        # need to satisfy a number of requirements. first consider these three:
                        # 1. num_binary_features <= len(binary_features)
                        # 2. num_cont_features <= len(cont_features)
                        # 3. num_binary_features + num_cont_features = n_features
                        # combining the three conditions by substituting out num_cont_features yields requirement [i]:
                        # n_features - len(cont_features) <= num_binary_features <= len(binary_features)
                        # then we have two further non-negativity constraints:
                        # 4. 0 <= num_binary_features
                        # 5. 0 <= num_cont_features = n_features - num_binary_features
                        # combining 4. and 5. gives a second requirement [ii]:
                        # 0 <= num_binary_features <= n_features
                        # We combine requirements [i] and [ii] by using whichever of the LHS and RHS constraints are more binding:

                        num_binary_features = random.randint(max(0, n_features - len(cont_features)),
                                                             min(len(binary_features), n_features))
                        num_cont_features = n_features - num_binary_features

                        selected_binary_feature_indices = random.sample(range(0, len(binary_features)),
                                                                        num_binary_features)  # sample without replacement
                        binary_proposals = binary_features[selected_binary_feature_indices]

                        selected_cont_feature_indices = random.sample(range(0, len(cont_features)), num_cont_features)
                        cont_selected_features = cont_features[selected_cont_feature_indices]
                    else:
                        binary_proposals = binary_features.copy()
                        cont_selected_features = cont_features.copy()
                    
                    binary_thr_proposals = .5 * np.ones(
                        len(binary_proposals))  # use a threshold of 0.5 for each selected binary feature
                    
                    cont_proposals = np.repeat(cont_selected_features,
                                               self.n_proposals_per_feat)  # for each selected continous feature, plan to generate n_proposals_per_feat many thresholds
                    dim_proposals = np.concatenate((binary_proposals,
                                                    cont_proposals))  # get index of all binary and continuous features that we'll split on

                    cont_thrs = np.percentile(
                        node_X[:, cont_selected_features], np.linspace(.05, .95, self.n_proposals_per_feat) * 100, axis=0)
                    thr_proposals = np.concatenate(
                        (binary_thr_proposals, cont_thrs.T.flatten()))

                    # calculate the binary indicator of whether sample unit i is on the left or the right
                    # side of proposed split j. So this is an n_samples x n_proposals matrix
                    side = node_X[:, dim_proposals] <= thr_proposals

                    # to be valid so as for the split we need to leave at least min_leaf_size on each side.
                    valid_split = _get_valid_splits(side, node_X, node_W,
                                                    self.min_leaf_size, self.min_treated_leaf, self.min_untreated_leaf,
                                                    self.balance_tol)

                    # if there is no valid split then don't create any children
                    if ~np.any(valid_split):
                        continue

                    # filter only the valid splits
                    valid_dim_proposals = dim_proposals[valid_split]
                    valid_thr_proposals = thr_proposals[valid_split]

                    # for valid splits only, get binary indicator for whether sample unit i is on left or right of those splits
                    # (indicator = 1 if on the left side)
                    # each row still corresponds to a different sample unit
                    valid_side = side[:, valid_split]

                    if self._mode == "binary":
                        # calculate the policy tstat for each candidate split and treatment assignment post split (global calculation)

                        # reshape the global current_policy to a column vector
                        # then clone this column once for each valid split in valid_side
                        # put these side-by-side into a matrix with dimensions (number of sample units) x (number of valid splits)
                        # this provides a starting policy that we'll update when considering new splits below
                        target_policies = np.tile(current_policy.reshape(-1, 1), (1, valid_side.shape[1]))
                        # target_leaf_map = None
                        # if (w is not None) and (self.variant == 'leafwise'):
                        target_leaf_map = np.tile(leaf_map.reshape(-1, 1), (1, valid_side.shape[1]))

                        # for a given proposed split, target_leaf_map will assign relevant sample units to the proposed new leaf nodes
                        # based on whether they are to the left or right of the split point.
                        # it does this by labelling sample units with the corresponding new leaf indices.
                        # the binary indicators in valid_side are 1 if the sample unit is to the left of the split point;
                        # else the binary indicator is 0 if sample unit is to the right
                        # at the split point we generate a new left and right leaf node, indexed by node_cnt and node_cnt + 1 respectively
                        # (where node_cnt is our cumulative node count)

                        target_leaf_map[node.split_sample_inds, :] = valid_side * node_cnt
                        target_leaf_map[node.split_sample_inds, :] += (1 - valid_side) * (node_cnt + 1)

                        for treat in [0, 1]:
                            # treat == 0 treats the new left leaf
                            # (recall the binary indicators in valid_side are 1 if the sample unit is to the left of the split point, else 0 if right side)
                            # update the *global* policy accordingly
                            target_policies[node.split_sample_inds, :] = valid_side if treat == 0 else 1 - valid_side
                            # then score the new policy globally
                            if self.variant == 'leafwise':
                                t_stats = score_policies(target_policies, y, w, target_leaf_map=target_leaf_map,
                                                     variant=self.variant, subset=self.subset)
                            else:
                                t_stats = score_policies(target_policies, y, w, target_leaf_map=None,
                                                         variant=self.variant, subset=self.subset)
                            scores = t_stats
                            finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
                            best_split_ind = np.argmax(finite_scores)
                            best_candidate_score = finite_scores[best_split_ind]

                            if best_candidate_score > best_score + self.min_impurity_decrease:
                                found_split = True
                                best_score = best_candidate_score
                                best_node = node
                                best_treat = treat
                                best_feature = valid_dim_proposals[best_split_ind]
                                best_threshold = valid_thr_proposals[best_split_ind]
                    else:
                        # handle local splitting criterion for continuous version
                        node_Y = y[node.split_sample_inds].reshape(-1, 1)

                        parent_score, parent_weights = score_node(
                            node_Y,
                            node_W,
                            variant=self.variant,
                        )

                        children_scores, weights = score_splits(
                            valid_side,
                            node_Y,
                            node_W,
                            variant=self.variant,
                        )

                        # Since the overall partition objective is additive across leaves,
                        # the improvement from a split is the new children contribution
                        # minus the contribution of the parent they replace
                        gains = children_scores - parent_score

                        # Reject candidates with undefined or infinite scores
                        finite_gains = np.where(np.isfinite(gains), gains, -np.inf)
                        best_split_ind = np.argmax(finite_gains)
                        candidate_gain = finite_gains[best_split_ind]

                        if candidate_gain > self.min_impurity_decrease and candidate_gain > best_gain:
                            found_split = True
                            best_gain = candidate_gain
                            best_node = node
                            best_weights = weights[:, best_split_ind]
                            best_feature = valid_dim_proposals[best_split_ind]
                            best_threshold = valid_thr_proposals[best_split_ind]

            # If found a node that improves score by splitting, then construct children
            # nodes and update the leaf list
            if found_split:
                while len(helper_node_list) > 0:
                    node = helper_node_list.pop()
                    if node == best_node:
                        left = (X[node.split_sample_inds, best_feature] <= best_threshold)
                        if self._mode == "binary":
                            current_policy[node.split_sample_inds] = left if best_treat == 0 else 1 - left
                            left_treat = (best_treat == 0)
                            right_treat = (best_treat == 1)
                        else:
                            left_treat = best_weights[0]
                            right_treat = best_weights[1]

                        node.feature = best_feature
                        node.threshold = best_threshold

                        left_sample_inds = node.split_sample_inds[left]
                        node.left = Node(sample_inds=left_sample_inds,
                                         treatment=left_treat, depth=node.depth + 1,
                                         node_id=node_cnt, parent_id=node.node_id)
                        if (self._mode == "binary"): # and (w is not None) and (self.variant == 'leafwise'):
                            leaf_map[left_sample_inds] = node_cnt
                        node_cnt += 1

                        right_sample_inds = node.split_sample_inds[~left]
                        node.right = Node(sample_inds=right_sample_inds,
                                          treatment=right_treat, depth=node.depth + 1,
                                          node_id=node_cnt, parent_id=node.node_id)
                        if (self._mode == "binary"): # and (w is not None) and (self.variant == 'leafwise'):
                            leaf_map[right_sample_inds] = node_cnt
                        node_cnt += 1

                        # add the created children to the list of not yet split nodes
                        leaf_list.append(node.left)
                        leaf_list.append(node.right)
                    else:
                        leaf_list.append(node)

                if self._mode != "binary":
                    best_score += best_gain

        return self

    def print_tree(self, xname=None):
        node_list = [self.tree]
        while node_list:
            node = node_list.pop(0)

            if node.feature >= 0:
                feature_index = node.feature
                if self.feature_selection_level == "tree":
                    feature_index = self.selected_features[feature_index]
                featname = (
                    xname[feature_index]
                    if xname is not None
                    else feature_index
                )
                print("Node: (id={}, depth={}, parent={}, "
                      "treatment={}, split_feat={}, "
                      "split_thres={})".format(node.node_id, node.depth, node.parent_id,
                                               node.treatment, featname, node.threshold))
            else:
                print("Leaf: (id={}, depth={}, "
                      "parent={}, treatment={})".format(node.node_id, node.depth, node.parent_id,
                                                        node.treatment))
            if node.left:
                node_list.append(node.left)
            if node.right:
                node_list.append(node.right)

        return self

    def print_tree_with_split_history(self, xname=None):
        # node_list now contains tuples of (node, split_history)
        node_list = [(self.tree, [])]
        while node_list:
            node, split_history = node_list.pop(0)

            if node.feature >= 0:
                feature_index = node.feature
                if self.feature_selection_level == "tree":
                    feature_index = self.selected_features[feature_index]
                featname = (
                    xname[feature_index]
                    if xname is not None
                    else feature_index
                )
                print("Node: (id={}, depth={}, parent={}, "
                      "treatment={}, split_feat={}, "
                      "split_thres={})".format(node.node_id, node.depth, node.parent_id,
                                               node.treatment, featname, node.threshold))

                left_history = split_history + [(featname, f"<= {node.threshold}")]
                right_history = split_history + [(featname, f"> {node.threshold}")]

                if node.left:
                    node_list.append((node.left, left_history))
                if node.right:
                    node_list.append((node.right, right_history))
            else:
                split_path = " -> ".join([f"{feat}: {cond}" for feat, cond in split_history])
                print("Leaf: (id={}, depth={}, "
                      "parent={}, treatment={}, split_path='{}')".format(node.node_id, node.depth, node.parent_id,
                                                                         node.treatment, split_path))
        return self

    def export_graphviz(self, fname, xname=None, format='png', view=True):
        from graphviz import Digraph
        dot = Digraph(comment='Policy Tree')
        node_list = [self.tree]
        while node_list:
            node = node_list.pop(0)
            constraint = ""
            if node.feature >= 0:
                feature_index = node.feature
                if self.feature_selection_level == "tree":
                    feature_index = self.selected_features[feature_index]
                featname = (
                    xname[feature_index]
                    if xname is not None
                    else f"X{feature_index}"
                )
                constraint = "{} <= {:.2f}".format(featname, node.threshold)
            if self.w_ is None:
                stats = "n={}, mean(y)={:.2f}, std(y)={:.2f}".format(len(node.split_sample_inds),
                                                                     np.mean(
                    self.y_[node.split_sample_inds]),
                    np.std(self.y_[node.split_sample_inds]))
                _, pval = ttest_1samp(self.y_[node.split_sample_inds], 0)
                stats += ("\\n Single-Sample t-test "
                          "p-value: {:.1E}".format(pval))
            else:
                untreated = (self.w_[node.split_sample_inds] == 0)
                stats = "n0={}, mean(y0)={:.2f}, std(y0)={:.2f}".format(untreated.sum(),
                                                                        np.mean(
                    self.y_[node.split_sample_inds][untreated]),
                    np.std(self.y_[node.split_sample_inds][untreated]))

                treated = (self.w_[node.split_sample_inds] == 1)
                stats += "\\n n1={}, mean(y1)={:.2f}, std(y1)={:.2f}".format(treated.sum(),
                                                                             np.mean(
                    self.y_[node.split_sample_inds][treated]),
                    np.std(self.y_[node.split_sample_inds][treated]))
                _, pval = ttest_ind(self.y_[node.split_sample_inds][treated],
                                    self.y_[node.split_sample_inds][untreated], equal_var=False)
                stats += ("\\n Two-Sample t-test "
                          "p-value: {:.1E}".format(pval))

            fillcolor = '#27ae5d' if node.treatment else '#e67936'
            dot.node("id{}".format(node.node_id),
                     "Treatment={}\\n"
                     "{}\\n \\n"
                     "{}".format(node.treatment, stats, constraint),
                     color='white',
                     shape='box',
                     fillcolor=fillcolor,
                     style='filled,rounded',
                     fontcolor='white',
                     fontsize='24pt')
            if node.left:
                dot.edge("id{}".format(node.node_id),
                         "id{}".format(node.left.node_id),
                         label="true")
                node_list.append(node.left)
            if node.right:
                dot.edge("id{}".format(node.node_id),
                         "id{}".format(node.right.node_id),
                         label="false")
                node_list.append(node.right)

        dot.render(fname, format=format, view=view)
        return dot

    def find_split(self, value):
        return self.tree.find_tree_node(value.astype(np.float64))

    def predict(self, X):
        if self.feature_selection_level == "tree":
            X = X[:, self.selected_features]
        policy = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            policy[i] = self.find_split(X[i]).treatment

        return policy

class SignificanceForest:
    """Base class for growing a significance forest
        Parameters
    ----------
    n_estimators  :  int, optional (default=2000)
        Number of trees to grow in the forest

    min_leaf_size : integer, optional (default 2)
        The minimum number of samples in a leaf.

    max_depth : integer, optional (default 1000)
        The maximum number of splits to be performed when expanding the tree.

    n_proposals_per_feat :  int, optional (default=10)
        Number of split proposals to be considered for each continuous variable.

    min_impurity_decrease : float
        The minimum increase in score that is required for a split to be activated

    min_treated_leaf : int,
        Minimum number of treated people on a node

    min_untreated_leaf : int
        Minumum number of untreated people on a node

    max_features : int, "sqrt" or None (default=None)
        Number of features to use when constructing this tree.
        If a subset of the total number of features, the subset will be randomly selected.
        If int, then consider no more than max_features
        If sqrt, consider sqrt(n_features), rounded down
        If None, use all features

    feature_selection_level : "node" or "tree" (default="node")
        Level at which to carry out random subset feature selection, if relevant
        If "node", randomly select a new feature subset at every split
        If "tree", randomly select a feature subset once for the whole tree

    balance_tol : float in [0, 1]
        A split is considered only if the children nodes that are created preserve the
        ratio of treated/total_samples of the parent node to within a (1 - tol) factor.

    variant : one of {'pseudooutcome', 'ate', 'leafwise'} (default='pseudooutcome')
        What finite sample estimate to use as an unbiased estimate for the policy effect

    subset : bool (default=False)
        Whether to calculate p-value of outcome only of the treated population under the new policy

    mode : one of {'binary', 'continuous'} (default = 'binary')
        Whether to compute a binary or continuous significance tree.
        The legacy aliases 'evidence' for 'binary' and 'weightedevidence' for 'continuous' are also supported

    random_state : int (default=False)
        Seed for random number generators

    bootstrap: bool (default=True)
        Whether to grow individual trees on bootstrapped samples or not
    """

    def __init__(self,
                 n_estimators=2000,
                 min_leaf_size=2,
                 max_depth=1000,
                 n_proposals_per_feat=10,
                 min_impurity_decrease=0,
                 min_treated_leaf=5,
                 min_untreated_leaf=5,
                 max_features="sqrt",
                 feature_selection_level="node",
                 balance_tol=1,
                 variant='pseudooutcome',
                 subset=False,
                 mode="binary",
                 random_state=None,
                 bootstrap=True):
        # Causal tree parameters
        self.n_estimators = n_estimators
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.n_proposals_per_feat = n_proposals_per_feat
        self.min_impurity_decrease = min_impurity_decrease
        self.min_treated_leaf = min_treated_leaf
        self.min_untreated_leaf = min_untreated_leaf
        self.max_features = max_features
        self.feature_selection_level = feature_selection_level
        self.balance_tol = balance_tol
        self.variant = variant
        self.subset = subset
        self.mode = mode
        self._mode = _normalize_mode(mode)
        self.random_state = random_state
        self.bootstrap=bootstrap
        self.tree = None # Tree structure

    def fit(self, X, y, w, effect_direction="positive"):
        """
        Recursively build a significance forest.

        Parameters
        ----------
        X : array-like, shape (n, d_x)
            Feature vector.

        y : array-like, shape (n, d_y)
            Outcomes.

        w : array-like, shape (n,)
            Treatment assignment (binary).

        effect_direction : one of {'positive', 'negative'} (default = 'positive')
            The algorithm is set up to search for positive effects by default.
            If instead we know the effect direction is negative, the algorithm will flip treatment and control before proceeding.
            If we're unsure we can ask the algorithm to propose a direction (based on training data only),
            using the separate method fit_propose_direction()
        """

        self.effect_direction = effect_direction
        if self.effect_direction == "negative" and w is not None:
            w = 1 - w

        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.estimators=[]
        estimator_seeds = [random.randint(0, 10000000) for i in range(self.n_estimators)]
        for i in range(self.n_estimators):
            estimator = SignificanceTree(min_leaf_size=self.min_leaf_size, max_depth=self.max_depth, n_proposals_per_feat=self.n_proposals_per_feat,
                                   min_impurity_decrease=self.min_impurity_decrease, min_treated_leaf=self.min_treated_leaf,
                                   min_untreated_leaf=self.min_untreated_leaf,
                                   max_features=self.max_features, feature_selection_level=self.feature_selection_level,
                                   balance_tol=self.balance_tol, variant=self.variant,
                                   subset=self.subset, mode=self._mode, random_state=estimator_seeds[i])
            bootstrap_sample_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=self.bootstrap)
            if w is not None:
                estimator.fit(X[bootstrap_sample_indices, ], y[bootstrap_sample_indices], w[bootstrap_sample_indices])
            elif w is None:
                estimator.fit(X[bootstrap_sample_indices,], y[bootstrap_sample_indices])

            self.estimators.append(estimator)

        return self

    def predict(self, X):
        """
        Return the prediction over the trees.

        Parameters
        ----------
        X : array-like, shape (n, d_x)
            Feature vector for which we want to return an assignment policy
        """

        self.estimator_predictions=[]
        for estimator in self.estimators:
            self.estimator_predictions.append(estimator.predict(X))

        if self._mode == "binary":
            # plurality vote
            predictions_sum = np.sum(np.array(self.estimator_predictions), axis=0)
            plurality_vote = predictions_sum > (len(self.estimator_predictions) * 0.5)
            return(plurality_vote)
            # I didn't add a tie-breaking procedure for now, though my sense is that one should be running this with
            # sufficiently many estimators that a tie would be extremely unlikely
        elif self._mode == "continuous":
            # average tree outputs
            return(np.nanmean(np.array(self.estimator_predictions), axis=0))
        else:
            raise ValueError("Incorrect value for mode")

# aliases to maintain backwards compatibility
PolicyTree = SignificanceTree
PolicyForest = SignificanceForest