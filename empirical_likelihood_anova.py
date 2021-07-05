import numpy as np 
import scipy
from scipy.optimize import fsolve, root_scalar
from tqdm import tqdm

# realisation of the Empirical Likelihood One-Way ANOVA (the description of the method can be found 
# in the book of its author: "Empirical Likelihood" by Art B. Owen, CHAPMAN & HALL/CRC, 2001). 

# utils.
def get_samples(s, n=20, K=5, std=None):
    '''
    samples from tasks for theme 6 (ksampletests)

    input:
    s:      string, type of sample
    n:      list of sample sizes
    K:      int, subsamples in sample
    std:    numpy array of standard deviations, shape = (K, )

    samples from:
    a) normal distribution:     N(a_i, 1).
    b) uniform distribution:    U[a_i-0.5, a_i+0.5]
    c) Laplace distribution:    Laplace(a_i, 1) ~ [exp(1) with sign] + a_i

    with:
    1) a_1=...=a_5=0,
    2) a_1=...=a_4=0, a_5=0.5,
    3) a_i=i/10,
    4) like 1) but: std(X_5) = 2*std(X_1).

    output:
    list with K subsamples, subsample shapes: (n[i], )
    '''

    if isinstance(n, int):
        n = np.ones(K).astype('int64') * n

    scale_norm = np.ones(K)          # std
    loc_uniform = -0.5 * np.ones(K)  # left bound of support
    scale_uniform = np.ones(K)       # length of support
    scale_laplace = np.ones(K)       # scale^{-1} is assosiated 'exp' parameter. 

    if isinstance(std, np.ndarray):
        # remind for [symmetric] uniform:   std(X) = (b-a) / sqrt(12) = [b = c, a = -c] = c / sqrt(3)
        # remind for Laplace:               std(X) = sqrt(2) * scale; 
        scale_norm = std
        loc_uniform = -std * np.sqrt(3)
        scale_uniform = 2 * std * np.sqrt(3)
        scale_laplace = std / np.sqrt(2)

    if s=='a1': # N(0, 1)
        X = [scipy.stats.norm.rvs(size=n[i], scale=scale_norm[i]) for i in range(K)]
    if s=='a2': # N(0, 1) & N(0.5, 1)
        X = [scipy.stats.norm.rvs(size=n[i], scale=scale_norm[i]) for i in range(K)]
        X[-1] = X[-1] + 0.5
    if s=='a3': # N(i/10, 1)
        X = [scipy.stats.norm.rvs(size=n[i], loc=i/10, scale=scale_norm[i]) for i in range(K)]
    if s=='a4': # N(0, 1) & N(0, 4)
        X = [scipy.stats.norm.rvs(size=n[i]) for i in range(K)]
        X[-1] = X[-1] * 2
    
    if s=='b1': # R[-1/2, 1/2]
        X = [scipy.stats.uniform.rvs(size=n[i], loc=loc_uniform[i], scale=scale_uniform[i]) for i in range(K)]
    if s=='b2': # R[-1/2, 1/2] & R[0, 1]
        X = [scipy.stats.uniform.rvs(size=n[i], loc=loc_uniform[i], scale=scale_uniform[i]) for i in range(K)]
        X[-1] = X[-1] + 0.5
    if s=='b3': # R[-0.5 + i/10, 0.5 + i/10]
        X = [scipy.stats.uniform.rvs(size=n[i], loc=loc_uniform[i], scale=scale_uniform[i]) + i/10 for i in range(K)]
    if s=='b4': # R[-1/2 , 1/2] & R[-1, 1]
        X = [scipy.stats.uniform.rvs(size=n[i], loc=-0.5, scale=1) for i in range(K)]
        X[-1] = X[-1] * 2
    
    if s=='c1': # Laplace(0, 1)
        X = [scipy.stats.laplace.rvs(size=n[i], scale=scale_laplace[i]) for i in range(K)]
    if s=='c2': # Laplace(0, 1) & Laplace(0.5, 1)
        X = [scipy.stats.laplace.rvs(size=n[i], scale=scale_laplace[i]) for i in range(K)]
        X[-1] = X[-1] + 0.5
    if s=='c3': # Laplace(i/10, 1)
        X = [scipy.stats.laplace.rvs(size=n[i], loc=i/10, scale=scale_laplace[i]) for i in range(K)]
    if s=='c4': # Laplace(0, 1) & Laplace(0, 0.5)
        X = [scipy.stats.laplace.rvs(size=n[i], scale=1) for i in range(K)]
        X[-1] = X[-1] * 2
    
    return X


def pvals_plot(pvals, ax, title, bisector=True):
    '''
    input:
    pvals:      sorted (!) numpy array
    ax:         matplotlib Axes object
    title:      string
    bisector:   y=x on plot

    change axes: plot of 'linearly interpolated' p-value ECDF. 
    '''
    if bisector:
        xvals = np.linspace(0, 1, num=pvals.shape[0]).tolist() + [1] 
        ax.plot(pvals.tolist()+[1], xvals, color='tab:blue')
        ax.plot(xvals, xvals, '--', color='tab:red')
    else:
        xvals = np.linspace(0, 1, num=pvals.shape[0])
        ax.plot(pvals, xvals, color='tab:blue')

    ax.set_title(title)


# utils.
def pseudo_log(x):
    '''
    log with quadratic approximation if x < 1.

    input: numpy array or float/int
    '''
    if not isinstance(x, np.ndarray):
        y = np.array(x)
    else:
        y = x.copy()
    
    y = np.where(y > 1, np.log(y), y-1 - (y-1)**2 / 2)
    return y if y.shape else y.item()


def L_star(l, mu, X):
    '''
    let K be the number of subsamples

    input:
    mu:         mean
    params:     array with shape (K,): [l1,...,l_K] - Lagrange multipliers
    X:          list of K numpy arrays with shape (n_i, )
    '''

    N = sum([x.shape[0] for x in X])
    K = len(X)
    
    stat = 0
    for i in range(K):
        term = N + l[i]*(X[i]-mu)
        if np.any(term < 0):
            print(f'Negative term: mu={mu}')
        stat += np.sum(pseudo_log(term))

    return stat - N*np.log(N)


def L(l, mu, X):
    '''
    let K be the number of subsamples.

    input:
    mu:         mean
    params:     array with shape (K,): [l1,...,l_K] - Lagrange multipliers
    X:          list of K numpy arrays with shape (n_i, )

    output:
    min_{mu} max_{l} L(l, mu) = L(l*, mu*) = -max_{mu} ln(R(mu))
    '''

    N = sum([x.shape[0] for x in X])
    K = len(X)
    
    stat = 0
    for i in range(K):
        term = N / (N + l[i]*(X[i]-mu))

		# This condition is bad (but right: we need to get log(R)=-inf).
		# It doesn't allow us to use optimizing techniques - cause optimizer will just do some negative term and go to inf...
        if np.any(term < 0):
            # print(f'Negative term under log: mu={mu}')
            return np.inf

        stat -= np.sum(np.log(term))
    return stat


def optimality_equations(l, X, mu):
    '''
    map R^{K+1} \to R^{K+1}

    input:
    l:    list with len K: [l1,...,l_K] - Lagrange multipliers (if mu=None, list with len K+1: [l1,...,l_K, mu])
    X:    list of K numpy arrays with shape (n_i, )

    optimality_equations(l*, X) = 0 for optimal l*

    output:
    mu=None:    numpy array, shape=(K+1,)
    mu=mu_0:    numpy array, shape=(K,)
    '''
    n = np.array([x.shape[0] for x in X]) # subsamples sizes
    N = n.sum()
    K = len(X)
    
    if mu is None: # mu is unknown: K+1 equations
        equations = np.zeros(K+1)
        for i in range(K):
            equations[i] += np.sum( (X[i]-l[-1]) / (N + l[i]*(X[i]-l[-1])) ) # sum_j
            equations[-1] += np.sum( l[i] / (N + l[i]*(X[i]-l[-1])) )
    else:          # mu = mu_0: K equations
        equations = np.zeros(K)
        for i in range(K):
            equations[i] += np.sum( (X[i]-mu) / (N + l[i]*(X[i]-mu)) ) # sum_j

    return equations


# main class.
class ANOVA_EL():
    '''
	options:
    1) run the Empirical Likelihood Ratio ANOVA test and get the (asymptotic-based) pvalue of this test.
    
    If H_0 wasn't rejected:
    2) find Maximum Empirical Likelihood Estimator (MELE) of the common mean.
    3) find asymptotic 95%-confidence interval for the common mean.
    '''
    def __init__(self):
        self.logR = 0           # log-profile function R(X)
        self.pvalue = 0         # pvalue of one-way ANOVA EL test
        self.l = []             # optimal Lagrange Multipliers
        self.MELE = 0           # optimal mu (Maximum Empirical Likelihood Estimator)
        
    def fit(self, X, verbose=False):
        '''
        X:          list of K numpy arrays with shape (n_i, ), n_1+...+n_K = N
        verbose:    prints output

        - MELE of mu.
        - log-statistic value.
        - test pvalue.
        '''
        self.df = len(X)-1
        self.logR = self.R(mu=None, X=X, verbose=verbose)
        self.pvalue = scipy.stats.chi2.sf(x=-2*self.logR, df=self.df)
    
    def R(self, mu, X, verbose=False):
        '''
        input:
        X:          list of K numpy arrays with shape (n_i, ), n_1+...+n_K = N
        mu:         if mu=None => find optimal mu (MELE); else: calculate log(R(mu))
        verbose:    if True - prints output

        output:
        log of maximum profile function.
        '''
        N = sum([x.shape[0] for x in X])
        K = len(X)
        
        if mu is None: # mu is unknown
            sample_mean = sum([x.sum() for x in X]) / N
            l = np.zeros(K).tolist() + [sample_mean]
            
            if verbose:
                l_optimal, d, _, message = fsolve(optimality_equations, l, args=(X, None), full_output=True)
                calls = d['nfev']                       # function calls.
                residual = np.sum(np.abs(d['fvec']))    # residual.
            else:
                l_optimal = fsolve(optimality_equations, l, args=(X, None))

            self.l, self.MELE = l_optimal[:-1], l_optimal[-1]

            if verbose:
                p = []
                for i in range(K):
                    p.append(1/( N + self.l[i]*(X[i]-self.MELE)))
                p = np.array(p)

                print()
                print(f"{f'Residual on call {calls}:':<25}", residual)
                print(f"{f'Opt. Lagrange mult.:':<25}", np.around(self.l, 6))
                print(f"{f'MELE:':<25}", self.MELE)
                print(f"{f'Probs is positive:':<25}", np.all([np.all(p_ > 0) for p_ in p]))
                print(f"{f'Sum of probs:':<25}", sum([p_.sum() for p_ in p]))
                print(message)
                print()
            return -L(self.l, self.MELE, X)
        else: # mu = mu_0

            # convex hull condition.
            #for x in X:
            #    if x.min() > mu or x.max() < mu:
            #        return -np.inf

            l = np.zeros(K).tolist()
            l_optimal = fsolve(optimality_equations, l, args=(X, mu))
            return -L(l_optimal, mu, X)

    def confidence_interval(self, X):
        '''
        can be applied only after fit method (to get self.MELE)
        '''
        x_1 = min([min(x) for x in X])
        x_N = max([max(x) for x in X])
        q = scipy.stats.chi2.ppf(0.95, df=(self.df+1)) # we need to add 1 to df as we don't get max_{\mu}.

        # log(R) is concave.
        try:
            left = root_scalar(lambda y: -2*self.R(y, X)-q, args=(), method='bisect', bracket=[x_1+1e-5, self.MELE], xtol=1e-6)
            right = root_scalar(lambda y: -2*self.R(y, X)-q, args=(), method='bisect', bracket=[self.MELE, x_N-1e-5], xtol=1e-6)
            return [left.root, right.root]
        except ValueError:
            return None # conf.set is empty...
