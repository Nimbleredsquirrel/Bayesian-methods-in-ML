import numpy as np
from scipy.signal import fftconvolve
import scipy.stats as st
from scipy.special import logsumexp, softmax

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, _ = X.shape
    h, w = F.shape
    
    sq_diff = fftconvolve(X, np.flip(F.reshape(h, w, 1), axis=(0, 1)), mode='valid')
    backgr_conv = X * B[:, :, None]
    sq_diff -= fftconvolve(backgr_conv, np.ones((h, w, 1)), mode='valid')

    sq_diff += np.sum(backgr_conv, axis=(0, 1), keepdims=True)
    sq_diff *= -2
    sq_diff += np.sum(X ** 2, axis=(0, 1)).reshape(1, 1, -1)
    sq_diff += np.sum(F ** 2)
    sq_diff += np.sum(B ** 2, axis=(0, 1))
    sq_diff -= fftconvolve((B ** 2).reshape(H, W, 1), np.ones((h, w, 1)), mode='valid')
    
    ll = -1 / (2 * s ** 2) * sq_diff - H * W * (np.log(s * np.sqrt(2 * np.pi)))
    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    H, W, K = X.shape
    h, w = F.shape
    log_prob = calculate_log_probability(X, F, B, s)
    log_A = np.log(A + 1e-12)
    if not use_MAP:
        L = np.sum((log_prob + log_A.reshape(H - h + 1, W - w + 1, 1) - np.log(q + 1e-12)) * q)
        return L

    i_indx = q[0, :].astype(int)
    j_indx = q[1, :].astype(int)
    L = np.sum(log_prob[i_indx, j_indx, np.arange(K)] + log_A[i_indx, j_indx])
    return L


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    _, _, K = X.shape
    log_prob = calculate_log_probability(X, F, B, s)
    log_prob_A = log_prob + np.log(A + 1e-12)[:, :, None]
    log_prob_A -= logsumexp(log_prob_A, axis=(0, 1), keepdims=True)
    q = softmax(log_prob_A, axis=(0, 1))

    if use_MAP:
        q = np.array([np.unravel_index(np.argmax(q[:, :, k]), A.shape) for k in range(K)]).T
    return q



def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape

    if not use_MAP:
        A = np.mean(q, axis=2)
        F = np.mean(fftconvolve(X, np.flip(q, axis=(0, 1)), mode='valid', axes=(0, 1)), axis=2)
        backgr_q = 1 - fftconvolve(q, np.ones(F.shape)[:, :, None])
        B = np.sum(backgr_q * X, axis=2) / (np.sum(backgr_q, axis=2) + 1e-12)
        resid = K * (F ** 2).sum()
        resid += ((X - B[:, :, None]) ** 2).sum()
        in_exp = fftconvolve((2 * X - B[:, :, None]) * B[:, :, None], np.ones((h, w, 1)), mode='valid')
        in_exp -= 2 * fftconvolve(X, np.flip(F)[:, :, None], mode='valid')
        resid += np.sum(in_exp * q)
        noise_var = resid / (H * W * K)
        s = np.sqrt(noise_var)
        return F, B, s, A

    q_new = np.zeros((H - h + 1, W - w + 1, K))
    q_new[q[0], q[1], np.arange(K)] = 1
    B_denom = np.zeros((H, W))

    A = np.mean(q_new, axis=2)
    F = np.zeros((h, w))
    B = np.zeros((H, W))
    s = 0

    for k, (i, j) in enumerate(q.T):
        mask = np.ones((H, W))
        mask[i: i + h, j: j + w, k] = 0
        F += X[i: i + h, j: j + w, k]
        B += X[:, :, k] * mask
        B_denom += mask

    F /= K
    B /= np.where(B_denom == 0, 1, B_denom)

    for k, (i, j) in enumerate(q.T):
        img = B.copy()
        img[i: i + h, j: j + w, k] = F.copy()
        s += np.sum((X[:, :, k] - img) ** 2)

    s = np.sqrt(s / (H * W * K))
    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H, W, _ = X.shape
    if F is None:
        F = np.random.randint(0, 255, (h, w))
    if B is None:
        B = np.random.randint(0, 255, (H, W))
    if s is None:
        s = np.random.uniform(1, 100)
    if A is None:
        A = np.random.uniform(0,  X.max(), (H - h + 1, W - w + 1))
        A /= np.sum(A)

    LL = []
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        LL.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP))
        if i > 0 and np.abs(LL[i] - LL[i - 1]) < tolerance: 
            break
    return F, B, s, A, np.array(LL)


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    F_b, B_b, s_b, A_b, LL = run_EM(X, h, w, tolerance, max_iter, use_MAP)
    L_b = LL[-1]

    for _ in range(n_restarts - 1):
        F, B, s, A, LL = run_EM(X, h, w, tolerance, max_iter, use_MAP)
        if LL[-1] > L_b:
            F_b, B_b, s_b, A_b, L_b = F, B, s, A, LL[-1]

    return F_b, B_b, s_b, A_b, L_b
