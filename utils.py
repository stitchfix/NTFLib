import numpy as np
import numba

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def top_sparse3(x_indices, x_vals, out, beta, factor, A, B, C):
    # In einstein notation with factor=0 this is 'bz,cz,abc->az'
    # In einstein notation with factor=1 this is 'az,cz,abc->bz'
    # In einstein notation with factor=2 this is 'az,bz,abc->cz'
    # However, you can use to your advantage that the
    # x_indices are sparesely defined as [a, b, c]
    assert factor in (0, 1, 2), "Factor index must be < rank"
    if factor == 0:
        for (a, b, c), val in zip(x_indices, x_vals):
            core = np.sum(A[a, :] * B[b, :] * C[c, :])
            temp = val * B[b, :] * C[c, :] * (core ** (beta - 2))
            out[a, :] += temp
    if factor == 1:
        for (a, b, c), val in zip(x_indices, x_vals):
            core = np.sum(A[a, :] * B[b, :] * C[c, :])
            temp = val * A[a, :] * C[c, :] * (core ** (beta - 2))
            out[b, :] += temp
    elif factor == 2:
        for (a, b, c), val in zip(x_indices, x_vals):
            core = np.sum(A[a, :] * B[b, :] * C[c, :])
            temp = val * A[a, :] * B[b, :] * (core ** (beta - 2))
            out[c, :] += temp


def bot_sparse3(x_indices, x_vals, out, beta, factor, A, B, C):
    # This is the same as top_ssparse but in this case 
    # we don't have the `val` term in the sum
    assert factor in (0, 1, 2), "Factor index must be < rank"
    if factor == 0:
        for (a, b, c), val in zip(x_indices, x_vals):
            core = np.sum(A[a, :] * B[b, :] * C[c, :])
            temp = B[b, :] * C[c, :] * (core ** (beta - 2))
            out[a, :] += temp
    if factor == 1:
        for (a, b, c), val in zip(x_indices, x_vals):
            core = np.sum(A[a, :] * B[b, :] * C[c, :])
            temp = A[a, :] * C[c, :] * (core ** (beta - 2))
            out[b, :] += temp
    elif factor == 2:
        for (a, b, c), val in zip(x_indices, x_vals):
            core = np.sum(A[a, :] * B[b, :] * C[c, :])
            temp = A[a, :] * B[b, :] * (core ** (beta - 2))
            out[c, :] += temp


tops = {3: top_sparse3}
bots = {3: bot_sparse3}


def parafac(factors):
    """Computes the parafac model of a list of matrices

    if factors=[A,B,C,D..Z] with A,B,C..Z of shapes a*k, b*k...z*k, returns
    the a*b*..z ndarray P such that
    p(ia,ib,ic,...iz)=\sum_k A(ia,k)B(ib,k)C(ic,k)...Z(iz,k)

    Parameters
    ----------
    factors : list of arrays
        The factors

    Returns
    -------
    out : array
        The parafac model
    """
    rank = len(factors)
    request = ''
    for factor in range(rank):
        request += alphabet[factor] + 'z,'
    request = request[:-1] + '->' + alphabet[:rank]
    return np.einsum(request, *factors, dtype=np.float32)


def beta_divergence(x_indices, x_ravel, b, beta):
    """Computes the total beta-divergence between the current model and
    a sparse X
    """
    rank = len(x_indices[0])
    b_ravel = b[x_indices]
    a, b = x_ravel, b_ravel
    idx = np.isfinite(a)
    idx &= np.isfinite(b)
    idx &= a > 0
    idx &= b > 0
    a = a[idx]
    b = b[idx]
    if beta == 0:
        return a / b - np.log(a / b) - 1
    if beta == 1:
        return a * (np.log(a) - np.log(b)) + b - a
    return (1. / beta / (beta - 1.) * (a ** beta + (beta - 1.)
            * b ** beta - beta * a * b ** (beta - 1)))

