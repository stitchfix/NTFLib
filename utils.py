import numpy as np
import numba

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def put3(indices, source, out):
    for j, (a, b, c) in enumerate(indices):
        out[j] = source[a, b, c]


def top_sparse3(x_indices, x_vals, model, out, beta, A, B):
    # For example, for factor B this is summing over z:
    # num[a, b, c] = A_az C_cz (X * model** (beta-2) )_abc
    # However, as X is sparse we only consider it's coordinates
    for (a, b, c), val in zip(x_indices, x_vals):
        AB = np.dot(A[a, :], B[b, :])
        out[a, b, c] = AB
        out[a, b, c] *= val
        out[a, b, c] *= model[a, b, c] ** (beta - 2)


def bot_sparse3(x_indices, x_vals, model, out, beta, A, B,):
    # Similar to the numerator top_sparse
    # but solving A_az B_bz (model ** (beta - 1))_abc
    for (a, b, c), val in zip(x_indices, x_vals):
        AB = np.dot(A[a, :], B[b, :])
        out[a, b, c] = AB
        out[a, b, c] *= model[a, b, c] ** (beta - 1)

tops = {3: top_sparse3}
bots = {3: bot_sparse3}
puts = {3: put3}


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
    b_ravel = np.zeros(x_ravel.shape)
    puts[rank](x_indices, b, b_ravel)
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

