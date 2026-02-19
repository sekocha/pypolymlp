"""Function of CG used for minimizing function."""

import warnings

import numpy as np

# from ._linesearch import (
from scipy.optimize._linesearch import (
    LineSearchWarning,
    line_search_wolfe1,
    line_search_wolfe2,
)
from scipy.optimize._optimize import (
    OptimizeResult,
    _call_callback_maybe_halt,
    _check_unknown_options,
    _LineSearchError,
    _prepare_scalar_function,
    _print_success_message_or_warn,
    _status_message,
    vecnorm,
)


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """

    extra_condition = kwargs.pop("extra_condition", None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LineSearchWarning)
            kwargs2 = {}
            for key in ("c1", "c2", "amax"):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(
                f,
                fprime,
                xk,
                pk,
                gfk,
                old_fval,
                old_old_fval,
                extra_condition=extra_condition,
                **kwargs2,
            )

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def _line_search_wolfe1(
    f,
    fprime,
    xk,
    pk,
    gfk=None,
    old_fval=None,
    old_old_fval=None,
    args=(),
    c1=1e-4,
    c2=0.9,
    amax=50,
    amin=1e-8,
    xtol=1e-14,
):

    ret = line_search_wolfe1(
        f,
        fprime,
        xk,
        pk,
        gfk,
        old_fval,
        old_old_fval,
        c1=c1,
        c2=c2,
        amin=amin,
        amax=amax,
    )

    if ret[0] is None:
        raise _LineSearchError()
    return ret


def _line_search_wolfe2(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs):
    """Line search."""
    extra_condition = kwargs.pop("extra_condition", None)
    kwargs2 = {}
    for key in ("c1", "c2", "amax"):
        if key in kwargs:
            kwargs2[key] = kwargs[key]

    ret = line_search_wolfe2(
        f,
        fprime,
        xk,
        pk,
        gfk,
        old_fval,
        old_old_fval,
        extra_condition=extra_condition,
        **kwargs2,
    )
    if ret[0] is None:
        raise _LineSearchError()
    return ret


def minimize_cg(
    fun,
    x0,
    args=(),
    jac=None,
    callback=None,
    gtol=1e-5,
    norm=np.inf,
    eps=0.001,
    maxiter=None,
    disp=False,
    return_all=False,
    finite_diff_rel_step=None,
    c1=1e-4,
    c2=0.4,
    workers=None,
    **unknown_options,
):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.4
        Parameter for curvature condition rule.
    workers : int, map-like callable, optional
        A map-like callable, such as `multiprocessing.Pool.map` for evaluating
        any numerical differentiation in parallel.
        This evaluation is carried out as ``workers(fun, iterable)``.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)

    retall = return_all

    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(
        fun,
        x0,
        jac=jac,
        args=args,
        epsilon=eps,
        finite_diff_rel_step=finite_diff_rel_step,
        workers=workers,
    )

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    xk = x0
    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, ord=norm)

    sigma_3 = 0.01

    while (gnorm > gtol) and (k < maxiter):
        deltak = np.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            #
            # See Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step

            # Accept step if it leads to convergence.
            if gnorm <= gtol:
                return True

            # Accept step if sufficient descent condition applies.
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)

        try:
            # alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe2(
            #     f,
            #     myfprime,
            #     xk,
            #     pk,
            #     gfk,
            #     old_fval,
            #     old_old_fval,
            #     c1=c1,
            #     c2=c2,
            #     amax=1e100,
            #     # amin=1e-100,
            #     extra_condition=descent_condition,
            # )
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe1(
                f,
                myfprime,
                xk,
                pk,
                gfk,
                old_fval,
                old_old_fval,
                c1=c1,
                c2=c2,
                amax=1e100,
                amin=1e-100,
            )

        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)

        if retall:
            allvecs.append(xk)
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break

    fval = old_fval
    if warnflag == 2:
        msg = _status_message["pr_loss"]
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message["maxiter"]
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    if disp:
        _print_success_message_or_warn(warnflag, msg)
        print(f"         Current function value: {fval:f}")
        print(f"         Iterations: {k:d}")
        print(f"         Function evaluations: {sf.nfev:d}")
        print(f"         Gradient evaluations: {sf.ngev:d}")

    result = OptimizeResult(
        fun=fval,
        jac=gfk,
        nfev=sf.nfev,
        njev=sf.ngev,
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=xk,
        nit=k,
    )
    if retall:
        result["allvecs"] = allvecs
    return result
