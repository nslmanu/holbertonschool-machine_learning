#!/usr/bin/env python3
"""Module that calculates the integral of a polynomial."""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial.

    Return new list of coefficients, or None if not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    integral = [C]
    for i in range(len(poly)):
        if poly[i] == 0:
            integral.append(0)
        else:
            coef = poly[i] / (i + 1)
            if coef == int(coef):
                integral.append(int(coef))
            else:
                integral.append(coef)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
