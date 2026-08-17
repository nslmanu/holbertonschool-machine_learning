#!/usr/bin/env python3
"""Module that calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial.

    Return new list of coefficients, or None if poly is not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])
    if all(c == 0 for c in derivative):
        return [0]
    return derivative
