#!/usr/bin/env python3

def triangle_type(a, b, c):
    """
    Determines whether three lengths form a triangle.
    If yes, returns the type of triangle.
    """

    if a <= 0 or b <= 0 or c <= 0:
        return "Not a triangle"

    if a + b <= c or a + c <= b or b + c <= a:
        return "Not a triangle"

    if a == b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
