#!/usr/bin/env python3

def triangle_type(a, b, c):
    """
    Determines whether three lengths form a triangle.
    If yes, returns the type of triangle.
    """

    for x in (a,b,c):
        if not isinstance(x,(float,int)):
            raise TypeError("All sides must be numbers")

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

def test_isosceles_triangle_type():
    assert triangle_type(2,2,3)=="Isosceles"
    assert triangle_type(3,2,2)=="Isosceles"
    assert triangle_type(2,3,2)=="Isosceles"
    
def test_equilateral_triangle_type():
    assert triangle_type(2,2,2)=="Equilateral"

def test_equilateral_float_triangle_type():
    assert triangle_type(2.5,2.5,2.5)=="Equilateral"
    
def test_scalene_triangle_type():
    assert triangle_type(2,3,4)=="Scalene"

def test_not_triangle_type():
    assert triangle_type(2,3,5)=="Not a triangle"

def test_invalid_triangle_type():
    assert triangle_type(0,1,2)=="Not a triangle"
    assert triangle_type(-1,1,2)=="Not a triangle"

def test_boundary_case():
    assert triangle_type(1,1,2)=="Not a triangle"

def test_float_precision():
    assert triangle_type(0.1, 0.1, 0.2) == "Not a triangle"


def run_tests():
    test_isosceles_triangle_type()
    test_equilateral_triangle_type()
    test_equilateral_float_triangle_type()
    test_scalene_triangle_type()
    test_not_triangle_type()
    test_invalid_triangle_type()
    test_boundary_case()
    test_float_precision()
    print("All tests passed!")



if __name__ == "__main__":
    run_tests()



#def run_tests():
    #tests = [
        #test_isosceles_triangle_type,
        #test_equilateral_triangle_type,
        #test_equilateral_float_triangle_type,
        #test_scalene_triangle_type,
        #test_not_triangle_type,
        #test_invalid_triangle_type,
        #test_boundary_case
    #]

    #passed = 0

    #for test in tests:
        #try:
            #test()
            #print(f"{test.__name__}: PASSED")
            #passed += 1
        #except AssertionError as e:
            #print(f"{test.__name__}: FAILED -> {e}")

    #print(f"\n{passed}/{len(tests)} tests passed")