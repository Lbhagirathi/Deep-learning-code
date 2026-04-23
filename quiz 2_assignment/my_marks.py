#!/usr/bin/env python3

# n      # next line
# s      # step into
# c      # continue
# p var  # print variable
# l      # show code
# q      # quit

import pdb

def compute_average(marks):
    # (Pdb) p m
    # (Pdb) p total
    # (Pdb) n
    total = 0
    for m in marks:
        total = total + m
    avg = total / len(marks)
    return avg

def highest_mark(marks):
    max_mark = marks[0]
    for m in marks:
        if m > max_mark:
            max_mark = m
    return max_mark

def normalize_marks(marks):
    """
    Normalize marks to percentage scale (0-100)
    """
    pdb.set_trace()

    max_m = highest_mark(marks)
    normalized = []

    for m in marks:
        normalized.append((m / max_m) * 100)

    return normalized

def grade_students(marks):
    grades = []
    avg = compute_average(marks)

    for m in marks:
        if m >= avg:
            grades.append("A")
        elif m >= avg - 10:
            grades.append("B")
        else:
            grades.append("C")

    return grades

def analyze_marks(marks):
    print("Analyzing marks:", marks)

    pdb.set_trace()

    avg = compute_average(marks)
    high = highest_mark(marks)
    norm = normalize_marks(marks)
    grades = grade_students(marks)

    return {
        "average": avg,
        "highest": high,
        "normalized": norm,
        "grades": grades
    }

if __name__ == "__main__":
    marks = [78, 82, 91, 67, 0]

    result = analyze_marks(marks)

    print("Final result:")
    for k, v in result.items():
        print(k, ":", v)
