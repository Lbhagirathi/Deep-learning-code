#!/usr/bin/env python3

try:
    with open("data.txt", "r") as f:
        content = f.read()
        number = int(content.strip())   # May raise ValueError
        result = 100 / number           # May raise ZeroDivisionError

except FileNotFoundError:
    print("Error: File not found.")

except ValueError:
    print("Error: File does not contain a valid integer.")

except ZeroDivisionError:
    print("Error: Cannot divide by zero.")

else:
    print("File read successfully.")
    print("Result is:", result)

finally:
    print("Execution complete.")
