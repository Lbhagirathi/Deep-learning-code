#!/usr/bin/env python3 

DATA_FILE = "students.txt"

def load_records():
    """
    Load student records from a file.
    Each line: roll,name,marks
    """
    records = {}

    try:
        f = open(DATA_FILE, "r")
        for line in f:
            roll, name, marks = line.strip().split(",")
            records[int(roll)] = {
                "name": name,
                "marks": int(marks)
            }

    except FileNotFoundError:
        print("Data file not found. Starting with empty records.")

    except ValueError:
        print("Corrupted data in file.")

    finally:
        try:
            f.close()
        except Exception:
            pass

    return records

def save_records(records):
    try:
        f = open(DATA_FILE, "w")
        for roll in records:
            s = records[roll]
            f.write(f"{roll},{s['name']},{s['marks']}\n")

    except IOError:
        print("Error while writing to file.")

    finally:
        try:
            f.close()
        except Exception:
            pass

def add_student(records):
    roll = int(input("Enter roll number: "))
    name = input("Enter name: ")
    marks = int(input("Enter marks: "))
    records[roll] = {"name": name, "marks": marks}
    print("Student added successfully.")

def display_student(records):
    try:
        roll = int(input("Enter roll number: "))
        student = records[roll]
        print("Name:", student["name"])
        print("Marks:", student["marks"])

    except ValueError:
        print("Roll number must be an integer.")

    except KeyError:
        print("Student not found.")

def class_average(records):
    try:
        total = 0
        for s in records.values():
            total += s["marks"]

        avg = total / len(records)
        print("Class average:", avg)

    except ZeroDivisionError:
        print("No students in the class.")

def menu():
    print("\n1. Add student")
    print("2. Display student")
    print("3. Class average")
    print("4. Save and exit")

if __name__ == "__main__":
    records = load_records()

    while True:
        menu()
        try:
            choice = int(input("Enter choice: "))

            if choice == 1:
                add_student(records)

            elif choice == 2:
                display_student(records)

            elif choice == 3:
                class_average(records)

            elif choice == 4:
                save_records(records)
                print("Data saved. Exiting.")
                break

            else:
                print("Invalid choice.")

        except ValueError:
            print("Please enter a valid menu number.")
