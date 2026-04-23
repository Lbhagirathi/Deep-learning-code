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
            try:

                roll, name, marks = line.strip().split(",")
                records[int(roll)] = {
                    "name": name,
                    "marks": int(marks)
                }
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")

    except PermissionError:
        print("Permission denied")

    except IndexError:
        print("Missing fields in a line.")

    except FileNotFoundError:
        print("Data file not found. Starting with empty records.")

    except ValueError:
        print("Corrupted data in file.")

    except TypeError:
        print("Type error occurred.")
    
    except IOError:
        print("Error while reading the file.")

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
            try:

                s = records[roll]
                f.write(f"{roll},{s['name']},{s['marks']}\n")
            except KeyError:
                print("Missing 'name' or 'marks' in record.")

            except TypeError:
                print("Invalid data type in records.")

    except IOError:
        print("Error while writing to file.")

    except PermissionError:
        print("Permission denied while writing to file.")
    
    except ValueError:
        print("Invalid value encountered.")

    finally:
        try:
            f.close()
        except Exception:
            pass

def add_student(records):
    try:
        roll = int(input("Enter roll number: "))
        if roll in records:
            print("Student already exists.")
            return

        name = input("Enter name: ")
        marks = int(input("Enter marks: "))

        if marks < 0 or marks > 100:
            print("Marks should be between 0 and 100.")
            return

        records[roll] = {"name": name, "marks": marks}
        print("Student added successfully.")

    except ValueError:
        print("Invalid input. Roll number and marks must be integers.")

    except TypeError:
        print("Unexpected data type error.")

def display_student(records):
    if not records:
        print("No records available.")
        return

    try:
        roll = int(input("Enter roll number: "))
        student = records[roll]
        print("Name:", student["name"])
        print("Marks:", student["marks"])

    except ValueError:
        print("Roll number must be an integer.")

    except KeyError:
        print("Student not found.")

    except TypeError:
        print("Invalid records data.")

def class_average(records):
    if not records:
        print("No students in the class.")
        return

    total = 0
    count = 0

    for roll, s in records.items():
        try:
            marks = int(s["marks"])
            total += marks
            count += 1
        except KeyError:
            print(f"Skipping {roll}: missing marks")
        except ValueError:
            print(f"Skipping {roll}: invalid marks")
        except TypeError:
            print(f"Skipping {roll}: bad data format")

    try:
        avg = total / count
        print("Class average:", avg)
    except ZeroDivisionError:
        print("No valid student records to calculate average.")

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
            
        except KeyboardInterrupt:
            print("\nInterrupted! Saving data before exit...")
            save_records(records)
            break

        except EOFError:
            print("\nInput ended unexpectedly. Saving data...")
            save_records(records)
            break

        except Exception as e:
            print("Unexpected error:", e)