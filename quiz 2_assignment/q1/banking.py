#!/usr/bin/python3

# 1. Identifylocations where exceptions may occur.
# 2. Modify the program to handle exception identified in step (1).
# 3. Ensure the program continues execution even after errors.
# 4. Implement at least one custom exception.

# 5. Develop test cases for thoroughly testing all operations.

def load_accounts(filename):
    accounts = {}
    try:
        file = open(filename, "r")
        #permission error, file not found error

        for line in file:
            try:
                parts = line.strip().split(",")
                acc_no = parts[0]
                name = parts[1]
                #value error, index error
                balance = float(parts[2])
            except ValueError:
                print("Invalid balance for account", acc_no)

            except IndexError:
                print("Invalid line format:", line)
            
                accounts[acc_no] = {
                    "name": name,
                    "balance": balance,
                    "transactions": []
                }

        file.close()
    except FileNotFoundError:
        print("File not found. Starting with empty accounts.")
    except PermissionError:
        print("Permission denied.")
    except Exception as e:
        print("Error processing line:", line, "Error:", str(e))
    return accounts

def save_accounts(filename, accounts):
    try:
        file = open(filename, "w")
    #permission error, file not found error
    except FileNotFoundError:
        print("File not found. Starting with empty accounts.")
    except PermissionError:
        print("Permission denied.")
    except Exception as e:
        print("Error opening file:", str(e))
        return

    for acc in accounts:
        try:
            acc_data = accounts[acc]
            line = acc + "," + acc_data["name"] + "," + str(acc_data["balance"])
            file.write(line + "\n")
        except Exception as e:
            print("Error writing account", acc, "Error:", str(e))

    file.close()

def create_account(accounts):
    try:
        acc_no = input("Enter account number: ")
        name = input("Enter account holder name: ")
        balance = float(input("Enter initial deposit: "))

        if balance<0:
            raise ValueError("Balance invalid")

        accounts[acc_no] = {
            "name": name,
            "balance": balance,
            "transactions": []
        }

        print("Account created")
    except ValueError as e:
        print("Invalid input:", e)


def deposit(accounts):
    try:
        acc_no = input("Enter account number: ")
        if acc_no not in accounts:
            raise KeyError("Account not found")
        
        amount = float(input("Enter amount to deposit: "))
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        account = accounts[acc_no]
        account["balance"] += amount
        account["transactions"].append(("deposit", amount))
        print("Deposit successful")

    except ValueError as e:
        print("Invalid input:", e)

    except KeyError as e:
        print("Error:", e)

class InsufficientFundsError(Exception):
    pass

def withdraw(accounts):
    try:
        acc_no = input("Enter account number: ")
        amount = float(input("Enter withdrawal amount: "))
        account = accounts[acc_no]

        if acc_no not in accounts:
            print("Account not found")
            return

        if account["balance"]<amount:
            raise InsufficientFundsError("insufficient balance")
        
        if account["balance"] >= amount:
            account["balance"] -= amount
            account["transactions"].append(("withdraw", amount))
            print("Withdrawal successful")
        else:
            print("Insufficient balance")

    except InsufficientFundsError as e:
        print("Error:", e)

def transfer(accounts):
    try:
        source = input("Enter source account: ")
        target = input("Enter target account: ")
        amount = float(input("Enter amount: "))

        if source not in accounts or target not in accounts:
            raise KeyError("Either one or both not present")

        src_acc = accounts[source]
        tgt_acc = accounts[target]

        if src_acc["balance"] >= amount:
            src_acc["balance"] -= amount
            tgt_acc["balance"] += amount
            src_acc["transactions"].append(("transfer_out", amount))
            tgt_acc["transactions"].append(("transfer_in", amount))
            print("Transfer successful")
        else:
            print("Insufficient funds")
    
    except KeyError as e:
        print("Error:", e)
    except InsufficientFundsError as e:
        print("Error:", e)


def display_account(accounts):
    acc_no = input("Enter account number: ")
    account = accounts[acc_no]
    if acc_no not in accounts:
        print("Account not found")
        return

    print("Account number:", acc_no)
    print("Name:", account["name"])
    print("Balance:", account["balance"])

def transaction_history(accounts):
    acc_no = input("Enter account number: ")
    account = accounts[acc_no]
    if acc_no not in accounts:
        print("Account not found")
        return

    print("Transaction history")
    for t in account["transactions"]:
        print(t[0], t[1])

def apply_interest(accounts):
    rate = float(input("Enter interest rate (%): "))

    for acc in accounts:
        account = accounts[acc]
        interest = account["balance"] * rate / 100
        account["balance"] += interest

    print("Interest applied to all accounts")

def richest_account(accounts):
    max_balance = -1
    richest = None

    for acc in accounts:
        bal = accounts[acc]["balance"]

        if bal > max_balance:
            max_balance = bal
            richest = acc

    print("Richest account:", richest)
    print("Balance:", max_balance)

def total_bank_balance(accounts):
    total = 0
    for acc in accounts:
        total += accounts[acc]["balance"]

    print("Total bank balance:", total)

def remove_account(accounts):
    acc_no = input("Enter account number to delete: ")
    if acc_no not in accounts:
        print("Account not found")
        return
    del accounts[acc_no]
    print("Account deleted")

def menu():
    print()
    print("Bank Management System")
    print("----------------------")
    print("1. Create account")
    print("2. Deposit")
    print("3. Withdraw")
    print("4. Transfer")
    print("5. Display account")
    print("6. Transaction history")
    print("7. Apply interest")
    print("8. Richest account")
    print("9. Total bank balance")
    print("10. Remove account")
    print("11. Save data")
    print("12. Exit")

def main():
    filename = input("Enter account file: ")
    accounts = load_accounts(filename)

    while True:
        menu()
        choice = int(input("Enter choice: "))
        try:
            choice = int(input("Enter choice: "))
        except ValueError:
            print("Invalid input")
            continue
        if choice == 1:
            create_account(accounts)

        elif choice == 2:
            deposit(accounts)

        elif choice == 3:
            withdraw(accounts)

        elif choice == 4:
            transfer(accounts)

        elif choice == 5:
            display_account(accounts)

        elif choice == 6:
            transaction_history(accounts)

        elif choice == 7:
            apply_interest(accounts)

        elif choice == 8:
            richest_account(accounts)

        elif choice == 9:
            total_bank_balance(accounts)

        elif choice == 10:
            remove_account(accounts)

        elif choice == 11:
            save_accounts(filename, accounts)

        elif choice == 12:
            break

        else:
            print("Invalid choice")


# main()


# Manual unit tests for bank system

# Assume all your functions are already imported here

# def test_create_account():
#     accounts = {}

#     # Simulate
#     acc_no = "101"
#     accounts[acc_no] = {"name": "Alice", "balance": 1000, "transactions": []}

#     assert acc_no in accounts
#     assert accounts[acc_no]["balance"] == 1000

#     print("✅ test_create_account passed")


# def test_deposit():
#     accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

#     accounts["101"]["balance"] += 500

#     assert accounts["101"]["balance"] == 1500

#     print("✅ test_deposit passed")


# def test_withdraw():
#     accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

#     accounts["101"]["balance"] -= 300

#     assert accounts["101"]["balance"] == 700

#     print("✅ test_withdraw passed")


# def test_withdraw_insufficient():
#     accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

#     try:
#         if accounts["101"]["balance"] < 5000:
#             raise Exception("Insufficient funds")
#     except:
#         pass

#     assert accounts["101"]["balance"] == 1000

#     print("✅ test_withdraw_insufficient passed")


# def test_transfer():
#     accounts = {
#         "101": {"name": "Alice", "balance": 1000, "transactions": []},
#         "102": {"name": "Bob", "balance": 500, "transactions": []}
#     }

#     amount = 200
#     accounts["101"]["balance"] -= amount
#     accounts["102"]["balance"] += amount

#     assert accounts["101"]["balance"] == 800
#     assert accounts["102"]["balance"] == 700

#     print("✅ test_transfer passed")


# def test_apply_interest():
#     accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

#     rate = 10
#     accounts["101"]["balance"] += accounts["101"]["balance"] * rate / 100

#     assert accounts["101"]["balance"] == 1100

#     print("✅ test_apply_interest passed")


# def test_total_balance():
#     accounts = {
#         "101": {"name": "Alice", "balance": 1000, "transactions": []},
#         "102": {"name": "Bob", "balance": 500, "transactions": []}
#     }

#     total = sum(acc["balance"] for acc in accounts.values())

#     assert total == 1500

#     print("✅ test_total_balance passed")


# def test_remove_account():
#     accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

#     del accounts["101"]

#     assert "101" not in accounts

#     print("✅ test_remove_account passed")


# def test_richest_account():
#     accounts = {
#         "101": {"name": "Alice", "balance": 1000, "transactions": []},
#         "102": {"name": "Bob", "balance": 2000, "transactions": []}
#     }

#     richest = max(accounts, key=lambda x: accounts[x]["balance"])

#     assert richest == "102"

#     print("✅ test_richest_account passed")


# # Run all tests
# if __name__ == "__main__":
#     mode = input("Enter 'run' or 'test': ")

#     if mode == "run":
#         main()

#     elif mode == "test":
#         test_create_account()
#         test_deposit()
#         test_withdraw()
#         test_withdraw_insufficient()
#         test_transfer()
#         test_apply_interest()
#         test_total_balance()
#         test_remove_account()
#         test_richest_account()

#         print("\n🎉 All tests passed!")

#     else:
#         print("Invalid option")

# ---------------- TESTING ---------------- #

def mock_inputs(inputs):
    """Generator to simulate input()"""
    def input_mock(_):
        return inputs.pop(0)
    return input_mock


def test_create_account():
    accounts = {}

    inputs = ["101", "Alice", "1000"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    create_account(accounts)

    input = input_backup

    assert "101" in accounts
    assert accounts["101"]["balance"] == 1000
    print("✅ test_create_account passed")


def test_deposit():
    accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

    inputs = ["101", "500"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    deposit(accounts)

    input = input_backup

    assert accounts["101"]["balance"] == 1500
    print("✅ test_deposit passed")


def test_withdraw():
    accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

    inputs = ["101", "300"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    withdraw(accounts)

    input = input_backup

    assert accounts["101"]["balance"] == 700
    print("✅ test_withdraw passed")


def test_withdraw_insufficient():
    accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

    inputs = ["101", "5000"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    withdraw(accounts)

    input = input_backup

    assert accounts["101"]["balance"] == 1000
    print("✅ test_withdraw_insufficient passed")


def test_transfer():
    accounts = {
        "101": {"name": "Alice", "balance": 1000, "transactions": []},
        "102": {"name": "Bob", "balance": 500, "transactions": []}
    }

    inputs = ["101", "102", "200"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    transfer(accounts)

    input = input_backup

    assert accounts["101"]["balance"] == 800
    assert accounts["102"]["balance"] == 700
    print("✅ test_transfer passed")


def test_apply_interest():
    accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

    inputs = ["10"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    apply_interest(accounts)

    input = input_backup

    assert accounts["101"]["balance"] == 1100
    print("✅ test_apply_interest passed")


def test_remove_account():
    accounts = {"101": {"name": "Alice", "balance": 1000, "transactions": []}}

    inputs = ["101"]
    global input
    input_backup = input
    input = mock_inputs(inputs)

    remove_account(accounts)

    input = input_backup

    assert "101" not in accounts
    print("✅ test_remove_account passed")


def test_richest_account():
    accounts = {
        "101": {"name": "Alice", "balance": 1000, "transactions": []},
        "102": {"name": "Bob", "balance": 2000, "transactions": []}
    }

    richest = max(accounts, key=lambda x: accounts[x]["balance"])

    assert richest == "102"
    print("✅ test_richest_account passed")


# Run tests
if __name__ == "__main__":
    mode = input("Enter 'run' or 'test': ")

    if mode == "run":
        main()

    elif mode == "test":
        test_create_account()
        test_deposit()
        test_withdraw()
        test_withdraw_insufficient()
        test_transfer()
        test_apply_interest()
        test_remove_account()
        test_richest_account()

        print("\n🎉 All tests passed!")

    else:
        print("Invalid option")