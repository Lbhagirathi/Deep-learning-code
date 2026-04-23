#!/usr/bin/env python3

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(message)s"
)

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler("bank.log")
# file_handler.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

# formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

#if logger.hasHandlers():
    #logger.handlers.clear()

# import sys

# logging.StreamHandler(sys.stdout)  # normal output
# logging.StreamHandler(sys.stderr)  # error output (default)

# from logging.handlers import RotatingFileHandler

# handler = RotatingFileHandler(
#     "bank.log", maxBytes=10000, backupCount=3
# )
# logger.addHandler(handler)

# from logging.handlers import TimedRotatingFileHandler

# handler = TimedRotatingFileHandler(
#     "bank.log", when="midnight", interval=1
# )
# logger.addHandler(handler)

# import json

# class JSONHandler(logging.FileHandler):
#     def emit(self, record):
#         log_entry = {
#             "time": self.formatTime(record),
#             "level": record.levelname,
#             "msg": record.msg
#         }
#         self.stream.write(json.dumps(log_entry) + "\n")

logging.info("Bank application started")

class BankAccount:
    def __init__(self, account_no, owner, balance):
        self.account_no = account_no
        self.owner = owner
        self.balance = balance
        logging.debug(f"Created account {account_no} for {owner}")

    def deposit(self, amount):
        logging.debug(f"Attempting deposit: {amount}")

        if amount <= 0:
            logging.error("Deposit amount must be positive")
            raise ValueError("Invalid deposit amount")

        self.balance += amount
        logging.info(f"Deposited {amount}. New balance: {self.balance}")

    def withdraw(self, amount):
        logging.debug(f"Attempting withdrawal: {amount}")

        if amount <= 0:
            logging.error("Withdrawal amount must be positive")
            raise ValueError("Invalid withdrawal amount")

        if amount > self.balance:
            logging.warning(
                f"Withdrawal failed for account {self.account_no}: "
                f"Insufficient balance"
            )
            raise ValueError("Insufficient balance")

        self.balance -= amount
        logging.info(f"Withdrawn {amount}. New balance: {self.balance}")

    def get_balance(self):
        logging.debug(f"Balance checked for account {self.account_no}")
        return self.balance


def transfer(source, target, amount):
    logging.info(
        f"Initiating transfer of {amount} "
        f"from {source.account_no} to {target.account_no}"
    )

    try:
        source.withdraw(amount)
        target.deposit(amount)

    except ValueError as e:
        logging.error(f"Transfer failed: {e}")
        raise

    else:
        logging.info("Transfer completed successfully")

    finally:
        logging.debug("Transfer operation ended")

if __name__ == "__main__":
    try:
        acc1 = BankAccount(101, "Alice", 1000)
        acc2 = BankAccount(102, "Bob", 500)

        acc1.deposit(200)
        acc1.withdraw(150)

        transfer(acc1, acc2, 300)

        transfer(acc1, acc2, 5000)  # should fail

    except Exception as e:
        logging.critical(f"Unexpected error: {e}")

    logging.info("Bank application terminated")
