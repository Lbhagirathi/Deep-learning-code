#!/usr/bin/env python3

class InsufficientBalanceError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Balance {balance} is insufficient for withdrawal of {amount}.")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientBalanceError(balance, amount)
    return balance - amount

try:
  rem = withdraw(10, 30)
  print(f'Remaining amount {rem}')

except InsufficientBalanceError as e:
  print("Error while withdrawing money: ", e)
