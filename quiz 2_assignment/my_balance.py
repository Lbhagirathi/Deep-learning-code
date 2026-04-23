class InsufficientBalanceError(Exception):
    def __init__(self,balance, amount):
        self.balance=balance
        self.amount=amount
        super().__init__("Balance {} is insufficient for withdrawal of amount {}.".format(balance, amount))


def withdraw(balance,amount):
    if amount>balance:
        raise InsufficientBalanceError(balance,amount)
    return balance-amount

try:
    rem=withdraw(10,30)
    print("Remaining amount {}".format(rem))

except InsufficientBalanceError as e:
    print("error while withdrawing",e)




