import random
... walks=[]
... def random_walk(steps=100):
...     for i in range(0,steps):
...         r=random.choice([-1,1])
...         walks.append(r)
...     return walks
>>> random_walk()
