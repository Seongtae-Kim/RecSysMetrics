
# Seongtae Kim / 2022-07-26

def sqrt(n):
    return n ** .5

def log(x): # Using Taylor Series
    n = 1000000.
    return n * ((x ** (1/n)) - 1)

def log2(x):
    return log(x) / log(2)
