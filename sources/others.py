def isPrime(n): 
    # Corner case 
    if n <= 1: 
        return False
    # Check from 2 to n-1 
    for i in range(2, n): 
        if (n % i == 0): 
            return False
    return True

def isEven(n):
    if n%2 == 0:
        return True
    return False