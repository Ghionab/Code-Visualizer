#!/usr/bin/env python3
"""
Example with loops and conditionals for interactive stepping.
"""

def fibonacci(n):
    """Generate fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        next_fib = fib[i-1] + fib[i-2]
        fib.append(next_fib)
    
    return fib

def main():
    print("Fibonacci sequence generator")
    
    # Generate first 8 fibonacci numbers
    sequence = fibonacci(8)
    
    print("First 8 fibonacci numbers:")
    for i, num in enumerate(sequence):
        print(f"F({i}) = {num}")

if __name__ == "__main__":
    main()