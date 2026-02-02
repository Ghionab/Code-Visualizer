#!/usr/bin/env python3
"""
Simple mathematical operations example.
"""

def add_numbers(a, b):
    """Add two numbers and return the result."""
    result = a + b
    return result

def main():
    # Simple arithmetic
    x = 10
    y = 20
    sum_result = add_numbers(x, y)
    
    print(f"Adding {x} + {y} = {sum_result}")
    
    # List operations
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    print(f"Sum of {numbers} = {total}")

if __name__ == "__main__":
    main()