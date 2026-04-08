#!/usr/bin/env python3

def quick_sort(arr):
    _quick_sort(arr, 0, len(arr) - 1)

def _quick_sort(arr, low, high):
    if low < high:
        p = partition(arr, low, high)
        _quick_sort(arr, low, p - 1)
        _quick_sort(arr, p + 1, high)

def partition(arr, low, high):
    pivot = arr[high]   # choose last element as pivot
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

if __name__ == "__main__":
    arr = [9,3,1,2,6,8,4,5,7]
    quick_sort(arr)
    print(arr)
