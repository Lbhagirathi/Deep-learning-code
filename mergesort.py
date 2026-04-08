#!/usr/bin/env python3

def merge_sort(arr):
    _merge_sort(arr, 0, len(arr) - 1)

def _merge_sort(arr, left, right):
    if left >= right:
        return

    mid = (left + right) // 2

    _merge_sort(arr, left, mid)
    _merge_sort(arr, mid + 1, right)

    merge(arr, left, mid, right)

def merge(arr, left, mid, right):
    i = left       # left subarray
    j = mid + 1    # right subarray
    temp = []

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp.append(arr[i])
            i += 1
        else:
            temp.append(arr[j])
            j += 1

    while i <= mid:
        temp.append(arr[i])
        i += 1

    while j <= right:
        temp.append(arr[j])
        j += 1

    for i, idx in enumerate(range(left, right + 1)):
        arr[idx] = temp[i]

if __name__ == "__main__":
    arr = [5, 2, 9, 1]
    merge_sort(arr)
    print(arr)
