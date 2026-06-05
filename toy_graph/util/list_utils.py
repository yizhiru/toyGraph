def partition(num: int, partitions: int) -> list[int]:
    if partitions <= 1:
        return [num]
    base, remainder = divmod(num, partitions)
    return [
        size
        for i in range(partitions)
        if (size := base + (1 if i < remainder else 0)) > 0
    ]
