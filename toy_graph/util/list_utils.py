
def partition(num: int,
              partitions: int):
    if partitions <= 1:
        return [num]

    remainder = num % partitions
    if remainder == 0:
        return [num // partitions] * partitions
    zp = partitions - num % partitions
    quotient = num // partitions
    partitioned_list = []
    for i in range(partitions):
        if i < zp:
            if quotient != 0:
                partitioned_list.append(quotient)
        else:
            partitioned_list.append(quotient+1)
    return partitioned_list



