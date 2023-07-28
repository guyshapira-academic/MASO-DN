from typing import List


def dec_to_bin(dec: int) -> List[int]:
    """
    Converts from decimal int to binary list.

    Parameters:
        dec (int): Decimal integer to be converted.
    """
    bin_list = []
    while dec > 0:
        bin_list.append(dec % 2)
        dec = dec // 2
    return bin_list[::-1]
