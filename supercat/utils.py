import torch


def generate_overlapping_intervals(total: int, interval_size: int, min_overlap: int, check:bool=True, variable_size:bool=False) -> list[tuple[int, int]]:
    """
    Creates a list of overlapping intervals within a specified range, adjusting the interval size to ensure
    that the overlap is approximately the same across all intervals.

    Args:
        total (int): The total range within which intervals are to be created.
        max_interval_size (int): The maximum size of each interval.
        min_overlap (int): The minimum number of units by which consecutive intervals overlap.
        check (bool): If True, checks are performed to ensure that the intervals meet the specified conditions.

    Returns:
        list[tuple[int, int]]: A list of tuples where each tuple represents the start (inclusive) 
        and end (exclusive) of an interval.

    Example:
        >>> generate_overlapping_intervals(20, 5, 2)
        [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
    """
    intervals = []
    start = 0

    if total == 0:
        return intervals
    
    max_interval_size = interval_size
    assert interval_size
    assert min_overlap is not None
    assert interval_size > min_overlap, f"Max interval size of {interval_size} must be greater than min overlap of {min_overlap}"

    # Calculate the number of intervals needed to cover the range
    num_intervals, remainder = divmod(total - min_overlap, interval_size - min_overlap)
    if remainder > 0:
        num_intervals += 1

    # Calculate the exact interval size to ensure consistent overlap
    overlap = min_overlap
    if variable_size:
        if num_intervals > 1:
            interval_size, remainder = divmod(total + (num_intervals - 1) * overlap, num_intervals)
            if remainder > 0:
                interval_size += 1
    else:
        # If the size is fixed, then vary the overlap to keep it even
        if num_intervals > 1:
            overlap, remainder = divmod( num_intervals * interval_size - total, num_intervals - 1)
            if overlap < min_overlap:
                overlap = min_overlap

    while True:
        end = start + interval_size
        if end > total:
            end = total
            start = max(end - interval_size,0)
        intervals.append((start, end))
        start += interval_size - overlap
        if end >= total:
            break

    if check:
        assert intervals[0][0] == 0
        assert intervals[-1][1] == total
        assert len(intervals) == num_intervals, f"Expected {num_intervals} intervals, got {len(intervals)}"

        assert interval_size <= max_interval_size, f"Interval size of {interval_size} exceeds max interval size of {max_interval_size}"
        for interval in intervals:
            assert interval[1] - interval[0] == interval_size, f"Interval size of {interval[1] - interval[0]} is not the expected size {interval_size}"

        for i in range(1, len(intervals)):
            overlap = intervals[i - 1][1] - intervals[i][0]
            assert overlap >= min_overlap, f"Min overlap condition of {min_overlap} not met for intervals {intervals[i - 1]} and {intervals[i]} (overlap {overlap})"

    return intervals


def distance_to_boundary(size_i:int, size_j:int, size_k:int) -> torch.Tensor:
    """
    Builds a weight tensor where each element represents the minimum distance 
    to any edge of a 3D volume of size (size_i, size_j, size_k).

    Args:
        size_i (int): The size of the first dimension.
        size_j (int): The size of the second dimension.
        size_k (int): The size of the third dimension.

    Returns:
        torch.Tensor: A tensor of shape (size_i, size_j, size_k) where each 
                      element is the minimum distance to any edge.
    """
    x = torch.arange(size_i).view(-1, 1, 1).expand(size_i, size_j, size_k)
    y = torch.arange(size_j).view(1, -1, 1).expand(size_i, size_j, size_k)
    z = torch.arange(size_k).view(1, 1, -1).expand(size_i, size_j, size_k)

    # Compute the distance to the closest edge for each coordinate
    distance_x = torch.minimum(x, size_i - 1 - x)
    distance_y = torch.minimum(y, size_j - 1 - y)
    distance_z = torch.minimum(z, size_k - 1 - z)

    # Combine the distances to get the minimum distance to any edge
    return torch.minimum(torch.minimum(distance_x, distance_y), distance_z)





