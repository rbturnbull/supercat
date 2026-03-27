from pathlib import Path
import torch

def generate_intervals(n:int, k:int) -> list[tuple[int, int]]:
    """
    Generate intervals of size `k` that fit within a range of size `n`.

    Args:
        n (int): The size of the range.
        k (int): The size of each interval.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the intervals. Each tuple contains the start and end of an interval.
        If `k` is greater than `n`, returns an empty list.

    Raises:
        ValueError: If `k` is larger than `n`, a ValueError is raised.

    Examples:
        >>> generate_intervals(10, 2)
        [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

        >>> generate_intervals(10, 3)
        [(0, 3), (3, 6), (6, 9)]

        >>> generate_intervals(5, 6)
        []
    """
    if k > n or k == 0 or n == 0:
        return []

    # Calculate number of full intervals that fit into n
    num_intervals = n // k
    
    # Calculate remaining space (gap) after placing intervals
    total_gap = n - (num_intervals * k)
    
    # Distribute the gap equally at the start and end, with any remainder in the middle
    start_gap = total_gap // 2
    
    intervals = [(start_gap + i * k, start_gap + (i + 1) * k) for i in range(num_intervals)]
    
    return intervals


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


def write_image(data:torch.Tensor, path:Path|str) -> None:
    """
    Saves a 2D image or 3D volume data to the specified path in either .pt or image format.

    Args:
        data (torch.Tensor): The 3D tensor data to be saved.
        path (Path | str): The destination file path where the data will be saved. If the suffix is
            '.pt', the data will be saved as a PyTorch tensor. Otherwise, it will be saved as an image file.

    Returns:
        None

    Notes:
        - If the directory specified in the path does not exist, it will be created.
        - Supports saving in '.pt' format for PyTorch tensors. Other formats will
          be saved using `io.imsave` from skimage.io after converting the tensor to a NumPy array.
    """
    path = Path(path)
    print(f"Writing to {path}")
    path.parent.mkdir(exist_ok=True, parents=True)
    match path.suffix.lower():
        case ".nrrd":
            import nrrd
            import nrrd.writer

            # Add new type mappings
            nrrd.writer._TYPEMAP_NUMPY2NRRD.update({
                'f2': 'float16',
            })
            nrrd.write(str(path), data.numpy())
        case ".pt":
            torch.save(data, path)
        case '.tif':
            import numpy as np
            import tifffile as tiff

            if isinstance(data, torch.Tensor):
                data = data.numpy()
            
            # Clip data to valid range
            data = np.clip(data, -1.0, 1.0)
            
            # Save output
            data = (data + 1.0) * 255.0/2.0
            data = data.astype(np.uint8)
            tiff.imwrite(str(path), data, dtype=np.uint8)
        case _:
            from skimage import io
            import numpy as np
            
            if isinstance(data, torch.Tensor):
                data = data.numpy()

            # Clip data to valid range
            data = np.clip(data, -1.0, 1.0)
            
            # Save output
            data = (data + 1.0) * 255.0/2.0
            data = data.astype(np.uint8)
            io.imsave(path, data)    



