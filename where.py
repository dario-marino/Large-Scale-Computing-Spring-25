import re
import numpy as np

def parse_optimization_log(file_path):
    """
    Parse the optimization log file and return a vector of 128 elements
    showing the last processed block pair for each rank (core).
    
    Args:
        file_path (str): Path to the .opt log file
    
    Returns:
        numpy.ndarray: Vector of 128 elements where each element is the last
                      block pair number processed by that rank, or -1 if no
                      processing was recorded for that rank
    """
    # Initialize progress vector with -1 (indicating no progress recorded)
    progress_vector = np.full(128, -1, dtype=int)
    
    # Regular expression to match processing lines
    # Matches: "Rank X: Processing block pair Y/Z (a,b)"
    pattern = r'Rank (\d+): Processing block pair (\d+)/\d+'
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    rank = int(match.group(1))
                    block_pair = int(match.group(2))
                    
                    # Update progress for this rank if it's within valid range
                    if 0 <= rank < 128:
                        progress_vector[rank] = block_pair
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return progress_vector

def print_progress_summary(progress_vector):
    """
    Print a summary of the progress for each rank.
    """
    if progress_vector is None:
        return
    
    print("Progress Summary:")
    print("-" * 50)
    
    active_ranks = []
    inactive_ranks = []
    
    for rank in range(128):
        if progress_vector[rank] != -1:
            active_ranks.append((rank, progress_vector[rank]))
            print(f"Rank {rank:3d}: Last processed block pair {progress_vector[rank]}")
        else:
            inactive_ranks.append(rank)
    
    print(f"\nActive ranks: {len(active_ranks)}")
    print(f"Inactive ranks: {len(inactive_ranks)}")
    
    if inactive_ranks:
        print(f"Inactive rank numbers: {inactive_ranks}")
    
    if active_ranks:
        max_progress = max(active_ranks, key=lambda x: x[1])
        min_progress = min(active_ranks, key=lambda x: x[1])
        print(f"\nHighest progress: Rank {max_progress[0]} at block pair {max_progress[1]}")
        print(f"Lowest progress: Rank {min_progress[0]} at block pair {min_progress[1]}")

# Example usage
if __name__ == "__main__":
    # Your specific file path
    file_path = r"C:\Users\dario\Downloads\patent-sim_opt.out"
    
    progress = parse_optimization_log(file_path)
    
    if progress is not None:
        print("Progress vector (128 elements):")
        print(progress)
        print("\nDetailed summary:")
        print_progress_summary(progress)
        
        # You can also save the vector to a file for later use
        np.save("progress_checkpoint.npy", progress)
        print("\nProgress vector saved to 'progress_checkpoint.npy'")