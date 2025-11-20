import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

class BinPackingInstance:
    """Represents a single bin packing instance."""
    
    def __init__(self, name: str, capacity: int, items: List[int], optimal: int = None):
        self.name = name
        self.capacity = capacity
        self.items = np.array(items)
        self.n_items = len(items)
        self.optimal = optimal  # Known optimal solution if available
    
    def __repr__(self):
        return f"BinPackingInstance(name={self.name}, n_items={self.n_items}, capacity={self.capacity}, optimal={self.optimal})"


def load_beasley_format(filepath: str) -> List[BinPackingInstance]:
    """
    Load bin packing instances from OR-Library Beasley format.
    
    Format:
    - Line 1: number of instances in file
    - For each instance:
      - Line: instance name
      - Line: capacity n_items optimal_bins
      - Next n_items lines: item sizes
    """
    instances = []
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # First line is number of instances
    n_instances = int(lines[0])
    i = 1
    
    for _ in range(n_instances):
        # Skip empty lines
        while i < len(lines) and not lines[i]:
            i += 1
        
        # Read instance name
        name = lines[i].strip()
        i += 1
        
        # Read capacity, n_items, optimal
        parts = lines[i].split()
        capacity = int(float(parts[0]))  # Handle both int and float formats
        n_items = int(parts[1])
        optimal = int(float(parts[2])) if len(parts) > 2 else None
        i += 1
        
        # Read item sizes
        items = []
        for _ in range(n_items):
            items.append(int(float(lines[i])))  # Handle both int and float formats
            i += 1
        
        instances.append(BinPackingInstance(name, capacity, items, optimal))
    
    return instances


def load_all_instances(data_dir: str = "data/raw") -> Dict[str, List[BinPackingInstance]]:
    """Load all bin packing instances from the data directory."""
    data_path = Path(data_dir)
    all_instances = {}
    
    for file in sorted(data_path.glob("binpack*.txt")):
        instances = load_beasley_format(str(file))
        all_instances[file.stem] = instances
        print(f"Loaded {len(instances)} instances from {file.name}")
    
    return all_instances