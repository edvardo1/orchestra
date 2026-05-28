import sys
import os
import argparse
import array
import numpy as np

def convert_to_orchestra_csr_optimized(input_file, output_file, is_mtx=False, is_undirected=False):
    print(f"Reading graph from '{input_file}'... (Memory optimized with on-the-fly re-indexing)")

    # Dictionary to map huge string IDs (like Google+ 21-digit IDs) to contiguous 0-indexed integers
    node_map = {}
    next_node_id = 0
    
    # Efficient Parsing using Python's C-based array module for edges
    srcs = array.array('i')
    dsts = array.array('i')
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith(('#', '%', '\n')):
                if is_mtx and "symmetric" in line.lower():
                    print("Note: Detected symmetric Matrix Market format. Will treat as undirected.")
                    is_undirected = True
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                u_str, v_str = parts[0], parts[1]
                
                # Re-index nodes to contiguous integers on the fly
                if u_str not in node_map:
                    node_map[u_str] = next_node_id
                    next_node_id += 1
                if v_str not in node_map:
                    node_map[v_str] = next_node_id
                    next_node_id += 1
                
                # Append the small mapped integer, avoiding OverflowError
                srcs.append(node_map[u_str])
                dsts.append(node_map[v_str])

    # Free the mapping dictionary from memory as we no longer need it
    del node_map

    # Convert to NumPy arrays for lightning-fast vectorized operations
    u = np.frombuffer(srcs, dtype=np.int32)
    v = np.frombuffer(dsts, dtype=np.int32)
    
    # Free the original arrays to clear up RAM immediately
    del srcs
    del dsts
        
    if is_undirected:
        print("Duplicating edges for undirected graph...")
        # Avoid duplicating self-loops
        mask = u != v
        u_rev = v[mask]
        v_rev = u[mask]
        
        u = np.concatenate([u, u_rev])
        v = np.concatenate([v, v_rev])

    total_edges = len(u)
    total_nodes = next_node_id
    
    print(f"Parsed {total_nodes} unique nodes and {total_edges} edges.")
    
    # Vectorized CSR Construction
    print("Sorting edges and computing degrees...")
    
    # Sort primarily by source node (u) to align edges for CSR
    sort_idx = np.argsort(u)
    u_sorted = u[sort_idx]
    v_sorted = v[sort_idx]
    
    # Clean up unsorted arrays from memory
    del u, v, sort_idx 
    
    # Compute degrees and row pointers using vectorized bincount
    degrees = np.bincount(u_sorted, minlength=total_nodes)
    edge_start_idx = np.zeros(total_nodes, dtype=np.int64)
    edge_start_idx[1:] = np.cumsum(degrees)[:-1]
            
    max_degree = int(degrees.max())
    start_node = int(degrees.argmax())
            
    print(f"Selected Start Node: {start_node} (Degree: {max_degree})")
    print(f"Writing to '{output_file}' in chunks to prevent I/O freezing...")

    # Chunked File Writing
    CHUNK_SIZE = 500_000  # Batch size for disk writing

    with open(output_file, 'w') as f:
        # Write Header
        f.write(f"{total_nodes} {total_edges} {start_node}\n")
        
        # Write Nodes (Row Pointers: EdgeStartIndex, NumberOfEdges)
        print("Writing node array...")
        for i in range(0, total_nodes, CHUNK_SIZE):
            chunk_start = edge_start_idx[i:i+CHUNK_SIZE]
            chunk_deg = degrees[i:i+CHUNK_SIZE]
            # Fast string formatting for the batch
            f.writelines(f"{s} {d}\n" for s, d in zip(chunk_start, chunk_deg))
            
        # Write Edges (Column Indices: DestNodeIdx, DummyWeight)
        print("Writing edge array...")
        for i in range(0, total_edges, CHUNK_SIZE):
            chunk_v = v_sorted[i:i+CHUNK_SIZE]
            f.writelines(f"{dest} 0\n" for dest in chunk_v)

    print("Conversion complete! Ready for GPU processing. 🚀")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert standard graphs to Orchestra CSR format.")
    parser.add_argument("input", help="Path to the input graph file (.txt, .mtx, .tsv)")
    parser.add_argument("output", help="Path for the output .dat file")
    parser.add_argument("--mtx", action="store_true", help="Flag to indicate input is 1-indexed Matrix Market format")
    parser.add_argument("--undirected", action="store_true", help="Flag to duplicate edges (u->v and v->u)")
    
    args = parser.parse_args()
    
    # Auto-detect mtx extension
    if args.input.endswith('.mtx'):
        args.mtx = True
        
    convert_to_orchestra_csr_optimized(args.input, args.output, is_mtx=args.mtx, is_undirected=args.undirected)