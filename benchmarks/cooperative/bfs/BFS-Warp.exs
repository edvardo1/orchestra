require Orchestra

# Orchestra.set_debug_logs(true)

defmodule CsrReader do
  @header_regex ~r/^(?<n1>\d+)\s+(?<n2>\d+)\s+(?<n3>\d+)$/
  @pairs_regex ~r/^(?<n1>\d+)\s+(?<n2>\d+)$/

  defp parse_pair!(line) do
    case Regex.named_captures(@pairs_regex, line) do
      nil ->
        raise "regex error: expected a pair, found: '#{line}'"

      regex_map ->
        {String.to_integer(regex_map["n1"]), String.to_integer(regex_map["n2"])}
    end
  end

  # processing header
  defp process_line(line, %{reading_state: :header} = map) do
    case Regex.named_captures(@header_regex, line) do
      nil ->
        raise "regex error: expected the header, found: '#{line}'"

      regex_map ->
        # IO.inspect(regex_map, label: "Found header")

        n1 = String.to_integer(regex_map["n1"])
        n2 = String.to_integer(regex_map["n2"])
        n3 = String.to_integer(regex_map["n3"])

        Map.merge(map, %{
          total_nodes: n1,
          remaining_nodes: n1,
          total_edges: n2,
          remaining_edges: n2,
          start_node: n3,
          reading_state: :nodes
        })
    end
  end

  # processing nodes
  defp process_line(line, %{reading_state: :nodes} = map) do
    {n1, n2} = parse_pair!(line)

    # IO.puts("header already processed, reading nodes")

    nodes = [[n1, n2] | map.nodes]
    remaining_nodes = map.remaining_nodes - 1

    %{
      map
      | nodes: nodes,
        remaining_nodes: remaining_nodes,
        reading_state:
          if remaining_nodes <= 0 do
            :edges
          else
            :nodes
          end
    }
  end

  # processing edges
  defp process_line(line, %{reading_state: :edges} = map) do
    {n1, n2} = parse_pair!(line)

    # IO.puts("header already processed, reading edges")

    edges = [[n1, n2] | map.edges]
    remaining_edges = map.remaining_edges - 1

    if remaining_edges < 0 do
      raise "error: found more edges than defined in the header!"
    end

    %{
      map
      | edges: edges,
        remaining_edges: remaining_edges
    }
  end

  def read_and_process_file(file_path) do
    initial_map = %{
      reading_state: :header,
      nodes: [],
      edges: []
    }

    final_map =
      file_path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.filter(fn l -> l != "" end)
      |> Enum.reduce(initial_map, &process_line/2)

    nodes_rev = Enum.reverse(final_map.nodes)
    edges_rev = Enum.reverse(final_map.edges)

    # Creating the map that will be returned
    %{
      start_node: final_map.start_node,
      total_nodes: final_map.total_nodes,
      total_edges: final_map.total_edges,
      nodes: Orchestra.tensor(nodes_rev, :s32),
      edges: Orchestra.tensor(edges_rev, :s32)
    }
  end
end

Orchestra.defmodule BFS do
  defk cpu_bfs_kernel(
         nodes,
         n_nodes,
         edges,
         frontier,
         frontier_size,
         new_frontier,
         atomic(next_slot),
         visited
       ) do
    tid = get_global_id(0)

    if tid >= frontier_size || tid >= n_nodes do
      return
    end

    # Get node to process in this thread
    node_idx = frontier[tid]
    # Mark it as visited
    visited[node_idx] = 1

    # -- Get node info --
    # Node edges index in edges array
    node_edges_idx = nodes[node_idx * 2 + 0]
    # Number of edges this node has
    node_num_edges = nodes[node_idx * 2 + 1]

    for i in range(node_edges_idx, node_edges_idx + node_num_edges) do
      # Getting child node
      dest_node_idx = edges[i * 2 + 0]

      # Check if this node was visited
      if visited[dest_node_idx] == 0 do
        # Mark as visited
        visited[dest_node_idx] = 1

        # Update empty slot and get the index we will use here
        idx = add_atomic_int(next_slot, 1)

        # Add this node to new frontier
        new_frontier[idx] = dest_node_idx
      end
    end
  end

  # This kernel is warp-centric. Here, 32 threads (a warp) will work together to process a single node and its edges.
  defk gpu_bfs_kernel(
         nodes,
         n_nodes,
         edges,
         frontier,
         frontier_size,
         new_frontier,
         atomic(next_slot),
         atomic(visited)
       ) do
    tid = get_global_id(0)
    lid = get_local_id(0)

    # --- WARP MATH ---
    # A standard NVIDIA warp has 32 threads
    warp_size = 32
    # Which node this warp is responsible for
    warp_id = tid / warp_size
    # Which "worker" this thread is within the team of 32
    # This is a modulo operation in Orchestra (%)
    lane_id = tid ~>> warp_size

    # Declare block-level buffers
    __local(local_buffer[2048])
    __local(shift[1])
    __atomic_local(local_free_idx[1])

    if lid == 0 do
      init_atomic_int(local_free_idx, 0)
      shift[0] = 0
    end

    __syncthreads()

    if warp_id < frontier_size && warp_id < n_nodes do
      # All 32 threads in the warp load the exact same node!
      node_idx = frontier[warp_id]

      # Get node info
      node_edges_idx = nodes[node_idx * 2 + 0]
      node_num_edges = nodes[node_idx * 2 + 1]

      # --- COOPERATIVE EDGE LOOP ---
      # Thread 0 starts at edge 0, Thread 1 at edge 1...
      i = node_edges_idx + lane_id
      end_idx = node_edges_idx + node_num_edges

      while i < end_idx do
        dest_node_idx = edges[i * 2 + 0]

        # TTAS Optimization
        was_visited = visited[dest_node_idx]

        if was_visited == 0 do
          was_visited = max_atomic_int(visited + dest_node_idx, 1)

          if was_visited == 0 do
            idx = add_atomic_int(local_free_idx, 1)

            if idx < 2048 do
              local_buffer[idx] = dest_node_idx
            else
              # Graceful Fallback (no nodes left behind)
              global_idx = add_atomic_int(next_slot, 1)
              new_frontier[global_idx] = dest_node_idx
            end
          end
        end

        # The whole warp steps forward by 32 to process the next batch of edges!
        i = i + warp_size
      end
    end

    __syncthreads()

    # --- FLUSH LOCAL BUFFER ---
    priv_buffer_size = load_atomic_int(local_free_idx)

    if priv_buffer_size > 2048 do
      priv_buffer_size = 2048
    end

    if lid == 0 do
      shift[0] = add_atomic_int(next_slot, priv_buffer_size)
    end

    __syncthreads()

    priv_shift = lid

    while priv_shift < priv_buffer_size do
      new_frontier[priv_shift + shift[0]] = local_buffer[priv_shift]
      priv_shift = priv_shift + get_local_size(0)
    end
  end

  def bfs(
        %{
          total_nodes: total_nodes,
          start_node: start_node,
          nodes: nodes_tensor,
          edges: edges_tensor
        } = nodes_map,
        cpu_limit,
        max_iterations \\ :infinity
      ) do
    frontier_size = 1

    start = System.monotonic_time()

    tensor_map = %{
      frontier: Orchestra.tensor({total_nodes}, :s32, fn _ -> start_node end),
      new_frontier: Orchestra.tensor({total_nodes}, :s32),
      visited: Orchestra.tensor({total_nodes}, :s32, fn _ -> 0 end),
      next_idx: Orchestra.tensor([0], :s32)
    }

    # ======================= GNx =======================
    tensor_map =
      Orchestra.with Orchestra.gpu() do
        Map.merge(tensor_map, %{
          frontier_gnx: Orchestra.new_gnx(tensor_map.frontier),
          new_frontier_gnx:
            Orchestra.new_gnx(
              Orchestra.get_shape(tensor_map.new_frontier),
              Orchestra.get_type(tensor_map.new_frontier)
            ),
          visited_gnx: Orchestra.new_gnx(tensor_map.visited),
          next_idx_gnx: Orchestra.new_gnx(tensor_map.next_idx),
          nodes_gnx: Orchestra.new_gnx(nodes_tensor),
          edges_gnx: Orchestra.new_gnx(edges_tensor)
        })
      end

    stop = System.monotonic_time()

    IO.puts(
      "Tensor creation took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms"
    )

    bfs_recursion(nodes_map, frontier_size, tensor_map, max_iterations, cpu_limit, :cpu, false)
  end

  @spec bfs_recursion(
          nodes_map :: map(),
          frontier_size :: integer(),
          tensor_map :: map(),
          max_iterations :: :infinity | integer(),
          cpu_limit :: integer(),
          last_device :: :cpu | :gpu,
          used_gpu :: boolean()
        ) :: boolean()
  # End BFS when frontier size goes to 0
  defp bfs_recursion(_map, 0, _tensor_map, _max_iterations, _cpu_limit, _last_device, used_gpu),
    do: used_gpu

  # End BFS when max_iterations becomes 0
  defp bfs_recursion(_map, _frontier_size, _tensor_map, 0, _cpu_limit, _last_device, used_gpu),
    do: used_gpu

  defp bfs_recursion(
         %{
           total_nodes: total_nodes,
           nodes: nodes_tensor,
           edges: edges_tensor
         } = nodes_map,
         frontier_size,
         %{
           frontier: frontier_tensor,
           new_frontier: new_frontier_tensor,
           visited: visited_tensor,
           next_idx: next_idx_tensor,
           frontier_gnx: frontier_gnx,
           new_frontier_gnx: new_frontier_gnx,
           visited_gnx: visited_gnx,
           next_idx_gnx: next_idx_gnx,
           nodes_gnx: nodes_gnx,
           edges_gnx: edges_gnx
         } = tensor_map,
         max_iterations,
         cpu_limit,
         last_device,
         used_gpu
       ) do
    {tensor_map, current_device, used_gpu} =
      if frontier_size > cpu_limit do
        # ----- Running on GPU -----

          if last_device == :cpu do
            # If we are switching from CPU to GPU, we need to copy the frontier and visited tensors
            Orchestra.write_gnx(frontier_gnx, frontier_tensor, frontier_size)
            Orchestra.write_gnx(visited_gnx, visited_tensor, nil)
          end

        Orchestra.with Orchestra.gpu() do
          # We have to zero out the next_idx tensor on the GPU before each iteration.
          # The 'next_idx_tensor' will always be zero before every iteration, so we can just copy it.
          Orchestra.write_gnx(next_idx_gnx, next_idx_tensor, 1)

          # We now need 32 threads for every single node in the frontier
          total_threads_needed = frontier_size * 32
          threads_per_block = 128

          num_blocks = div(total_threads_needed + threads_per_block - 1, threads_per_block)

          Orchestra.spawn(
            &BFS.gpu_bfs_kernel/8,
            {num_blocks},
            {threads_per_block},
            [
              nodes_gnx,
              total_nodes,
              edges_gnx,
              frontier_gnx,
              frontier_size,
              new_frontier_gnx,
              next_idx_gnx,
              visited_gnx
            ]
          )

          # After GPU execution, only take the next_idx tensor back
          # We will leave the heavy boys on the GPU
          Orchestra.get_gnx(next_idx_gnx, next_idx_tensor)

          {tensor_map, :gpu, true}
        end
      else
        # ----- Running on CPU -----

        if last_device == :gpu do
          # If we are switching from GPU to CPU, we need to copy the frontier and visited tensors back to the CPU
          Orchestra.with Orchestra.gpu() do
            Orchestra.get_gnx(frontier_gnx, frontier_tensor, frontier_size)
            Orchestra.get_gnx(visited_gnx, visited_tensor)
          end
        end

        Orchestra.with Orchestra.cpu() do
          Orchestra.spawn(
            &BFS.cpu_bfs_kernel/8,
            {frontier_size},
            {0},
            [
              nodes_tensor,
              total_nodes,
              edges_tensor,
              frontier_tensor,
              frontier_size,
              new_frontier_tensor,
              next_idx_tensor,
              visited_tensor
            ]
          )

          # Here we just return the map unaltered, because the CPU execution
          # modifies the tensors in place
          {tensor_map, :cpu, used_gpu}
        end
      end

    # For the next iteration, the next free index will be reset and the current frontier and new
    # frontier will be swapped. We will also zero out the next_idx tensor for the next iteration
    tensor_map =
      Map.merge(tensor_map, %{
        next_idx: Orchestra.tensor([0], :s32),
        frontier: new_frontier_tensor,
        new_frontier: frontier_tensor,
        frontier_gnx: new_frontier_gnx,
        new_frontier_gnx: frontier_gnx
      })

    # Updating frontier size for the next iteration
    new_frontier_size = Nx.to_number(next_idx_tensor[0])

    remaining_iterations =
      cond do
        is_integer(max_iterations) -> max_iterations - 1
        true -> max_iterations
      end

    bfs_recursion(
      nodes_map,
      new_frontier_size,
      tensor_map,
      remaining_iterations,
      cpu_limit,
      current_device,
      used_gpu
    )
  end
end

# Setting default cpu_limit
default_cpu_limit = 1024

# Getting name of file to process. The user can specify
argv = System.argv()
argv_len = length(argv)

{file, cpu_limit, it} =
  case argv_len do
    1 ->
      [f] = argv
      # Default cpu limit
      {f, default_cpu_limit, :infinity}

    2 ->
      [f, c] = argv
      c = String.to_integer(c)

      if c > 0 do
        {f, c, :infinity}
      else
        {f, default_cpu_limit, :infinity}
      end

    3 ->
      [f, c, i] = argv
      c = String.to_integer(c)
      i = String.to_integer(i)

      c = if c > 0, do: c, else: default_cpu_limit
      i = if i > 0, do: i, else: :infinity

      {f, c, i}

    _ ->
      IO.puts(
        "Usage: mix run #{Path.basename(__ENV__.file)} FILE_PATH CPU_LIMIT [MAX_ITERATIONS]\n"
      )

      IO.puts(
        "The MAX_ITERATIONS is an optional parameter that must be a positive number greater than 0. It specifies how many levels of the graph the algorithm is allowed to explore. If omitted, BFS will assume infinite iterations are allowed."
      )

      IO.puts(
        "The CPU_LIMIT is an optional parameter that must be a positive number greater than 0. It specifies the maximum frontier size that will be processed on the CPU. If the frontier size exceeds this limit, it will be processed on the GPU. If omitted, the default CPU_LIMIT is #{default_cpu_limit}."
      )

      System.halt(0)
  end

IO.puts("--- Processing Input File '#{Path.basename(file)}' ---")

start = System.monotonic_time()
graph_map = CsrReader.read_and_process_file(file)
stop = System.monotonic_time()

# IO.inspect(graph_map, label: "Graph Map")

IO.puts(
  "Time taken to read input file: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms"
)

IO.puts("\n--- Starting BFS-Warp with CPU limit: #{cpu_limit} and max iterations: #{it} ---")

start = System.monotonic_time()
used_gpu = BFS.bfs(graph_map, cpu_limit, it)
stop = System.monotonic_time()

IO.puts("BFS-Warp took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms")
IO.puts("BFS-Warp used GPU: #{used_gpu}")
