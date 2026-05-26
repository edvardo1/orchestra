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

  defk gpu_bfs_kernel(
         nodes,
         n_nodes,
         edges,
         frontier,
         frontier_size,
         new_frontier,
         atomic(next_slot),
         atomic(visited),
         overflow
       ) do
    tid = get_global_id(0)
    lid = get_local_id(0)

    # Declare a local buffer for this thread
    __local(local_buffer[256])
    __local(shift[1])
    __atomic_local(local_free_idx[1])

    # Only thread 0 of the work group will initialize the local free index and shift
    if lid == 0 do
      init_atomic_int(local_free_idx, 0)
      shift[0] = 0
    end

    # All threads need to wait for the initialization to be done
    __syncthreads()

    if tid < frontier_size && tid < n_nodes do
      # Get node to process in this thread
      node_idx = frontier[tid]
      # Mark it as visited
      was_visited = max_atomic_int(visited + node_idx, 1)

      # -- Get node info --
      # Node edges index in edges array
      node_edges_idx = nodes[node_idx * 2 + 0]
      # Number of edges this node has
      node_num_edges = nodes[node_idx * 2 + 1]

      for i in range(node_edges_idx, node_edges_idx + node_num_edges) do
        # Getting child node
        dest_node_idx = edges[i * 2 + 0]

        # Mark as visited and check if it was already visited in the same atomic operation
        was_visited = max_atomic_int(visited + dest_node_idx, 1)

        # Check if this node was visited
        if was_visited == 0 do
          # Update local buffer with this node
          idx = add_atomic_int(local_free_idx, 1)

          if idx < 256 do
            local_buffer[idx] = dest_node_idx
          else
            overflow[0] = 1
          end
        end
      end
    end

    # Wait for all threads to finish populating the buffer before reading its size
    __syncthreads()

    priv_buffer_size = load_atomic_int(local_free_idx)

    if lid == 0 do
      shift[0] = add_atomic_int(next_slot, priv_buffer_size)
    end

    # Wait for shift to be updated
    __syncthreads()

    # Now we need to update the global new frontier with the local buffer of the work group
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
    IO.puts("============== Creating Tensors... ==============")

    frontier_size = 1

    start = System.monotonic_time()

    tensor_map = %{
      frontier: Orchestra.tensor({total_nodes}, :s32, fn _ -> start_node end),
      new_frontier: Orchestra.tensor({total_nodes}, :s32),
      visited: Orchestra.tensor({total_nodes}, :s32, fn _ -> 0 end),
      next_idx: Orchestra.tensor([0], :s32),
      overflow: Orchestra.tensor([0], :s32)
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
          edges_gnx: Orchestra.new_gnx(edges_tensor),
          overflow_gnx: Orchestra.new_gnx(tensor_map.overflow)
        })
      end

    stop = System.monotonic_time()

    IO.puts(
      "Tensor creation took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms"
    )

    IO.puts("============== Starting Recursion ==============")

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
        ) :: :ok
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
           overflow: overflow_tensor,
           frontier_gnx: frontier_gnx,
           new_frontier_gnx: new_frontier_gnx,
           visited_gnx: visited_gnx,
           next_idx_gnx: next_idx_gnx,
           nodes_gnx: nodes_gnx,
           edges_gnx: edges_gnx,
           overflow_gnx: overflow_gnx
         } = tensor_map,
         max_iterations,
         cpu_limit,
         last_device,
         used_gpu
       ) do
    {tensor_map, current_device, used_gpu} =
      if frontier_size > cpu_limit do
        # IO.puts("============== FRONTIER: #{frontier_size} > #{cpu_limit} | GPU")

        if last_device == :cpu do
          # IO.puts("== Switching from CPU to GPU. Copying tensors to GPU...")

          # If we are switching from CPU to GPU, we need to copy the frontier and visited tensors to the GPU
          Orchestra.write_gnx(frontier_gnx, frontier_tensor)
          Orchestra.write_gnx(visited_gnx, visited_tensor)
        end

        Orchestra.with Orchestra.gpu() do
          # We have to zero out the next_idx and overflow tensors on the GPU before each iteration.
          # The 'next_idx_tensor' will always be zero before every iteration, so we can just copy it.
          Orchestra.write_gnx(next_idx_gnx, next_idx_tensor)
          Orchestra.write_gnx(overflow_gnx, next_idx_tensor)

          threads_per_block = 128
          num_blocks = div(frontier_size + threads_per_block - 1, threads_per_block)

          Orchestra.spawn(
            &BFS.gpu_bfs_kernel/9,
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
              visited_gnx,
              overflow_gnx
            ]
          )

          # After GPU execution, only take the next_idx and overflow tensors back
          # We will leave the heavy boys on the GPU
          Orchestra.get_gnx(next_idx_gnx, next_idx_tensor)
          Orchestra.get_gnx(overflow_gnx, overflow_tensor)

          {tensor_map, :gpu, true}
        end
      else
        # IO.puts("============== FRONTIER: #{frontier_size} <= #{cpu_limit} | CPU")

        if last_device == :gpu do
          # IO.puts("== Switching from GPU to CPU. Copying tensors to CPU...")

          # If we are switching from GPU to CPU, we need to copy the frontier and visited tensors back to the CPU
          Orchestra.with Orchestra.gpu() do
            Orchestra.get_gnx(frontier_gnx, frontier_tensor)
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

    if Nx.to_number(overflow_tensor[0]) == 1 do
      # Overflow happened in GPU kernel. Print a message to stderr
      IO.puts(
        :stderr,
        "Warning: overflow in GPU kernel. Some nodes may not have been added to the frontier. Consider increasing the CPU limit to avoid this issue."
      )
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

# Getting name of file to process. The user can specify
argv = System.argv()
argv_len = length(argv)

{file, cpu_limit, it} =
  case argv_len do
    1 ->
      [f] = argv
      # Default cpu limit is 512
      {f, 512, :infinity}

    2 ->
      [f, c] = argv
      c = String.to_integer(c)

      if c > 0 do
        {f, c, :infinity}
      else
        {f, 512, :infinity}
      end

    3 ->
      [f, c, i] = argv
      c = String.to_integer(c)
      i = String.to_integer(i)

      if i > 0 and c > 0 do
        {f, c, i}
      else
        {f, 512, :infinity}
      end

    _ ->
      IO.puts(
        "Usage: mix run #{Path.basename(__ENV__.file)} FILE_PATH CPU_LIMIT [MAX_ITERATIONS]\n"
      )

      IO.puts(
        "The MAX_ITERATIONS is an optional parameter that must be a positive number greater than 0. It specifies how many levels of the graph the algorithm is allowed to explore. If omitted, BFS will assume infinite iterations are allowed."
      )

      IO.puts(
        "The CPU_LIMIT is an optional parameter that must be a positive number greater than 0. It specifies the maximum frontier size that will be processed on the CPU. If the frontier size exceeds this limit, it will be processed on the GPU. If omitted, the default CPU_LIMIT is 512."
      )

      System.halt(0)
  end

IO.puts("--- Processing Input File '#{Path.basename(file)}' ---")

start = System.monotonic_time()
graph_map = CsrReader.read_and_process_file(file)
stop = System.monotonic_time()

IO.inspect(graph_map)

IO.puts(
  "Time taken to read input file: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms"
)

IO.puts("--- Starting BFS with CPU limit: #{cpu_limit} and max iterations: #{it} ---")

start = System.monotonic_time()
used_gpu = BFS.bfs(graph_map, cpu_limit, it)
stop = System.monotonic_time()

IO.puts("BFS took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms")
IO.puts("BFS used GPU: #{used_gpu}")
