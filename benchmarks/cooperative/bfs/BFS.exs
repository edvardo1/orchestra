require Orchestra

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
      total_nodes: final_map.total_nodes,
      total_edges: final_map.total_edges,
      nodes: Orchestra.tensor(nodes_rev, :s32),
      edges: Orchestra.tensor(edges_rev, :s32)
    }
  end
end

graph_file_path = Path.join(__DIR__, "example-graph-1")

start = System.monotonic_time()
graph_map = CsrReader.read_and_process_file(graph_file_path)
stop = System.monotonic_time()

IO.inspect(graph_map)
IO.puts("Time taken: #{System.convert_time_unit(stop - start, :native, :millisecond)}")

Orchestra.defmodule BFS do
  defk cpu_bfs_kernel(nodes, n_nodes, edges, n_edges, frontier, frontier_size, new_frontier, next_slot, visited) do
    tid = get_global_id(0)

    if tid >= frontier_size || tid >= n_nodes do
      return
    end

    node_idx = frontier[tid]

    node_edges_idx = nodes[node_idx * 2 + 0]
    node_num_edges = nodes[node_idx * 2 + 1]

    # Getting child nodes
    for i in range(node_edges_idx, node_edges_idx + node_num_edges) do
      dest_node_idx = edges[i * 2 + 0] # I'm ignoring the cost for now

      if visited[dest_node_idx] != 1 do
        visited[dest_node_idx] = 1

      end
    end

  end
end

# Size of frontier
nodes_to_visit = 1

# Frontier queue and visited array
frontier = Orchestra.tensor({graph_map.total_nodes}, :s32)
new_frontier = Orchestra.tensor({graph_map.total_nodes}, :s32)
visited = Orchestra.tensor({graph_map.total_nodes}, :s32, fn _ -> 0 end)
