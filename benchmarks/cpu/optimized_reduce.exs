require Orchestra

# Orchestra.set_debug_logs(true)

Nx.default_backend(Nx.BinaryBackend)

Orchestra.defmodule Reduce do
  include(CAS_Poly)

  # This reduce kernel was rewritten to better perform in CPU
  defk reduce_kernel(arr, result_arr, initial_value, f, arr_size) do
    tid = get_global_id(0)
    num_cores = get_global_size(0)

    range = (arr_size + num_cores - 1) / num_cores
    start = tid * range
    stop = start + range

    if stop > arr_size do
      stop = arr_size
    end

    local_sum = initial_value

    for i in range(start, stop) do
      local_sum = f(local_sum, arr[i])
    end

    if start < arr_size do
      result_arr[tid] = local_sum
    else
      result_arr[tid] = initial_value
    end
  end

  # This is the original reduce kernel used for GPU execution
  defk reduce_kernel_gpu(a, ref4, initial, f, n) do
    __shared__(cache[256])

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cacheIndex = threadIdx.x

    temp = initial

    while tid < n do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
    __syncthreads()

    i = blockDim.x / 2

    while i != 0 do
      if cacheIndex < i do
        cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      current_value = ref4[0]

      while(!(current_value == cas_float(ref4, current_value, f(cache[0], current_value)))) do
        current_value = ref4[0]
      end
    end
  end

  def reduce(tensor, initial, f) do
    cores = 12

    shape = Orchestra.get_shape(tensor)
    type = Orchestra.get_type(tensor)
    len = Nx.size(shape)

    result_tensor = Orchestra.tensor({cores}, type, fn _i -> initial end)

    Orchestra.with Orchestra.cpu() do
      Orchestra.spawn(
        &Reduce.reduce_kernel/5,
        {cores},
        {1},
        [tensor, result_tensor, initial, f, len]
      )
    end

    Nx.sum(result_tensor)
  end

  def reduce_2(tensor, initial, f) do
    shape = Orchestra.get_shape(tensor)
    type = Orchestra.get_type(tensor)
    len = Nx.size(shape)

    threadsPerBlock = 8
    blocksPerGrid = div(len + threadsPerBlock - 1, threadsPerBlock)

    result_tensor = Orchestra.tensor([initial], type)

    Orchestra.with Orchestra.cpu() do
      Orchestra.spawn(
        &Reduce.reduce_kernel_gpu/5,
        {blocksPerGrid},
        {threadsPerBlock},
        [tensor, result_tensor, initial, f, len]
      )
    end

    result_tensor
  end
end

[size, type] = System.argv()
n = String.to_integer(size)

t =
  case type do
    "o" ->
      IO.puts("Using optimized reduce kernel")
      :optimized

    "g" ->
      IO.puts("Using GPU-like reduce kernel")
      :gpu

    _ ->
      IO.puts("Invalid type. Use 'o' for optimized or 'g' for GPU-like.")
      System.halt(1)
  end

IO.puts("Using Nx backend: #{inspect(Nx.default_backend())}\n")

vet1 = Orchestra.tensor({n}, :f32, fn _i -> 1.0 end)

IO.puts("Starting reduction...\n")

{res, time} =
  case t do
    :optimized ->
      prev = System.monotonic_time()
      reduce_optimized = vet1 |> Reduce.reduce(0.0, Orchestra.phok(fn a, b -> a + b end))
      next = System.monotonic_time()
      time = System.convert_time_unit(next - prev, :native, :millisecond)
      {reduce_optimized, time}

    :gpu ->
      prev = System.monotonic_time()
      reduce_gpu = vet1 |> Reduce.reduce_2(0.0, Orchestra.phok(fn a, b -> a + b end))
      next = System.monotonic_time()
      time = System.convert_time_unit(next - prev, :native, :millisecond)
      {reduce_gpu, time}
  end

IO.puts("Time taken: #{time} ms\n")
IO.inspect(res, label: "Result")
