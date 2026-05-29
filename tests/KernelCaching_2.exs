require Orchestra

# Orchestra.set_debug_logs(true)

Orchestra.defmodule PMap do
  defk map_ker(a1, a2, size, f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index, size, stride) do
      a2[i] = f(a1[i])
    end
  end

  defd sum_10(x) do
    x + 10
  end

  defd mul_2(x) do
    x * 2
  end

  def map(input, f) do
    Orchestra.with Orchestra.gpu() do
      input_gnx = Orchestra.new_gnx(input)

      shape = Orchestra.get_shape(input)
      type = Orchestra.get_type(input)
      result_gpu = Orchestra.new_gnx(shape, type)

      size = Tuple.product(shape)
      threadsPerBlock = 128
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      Orchestra.spawn(
        &PMap.map_ker/4,
        {numberOfBlocks, 1, 1},
        {threadsPerBlock, 1, 1},
        [input_gnx, result_gpu, size, f]
      )

      Orchestra.get_gnx(result_gpu)
    end
  end
end

a = Orchestra.tensor(Enum.to_list(1..1024), :s32)

IO.puts("Running kernel with anonymous function: x * 2")

result = a |> PMap.map(&PMap.mul_2/1)

IO.inspect(result, label: "Result")

IO.puts("Running kernel with anonymous function: x + 10")

result = a |> PMap.map(&PMap.sum_10/1)

IO.inspect(result, label: "Result")

IO.puts("Running kernel with anonymous function: x * 2 again")

result = result |> PMap.map(&PMap.mul_2/1)

IO.inspect(result, label: "Result")
