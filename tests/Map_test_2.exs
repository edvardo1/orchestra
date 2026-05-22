require Orchestra

Orchestra.defmodule PMap do
  defk map_ker(a1,a2,size,f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          a2[i] = f(a1[i])
    end
  end

  def map(input, f) do
    shape = Orchestra.get_shape(input)
    type = Orchestra.get_type(input)
    result_gpu = Orchestra.new_gnx(shape,type)

    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    Orchestra.spawn(&PMap.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
end

a = Nx.tensor(Enum.to_list(1..1000),type: {:s, 32})
size = Tuple.product(Nx.shape(a))

result = a
    |> Orchestra.new_gnx
    |> PMap.map(Orchestra.phok fn x -> x + 1 end)
    |> Orchestra.get_gnx

IO.inspect(result, limit: :infinity)
