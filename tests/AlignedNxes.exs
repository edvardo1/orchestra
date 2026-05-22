require Orchestra

Orchestra.set_debug_logs(true)

IO.puts("\n---Testando Orchestra.tensor/2 que gera o array a partir de uma lista Elixir aninhada---")

tensor_1 = Orchestra.tensor(
  [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
  ], type: :f32
)
IO.inspect(tensor_1, label: "Tensor gerado a partir de lista Elixir aninhada")

if Orchestra.is_nx_aligned?(tensor_1) do
  IO.puts("tensor_1 é alinhado")
else
  IO.puts("tensor_1 NÃO é alinhado")
end

IO.puts("\n---Testando Orchestra.tensor/3 que gera o array usando uma função---")

tensor_2 = Orchestra.tensor({10, 1_000}, :f32, fn i -> i * 1.0 end)
IO.inspect(tensor_2, label: "Tensor gerado usando função")

if Orchestra.is_nx_aligned?(tensor_2) do
  IO.puts("tensor_2 é alinhado")
else
  IO.puts("tensor_2 NÃO é alinhado")
end
