require Orchestra
use Ske

n = 1000

arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})


arr1
    |> Orchestra.new_gnx
    |> Ske.map(Orchestra.phok fn (x) -> x + 1 end)
    |> Orchestra.get_gnx
    |> IO.inspect
