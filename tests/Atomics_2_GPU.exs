require Orchestra

Orchestra.set_debug_logs(true)

Orchestra.defmodule Atomics_2 do
  defd atomic_add_d(atomic(counter), val) do
    return atomic_add_int(counter, val)
  end

  defk atomic_kernel_2(atomic(counter), out_array) do
    tid = get_global_id(0)

    previous_val = atomic_add_d(counter, 1)

    out_array[tid] = previous_val
  end

  def launch(counter_tensor, total_threads) do
    block_size = 256
    num_blocks = div(total_threads + block_size - 1, block_size)

    {res_counter, res_array} =
      Orchestra.with Orchestra.gpu() do
        counter_gnx = Orchestra.new_gnx(counter_tensor)
        out_array_gnx = Orchestra.new_gnx({total_threads}, :s32)

        Orchestra.spawn(
          &Atomics_2.atomic_kernel_2/2,
          {num_blocks},
          {block_size},
          [counter_gnx, out_array_gnx]
        )

        res_counter = Orchestra.get_gnx(counter_gnx)
        res_array = Orchestra.get_gnx(out_array_gnx)

        {res_counter, res_array}
      end

    {res_counter, res_array}
  end
end

# Total threads
total_threads = 1024

# Counter tensor
counter_tensor = Orchestra.tensor([0], :s32)

# Launching kernel
{counter_res, array_res} = Atomics_2.launch(counter_tensor, total_threads)

IO.inspect(counter_res, label: "* Counter")
IO.inspect(array_res, limit: :infinity, label: "* Output Array")
