# Orchestra

Orchestra is a DSL for cooperative heterogeneous programming based on the [OCL-PolyHok](https://github.com/Equiel-1703/ocl-polyhok) DSL.

## Prerequisites

To get started with Orchestra, first ensure you have the following prerequisites on your system:

- **Elixir 1.17 with Erlang/OTP 27 or higher**. We recommend using [asdf](https://asdf-vm.com/) for managing Elixir and Erlang versions.

- **Erlang development libraries**. In Debian/Ubuntu systems, you can install them via:

  ```bash
  sudo apt install erlang-dev
  ```

- **OpenCL 2.0 compatible hardware**. Check your GPU specifications to ensure it supports OpenCL version 2.0 or higher.

- **OpenCL generic ICD loader and C/C++ headers**. You can install the necessary OpenCL packages using your package manager. For Debian/Ubuntu systems, you can use:

  ```bash
  sudo apt install ocl-icd-opencl-dev opencl-c-headers opencl-clhpp-headers
  ```

  **Note**: The packages listed above are for general OpenCL development in C/C++. You will need to ensure that your system has the latest drivers for your GPU. This ensures that the OpenCL ICD loader can find and route commands to the appropriate OpenCL driver provided by your GPU vendor. You can verify that OpenCL is properly set up on your system by using the `clinfo` tool, which provides detailed information about the OpenCL platforms and devices available on your system. This tool may not be installed by default in some Linux distributions, but you can easily install it using your package manager. For Debian/Ubuntu systems, you can run:

  ```bash
  sudo apt install clinfo
  ```

  **Pro Tip**: OCL-PolyHok was developed and tested on a Lenovo IdeaPad 3 running Linux Mint 22.3 (Zena), equipped with an AMD Ryzen 5 5500U CPU and its integrated GPU, the AMD Lucienne. If you have a similar setup, we strongly recommend using the Mesa OpenCL driver, which works very well with AMD's iGPUs, and enabling the `rusticl` implementation (a modern OpenCL implementation written in Rust). You can install the Mesa OpenCL driver by running the following command:

  ```bash
  sudo apt install mesa-opencl-icd
  ```

  And to enable the `rusticl` implementation, set the `RUSTICL_ENABLE` environment variable to `radeonsi` in your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

  ```bash
  export RUSTICL_ENABLE='radeonsi'
  ```

## Getting Started

Once you have the prerequisites, follow these steps to set up and compile the project:

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/Equiel-1703/orchestra.git
   cd orchestra
   ```

2. Get Elixir dependencies:

    ```bash
    mix deps.get
    ```

3. Generate CMake build scritps and compile the C++ NIFS for Orchestra's OpenCL runtime and BMP generation:

    ```bash
    mkdir CMake
    cmake -S . -B CMake
    cmake --build CMake
    ```

4. Compile the Elixir project:

    ```bash
    mix compile
    ```

5. All done! You can now run the provided benchmarks or start developing your own Orchestra's applications.

## Licensing

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
