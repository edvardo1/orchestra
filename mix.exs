defmodule Orchestra.MixProject do
  use Mix.Project

  def project do
    [
      app: :orchestra,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Orchestra.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.12.1"},
      {:csv, "~> 3.2.2"}
    ]
  end
end
