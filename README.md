# jero-arch-mistral

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-arch-mistral/card.svg" alt="jero-arch-mistral Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-mistral/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-mistral/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-mistral/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-mistral/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Mistral and Mixtral architecture recipe (supporting `mistral` Sliding Window Attention, `mistral3`, and `mixtral` Sparse Mixture-of-Experts) for the SpacemiT K3 platform.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependencies**: [`jero-core`](https://github.com/LMA1980/jero-core) and [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv) (via the optional `with-kernels` feature)
- **Downstream Consumers**: [`jero-engine`](https://github.com/LMA1980/jero-engine) and [`jero-manager`](https://github.com/LMA1980/jero-manager)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
