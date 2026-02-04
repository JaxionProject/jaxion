# Jaxion Project Context

## Overview

Jaxion is a high-performance JAX-based simulation library for modeling fuzzy dark matter alongside stars, gas, and cosmological dynamics. It is built for multi-GPU scalability and is fully differentiable, enabling seamless integration with pipelines for inverse-problems, inference, optimization, and coupling to ML models.

**Author**: Philip Mocz (@pmocz)

**Companion Project**: Jaxion is the simpler companion to [Adirondax](https://github.com/AdirondaxProject/adirondax), a more comprehensive differentiable astrophysics code.

## Project Structure

- `src/jaxion/` - Main source code for the library
- `examples/` - Demonstration scripts showing various simulation capabilities
- `docs/` - Documentation source files
- `tests/` - Test suite

## Key Technologies

- **JAX**: Core computational framework providing automatic differentiation and GPU/TPU acceleration
- **Python**: Primary language
- **Multi-GPU**: Designed for scalable parallel computation
- **Differentiable**: All operations support automatic differentiation

## Simulation Capabilities

Jaxion supports simulations of:
- Fuzzy dark matter dynamics
- Stellar systems
- Gas dynamics
- Cosmological evolution
- Inverse problems and optimization

## Development Guidelines

### Code Style
- Follows Ruff linting standards
- Type hints should be used where appropriate
- JAX functional programming patterns preferred

### Testing
- Tests are run via GitHub Actions
- Coverage tracked via Codecov
- Benchmarking via ASV (airspeed velocity)

### Performance Considerations
- Code must be JAX-compatible (pure functions, no side effects)
- GPU/TPU performance is critical
- Multi-device parallelization should be considered

### Documentation
- Hosted on ReadTheDocs
- Examples should be runnable and well-commented
- API documentation generated from docstrings

## Common Patterns

### JAX Best Practices
- Use `jax.jit` for compilation
- Ensure functions are pure (no side effects)
- Use `jax.vmap` for vectorization
- Use `jax.grad` for automatic differentiation

### Multi-GPU Support
- Code should be device-agnostic where possible
- Consider sharding strategies for large simulations

## Examples

The `examples/` directory contains various demonstrations:
- `cosmological_box/` - Cosmological simulation in a periodic box
- `dynamical_friction/` - Dynamical friction effects
- `heating_gas/` - Gas heating simulations
- `heating_stars/` - Stellar heating effects
- `kinetic_condensation/` - Kinetic condensation phenomena
- `logo_inverse_problem/` - Inverse problem demonstration
- `soliton_binary_merger/` - Binary soliton merger
- `soliton_merger/` - Soliton merger dynamics
- `tidal_stripping/` - Tidal stripping effects

## Current TODO/Wishlist

- See README.md for current TODO/Wishlist

## Installation

```bash
pip install jaxion
```

Or build from source following the documentation.

## Links

- [GitHub Repository](https://github.com/JaxionProject/jaxion)
- [Documentation](https://jaxion.readthedocs.io)
- [Benchmarks](https://jaxionproject.github.io/jaxion-benchmarks/)
- [PyPI Package](https://pypi.org/project/jaxion)

## License

Apache License 2.0

## Citation

```bibtex
@software{Mocz_Jaxion_2025,
   author = {Mocz, Philip},
      doi = {10.5281/zenodo.17438467},
    month = dec,
    title = {{Jaxion}},
      url = {https://github.com/JaxionProject/jaxion},
  version = {0.0.9},
     year = {2025}
}
```
