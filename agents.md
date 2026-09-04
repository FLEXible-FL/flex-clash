# Agent Guide for `flex-clash`

This document provides essential architectural context, design patterns, testing conventions, and development practices for AI agents working in this repository.

---

## 1. Project Overview & Mission

`flex-clash` is a Python library dedicated to **adversarial attacks and defenses in Federated Learning (FL)**. It extends the [`FLEXible`](https://github.com/FLEXible-FL/FLEXible) (`flexible-fl`) framework.

Key capabilities provided:
- **Data Poisoning Attacks**: Techniques to manipulate training datasets on client nodes (both targeted backdoor and untargeted Byzantine poisoning).
- **Model Poisoning Attacks**: Techniques to manipulate client model updates/weights prior to aggregation (e.g. weight randomizer, Inner Product Manipulation / IPM).
- **Robust Aggregation Defenses**: Aggregator operators designed to withstand Byzantine client attacks (Median, Trimmed Mean, MultiKrum, Bulyan, Central Differential Privacy).

---

## 2. Environment & Tooling

- **Python Version**: `>= 3.12`.
- **Environment Management**: Always invoke Python and test tools through `uv run` or use the active virtual environment (`.venv`):
  ```bash
  uv run <command>
  ```
- **Editable Installation**:
  ```bash
  uv pip install -e .
  ```
- **Key Dependencies**:
  - `flexible-fl`: Core federated learning primitives (`FlexPool`, `FlexModel`, decorators).
  - `tensorly`: Universal tensor computation backend supporting NumPy, PyTorch, and TensorFlow.
    > [!IMPORTANT]
    > `tensorly` with the PyTorch backend requires the `packaging` package (`packaging.version.Version`).
  - `torch`: PyTorch models and operations.
  - `scikit-learn` & `numpy`: Standard dataset generation and evaluation.
  - `pytest`: Test runner.
  - `ruff`: Code formatting and linting.
- **Optional Dependencies**:
  - `tensorflow`: Optional backend for TensorFlow models and tests. Test suites use `importlib.util.find_spec("tensorflow")` to skip TensorFlow tests when not installed.

---

## 3. Repository Architecture

```
flex-clash/
├── flexclash/
│   ├── data/
│   │   ├── dataset.py                # PoisonedDataset class wrapping underlying client data
│   │   └── poisoning_decorators.py   # @data_poisoner decorator
│   ├── model/
│   │   ├── attacks.py                # Model poisoning attacks (IPM) and framework weight adapters
│   │   └── poisoning_decorators.py   # @model_poisoner decorator
│   └── pool/
│       └── defences.py               # Robust aggregation defense operators (MultiKrum, Bulyan, etc.)
├── notebooks/                        # Walkthrough tutorials and evaluation notebooks
│   ├── inner_product_manipulation.ipynb
│   ├── poison_models.ipynb
│   ├── poisoning_data.ipynb
│   └── using_defenses.ipynb
├── tests/
│   ├── data/
│   │   └── test_data_decorators.py
│   ├── model/
│   │   ├── test_attacks.py           # IPM math, backend adapters, and pool integration tests
│   │   └── test_model_decorators.py
│   └── pool/
│       └── test_aggregators.py       # Aggregator defense tests across backends
├── pyproject.toml
├── setup.py
└── README.md
```

---

## 4. Architectural Patterns & Integration with FLEXible

### 4.1 `FlexPool` Architecture
A `FlexPool` organizes distributed FL nodes into client, server, and aggregator roles:
```python
pool = FlexPool.client_server_pool(fed_dataset, init_func=build_server_model)
server = pool.servers          # Server nodes
clients = pool.clients        # Client nodes (can be partitioned into subpools)
aggregator = pool.aggregators  # Aggregator nodes (often server node co-located)
```

- Subsets can be created using `pool.select(count)` (random sample) or `pool.select(lambda actor_id, role: ...)` (predicate filter).
- Operations are dispatched using `.map(func, *args, **kwargs)`.

### 4.2 `FlexModel` Structure
- `FlexModel` is a dictionary-like wrapper (`collections.UserDict`).
- The underlying model is typically stored under `client_model["model"]` (e.g. `torch.nn.Module`, `tf.keras.Model`, or scikit-learn estimator).
- Raw weights or layer lists may alternatively be stored under `client_model["weights"]`.

### 4.3 Federated Round Lifecycle in `FLEXible` + `flexclash`
```
1. Deploy Server Model -> server.map(deploy_server_model_pt, clients)
2. Local Training       -> clients.map(local_train_func)
3. Poisoning Attack     -> inner_product_manipulation(clients, server_model, malicious_clients=[...])
                           OR malicious_pool.map(poison_fn)
4. Collect Weights     -> aggregator.map(collect_clients_weights_pt, clients)
5. Robust Aggregation  -> aggregator.map(multikrum)  # or median, bulyan, trimmed_mean
6. Deploy Aggregated   -> aggregator.map(set_aggregated_weights_pt, server)
```

---

## 5. Model Poisoning Conventions

When adding or modifying model poisoning attacks in `flexclash/model/attacks.py`:

1. **Low-Level Functional Core (`*_f`)**:
   - Provide a functional version (e.g., `inner_product_manipulation_f`) that operates purely on weight lists of tensors/arrays.
   - Use `set_tensorly_backend(weights)` to support NumPy, PyTorch, and TensorFlow transparently.
   - Safely ignore or preserve non-trainable buffers (such as PyTorch `BatchNorm`'s `num_batches_tracked` or empty tensors).
2. **Framework Weight Adapters**:
   - Use `extract_model_weights` and `set_model_weights` in `flexclash/model/attacks.py` to extract from or mutate `torch.nn.Module`, `tf.keras.Model`, or `FlexModel` containers in-place.
3. **High-Level Pool Operators & Poisoners**:
   - Provide high-level operators accepting either explicit pools (`(malicious_pool, server_model, honest_pool=...)`) or unified client pools (`(clients, server_model, malicious_clients=[...])`).
   - Provide a factory returning a `@model_poisoner`-decorated callable for `.map()` workflows (e.g., `ipm_poisoner`).
4. **Public Exports**:
   - Always expose new public functions in `flexclash/model/__init__.py` and include them in `__all__`.

---

## 6. Testing & Quality Standards

- **Running Tests**:
  ```bash
  uv run pytest -v
  ```
- **Linting**:
  ```bash
  uv run ruff check flexclash tests
  ```
- **Backend Portability**:
  - Tests should pass even if `tensorflow` is not present in the runtime environment.
  - Test math invariants explicitly (e.g. verifying that the inner product $\langle v, \bar{\Delta w} \rangle$ is strictly negative for IPM).
  - Verify that honest client models remain untouched when attacking a subset of a pool.

---

## 7. Useful Tips & References

- **Git Branching**: Keep branches focused (e.g., feature branches for attacks/defenses).
- **Citations**: Always verify academic references against official conference proceedings (e.g., PMLR Volume 115 for Xie et al., UAI 2020).
