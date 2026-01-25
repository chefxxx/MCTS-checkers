# 🏁 CUDA-Accelerated MCTS Checkers Engine

A high-performance Checkers engine utilizing **Monte Carlo Tree Search (MCTS)** with a hybrid CPU-GPU architecture.

## 🚀 Key Features

* **Hybrid MCTS Architecture**:
    * Implements the Upper Confidence Bound applied to Trees (**UCT**) formula to navigate the search space.
    * **CPU**: Manages the high-level MCTS tree (Selection, Expansion, and Backpropagation).
    * **GPU**: Executes parallel rollouts using custom CUDA kernels.
* **Bitboard Move Generation**:
    * **Hyperbola Quintessence**: Implements sliding move logic for Flying Kings.
* **High-Performance CUDA Kernels**:
    * **Warp-Aggregated Atomics**: Optimized score summation to reduce memory contention.
    * **Persistent PRNG**: Managed `curand` states across rollout iterations for high-quality randomness.

---

## 🏗 Project Structure

```text
.
├── project/
│   ├── include/            # C++ & CUDA Headers
│   │   ├── cpu/            # MCTS tree and host logic
│   │   ├── gpu/            # Device kernels and GPU board state
│   │   ├── cuda_utils/     # CUDA helper headers
│   │   └── utils/          # Bit manipulation and logging
│   └── src/
│       ├── cpu/            # CPU implementation files
│       └── cuda/           # CUDA implementation files (.cu)
├── tests/                  # GTest suite for move generation and MCTS logic
├── games/                  # Generated game logs and histories
└── CMakeLists.txt          # Unified build configuration
```

## 🛠 Getting Started

Before building, ensure your environment meets the following requirements:

* **NVIDIA** **GPU**: Architecture ```sm_61``` (Pascal) or newer.
* **CUDA** **TOOLKIT**: Version 12.x or higher.
* **C++** **Compiler**: GCC 12+, Clang 15+, or MSVC 19.30+ (Full C++20 support required).
* **CMake**: Version 3.20 or higher.

## 📑 License

This project is licensed under GPL-3.0 license. See the [LICENSE](LICENSE) file for the full license text.
