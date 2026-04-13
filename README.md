# flux-evolve

> Behavioral evolution engine with mutation, scoring, fitness-based selection, and rollback for FLUX agents.

## What This Is

`flux-evolve` is a Rust crate implementing an **evolutionary engine** for agent behaviors — it manages tunable parameters with mutation rates, tracks fitness scores, supports aggressive/normal/elite evolution modes, and provides full rollback capabilities.

## Role in the FLUX Ecosystem

Agent adaptation is core to the FLUX philosophy. `flux-evolve` drives continuous improvement:

- **`flux-trust`** provides fitness signals (trust scores) that feed evolution cycles
- **`flux-social`** determines which behaviors evolve based on agent role
- **`flux-memory`** persists evolved parameters across sessions via snapshots
- **`flux-profiler`** measures performance of evolved behaviors
- **`flux-grimoire`** records successful evolutions as "spells" for other agents to learn

## Key Features

| Feature | Description |
|---------|-------------|
| **Mutation Engine** | 8 mutation types: ParamAdjust, ThresholdShift, WeightRebalance, etc. |
| **Fitness Modes** | Aggressive (low fitness), Normal, Elite (high fitness — only worst mutate) |
| **Scoring** | `score(behavior, outcome)` accumulates evidence for each parameter |
| **Best/Worst** | Rank behaviors by average cumulative score |
| **Revert & Rollback** | Undo specific mutations or roll back to a generation |
| **Bounded Parameters** | Every behavior has min/max clamping with configurable mutation rates |

## Quick Start

```rust
use flux_evolve::Engine;

let mut engine = Engine::new();

// Define evolvable behaviors
engine.add_behavior("exploration_depth", 0.5, 0.0, 1.0, 0.1);
engine.add_behavior("caution_threshold", 0.3, 0.0, 1.0, 0.05);
engine.add_behavior("collaboration_weight", 0.7, 0.0, 1.0, 0.08);

// Score outcomes
engine.score("exploration_depth", 0.8);
engine.score("caution_threshold", -0.3);

// Evolution cycle (fitness determines strategy)
let mutations = engine.cycle(0.5); // normal fitness → normal mutation rate
println!("Generation {}: {} mutations", engine.generation(), mutations);

// Rollback if things go wrong
engine.rollback(2); // undo all mutations after generation 2

// Inspect
let worst = engine.worst_behaviors(3);
let best = engine.best_behaviors(3);
```

## Building & Testing

```bash
cargo build
cargo test
```

## Related Fleet Repos

- [`flux-trust`](https://github.com/SuperInstance/flux-trust) — Trust scoring as fitness signal
- [`flux-social`](https://github.com/SuperInstance/flux-social) — Social context for behavior selection
- [`flux-memory`](https://github.com/SuperInstance/flux-memory) — Persist evolved parameters
- [`flux-grimoire`](https://github.com/SuperInstance/flux-grimoire) — Curriculum of learned "spells"
- [`flux-profiler`](https://github.com/SuperInstance/flux-profiler) — Measure evolved behavior performance

## License

Part of the [SuperInstance](https://github.com/SuperInstance) FLUX fleet.
