# Scenario planning and paper terminology

The runtime implements a loop of scenario planning, GUI execution with asynchronous bug review, and Functional Flow Graph (FFG) updating. This document describes the current implementation, replacing the earlier design proposal whose marker-based counters and completion escape hatch are no longer used.

## Research concepts and implementation

The paper [FuncDroid: Towards Inter-Functional Flows for Comprehensive Mobile App GUI Testing](https://arxiv.org/abs/2602.12834) defines a functionality by `(goal, vars, traces)` and a functional flow by `(pi, phi, pi_prime)`. Long-Term View (LTV) scenarios refine functionality definitions and flow conditions. Short-Term View (STV) scenarios explore current flow hypotheses through metamorphic relations.

The current runtime records functionality nodes, pages, widget controls, operations, page transitions, resource dependencies, and execution outcomes. `FunctionDependency` records a source, destination, and resource variable. It is not the full paper-level conditional flow representation. Breadth and depth are runtime scheduling phases, not alternative names for LTV and STV. The runtime does not claim a separate DNF-condition or complete LTV/STV strategy engine.

## Intra-functionality breadth

The planner chooses an unexplored direction or recorded controls without execution evidence. Early exploration can continue through causally related results. Later scenarios name a small set of controls sharing one stable page or result domain. Independent workflows are scheduled separately. Repeated resource instances use representative controls rather than exhaustive enumeration.

The model determines semantic scope and avoids repeating recorded scenarios. Programmatic scheduling uses recorded coverage facts and attempt counters, not a fixed list of business-specific verification rules.

## Intra-functionality depth

The program derives the phase and passes it to both planning and execution. Depth scenarios exercise a complete operation chain and observable result using recorded pages and controls. Suggested directions include persistence after leaving and returning, create/edit/delete lifecycles, boundary inputs, state interactions, and cancellation behavior. These are guidance, not mandatory per-category quotas.

If the model cannot produce a usable scenario, the depth fallback requests a complete operation chain with a return/read-back observation. Tests operate on their own generated data. Account destruction, logout, payments, subscriptions, and identity verification are excluded by existing execution guidance and guards.

## Default scheduling and completion

| Setting | Default |
| --- | --- |
| Breadth attempt threshold | 3 |
| Controls per required depth scenario | 5 |
| Minimum / maximum required depth scenarios | 1 / 6 |
| Additional attempt margin | 2 |
| Completion coverage score | 80 |

The required depth count is `clamp(ceil(recorded_controls / 5), 1, 6)`. Switch to depth when its successful count is below target and either the breadth threshold is reached or all recorded controls have been covered. After successful depth coverage reaches its target, remaining control gaps return to breadth.

An attempt consumes budget once the test stage starts, including failures. Navigation skips and infrastructure failures do not count as test attempts. `depth_scenario_count` counts depth attempts; `successful_depth_scenario_count` requires a completed scenario and an observed, healthy post-action screen change. That change establishes a response, not the correctness of every business assertion.

Functionality completion requires the adjusted coverage score, a successful business-operation record, and sufficient successful depth scenarios. The coverage score scales handled controls by functionality size; it is not a raw percentage of all possible states or paths. Repeated actions on the same control do not increase control coverage.

The total attempt cap is `breadth_threshold + required_depth_count + margin`. Exhaustion without completion defers a functionality rather than marking it complete. Newly recorded controls may lower coverage or raise the depth target, reopening completed functionality when appropriate. Failure penalties take priority over depth scheduling to avoid retry loops.

## Inter-functional scenarios

Candidates arise from explicit resource dependencies and shared resource variables. Ready candidates may run before unrelated intra-functionality work: both endpoints need their own attempts and successful business operations. Each scenario exercises production/modification at the source followed by reading/display/consumption of the same resource at the destination. Navigation alone is insufficient.

Outcomes are tracked by source, destination, and resource, with bounded retries. Model-generated reports and successful process exits do not establish a reproduced bug; inspect the recorded actions, post-action observations, screenshots, and system logs.

## Validation

Offline regression tests cover scheduling, evidence counters, graph updates, resource-flow outcomes, and shutdown artifacts. Device acceptance additionally requires a version-matched target app, execution traces, screenshots, `run_result.json`, and `artifact_manifest.json`. Offline tests do not replace that acceptance run.
