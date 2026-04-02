# Tanaka Solver Workflow

```mermaid
flowchart TD
    A[User Chooses Task] --> B{Workflow}

    B -->|Single solitary wave| C[modified_tanaka.py]
    B -->|Uniform amplitude branch| D[build_amplitude_dataset.py]
    B -->|Multi-crest initial condition| E[gen_data/multi_crest.py]
    B -->|IC comparison against provided data| F[plot_ic_comparisons.py]

    C --> C1[Build transformed phi grid]
    C1 --> C2[Fixed-point Tanaka iteration]
    C2 --> C3[Recover q, theta, Froude, speed]
    C3 --> C4[Reconstruct x-profile and eta-profile]
    C4 --> C5[Interpolate onto periodic grid]
    C5 --> C6[Recover xi and Geta_xi]
    C6 --> C7[Single-wave solution]

    D --> D1[Read amplitude min/max from soliton data]
    D1 --> D2[Sample amplitudes uniformly]
    D2 --> D3[Batched Tanaka solves]
    D3 --> D4[Save canonical branch .npz]

    E --> E1[Preset crest specs or custom specs]
    E1 --> E2[Batched Tanaka solves for all crests]
    E2 --> E3[Sum component eta and xi]
    E3 --> E4[Optionally zero-mean xi]
    E4 --> E5[Recompute composite Geta_xi with DNO]
    E5 --> E6[Multi-crest initial condition payload]

    F --> F1[Load initial frame from provided soliton file]
    F1 --> F2{Case family}
    F2 -->|s*| F3[Batch solve both directions]
    F2 -->|h* or f*| F4[Detect crest centers and amplitudes]
    F4 --> F5[Batch solve crest-direction combinations]
    F3 --> F6[Assemble candidate model(s)]
    F5 --> F6
    F6 --> F7[Compute eta, xi, Geta_xi errors]
    F7 --> F8[Write png, npz, json summaries]

    E6 --> G[time_integrator.py rollout]
    G --> H[evals/render_multi_crest_rollouts.py]
    G --> I[evals/plot_multi_crest_rollouts.py]

    D4 --> E1
```
