# Issue and Pull Request Labels for Metal Compute Kernels

To keep the project organized and maintainable, contributors should use the following labels when creating issues or PRs.

## Categories

### 1. Type of Issue / PR

| Label        | Description                                            |
| ------------ | ------------------------------------------------------ |
| `bug`        | Indicates a bug or unexpected behavior in the codebase |
| `feature`    | New kernel, capability, or enhancement request         |
| `perf`       | Performance improvement or optimization                |
| `docs`       | Documentation update or clarification                  |
| `test`       | Test addition, improvement, or fix                     |
| `refactor`   | Code reorganization without changing behavior          |
| `question`   | For questions, discussions, or guidance requests       |
| `discussion` | Broader topic discussion not tied to a bug or feature  |

### 2. Priority / Impact

| Label             | Description                                       |
| ----------------- | ------------------------------------------------- |
| `high-priority`   | Requires immediate attention or blocks other work |
| `medium-priority` | Important but not blocking                        |
| `low-priority`    | Cosmetic, minor improvements, or optional         |

### 3. Status / Workflow

| Label              | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `good first issue` | Suitable for new contributors or first-time PRs        |
| `help wanted`      | Contributor help is needed to resolve or implement     |
| `in progress`      | Work is actively ongoing                               |
| `review needed`    | Waiting for review by maintainers                      |
| `blocked`          | Cannot proceed due to dependencies or external factors |
| `duplicate`        | Duplicate of another issue or PR                       |
| `wontfix`          | Issue or suggestion will not be addressed              |

### 4. Platform / Kernel Type (Optional)

| Label          | Description                                             |
| -------------- | ------------------------------------------------------- |
| `RNN`          | Related to recurrent neural network kernels (LSTM, GRU) |
| `Transformer`  | Related to attention or transformer kernels             |
| `Quantization` | Quantization operations or optimization                 |
| `Physics`      | Physics simulation kernels                              |
| `iOS`          | Related to iOS Metal usage                              |
| `macOS`        | Related to macOS Metal usage                            |
| `tvOS`         | Related to tvOS Metal usage                             |

## Best Practices

* Use **one type label** per issue/PR (`bug`, `feature`, etc.)
* Optionally assign **priority and status labels**
* Add platform/kernel labels only if relevant
* Maintainers should **review and update labels** as work progresses
* Keep labels consistent for discoverability and automation

---

Having a standard set of labels helps:

* Organize issues and PRs for maintainers
* Make it easier for contributors to pick tasks
* Facilitate automated workflows (like GitHub Actions for stale, triage, or PR automation)

Once this file is in the repo, you can also **pre-create these labels in GitHub** via Settings → Labels to enforce consistency.
