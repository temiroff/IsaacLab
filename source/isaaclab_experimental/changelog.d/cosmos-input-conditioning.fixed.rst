* Cosmos service can condition generation on the incoming frame rate and explicitly request control alignment.
  Batched prompts retain their frame rates.
* Cosmos service optionally disables cuDNN autotuning after model load for fixed-seed comparisons.
* JSON prompt files use compact serialization to reproduce calibrated caption bytes.
