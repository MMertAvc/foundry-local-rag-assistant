# Foundry Local Overview

Microsoft Foundry Local is an end-to-end local AI solution that provides a
lightweight runtime and SDK for running large language models completely on a
user's own device. It ships with a curated catalog of optimized models, and no
cloud account or dedicated GPU is required.

## Key features

Foundry Local automatically downloads and manages models from its catalog. Once
a model is cached on disk, it can be loaded and used for inference with zero
network calls, which means applications keep working even when the machine is
fully offline.

The runtime uses ONNX Runtime under the hood and automatically selects the best
available hardware acceleration on the device: CPU, GPU, or NPU. This makes the
same application code portable across very different laptops and desktops.

Foundry Local exposes OpenAI-compatible APIs for chat completions and
embeddings. Developers who already know the OpenAI SDK can reuse the same
message format (system, user, and assistant roles) against a local endpoint.

## Why it matters

Running models locally has three main benefits. First, privacy: documents and
questions never leave the machine, which is important for confidential company
data. Second, cost: there are no per-token API charges. Third, availability:
the assistant works on a plane, in a lab without internet, or behind a strict
corporate firewall.

## SDK basics

The Python SDK is installed with `pip install foundry-local-sdk`. The typical
flow is: initialize a `FoundryLocalManager` with a `Configuration`, look up a
model in the catalog by alias (for example `phi-3.5-mini`), call `load()` to
download and load it, then create a chat client or an embedding client from
the model object. Models are unloaded with `unload()` to free memory.

Foundry Local supports Windows, macOS, and Linux. On Windows there is also a
WinML variant of the SDK that adds Windows ML backends.
