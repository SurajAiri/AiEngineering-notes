# 🐳 Containerization & DevOps for AI

## 📚 Overview

AI systems have unique deployment challenges: large model files, GPU requirements, long cold-start times, and complex dependency chains (CUDA, PyTorch, sentence-transformers). This note covers Docker patterns for ML workloads, GPU container configuration, Kubernetes basics for AI inference, and CI/CD pipelines that handle model artifacts.

> 📌 _Your beautiful RAG pipeline is worthless if it takes 20 minutes to deploy or crashes because CUDA versions don't match._

---

## 🎯 Learning Objectives

- **Build** optimized Docker images for AI workloads (multi-stage, layer caching)
- **Configure** GPU passthrough for containerized inference
- **Design** CI/CD pipelines that handle model artifacts and large dependencies
- **Deploy** AI services with Kubernetes (basic patterns)
- **Manage** model versioning and artifact storage

---

## 🧠 Sections (To Be Written)

### 1. Docker for AI Workloads

- Base images: python-slim vs nvidia/cuda vs huggingface
- Multi-stage builds (separate build vs runtime)
- Layer caching for pip install (requirements.txt ordering)
- .dockerignore for AI projects (exclude model files, datasets)
- Image size optimization (slim vs full, removing build deps)

### 2. GPU Containers

- NVIDIA Container Toolkit setup
- CUDA version matching (driver vs runtime vs toolkit)
- Docker Compose with GPU allocation
- GPU memory limits and sharing
- CPU-only fallback patterns

### 3. Dependency Management

- Poetry/pip-tools for reproducible installs
- PyTorch CPU vs GPU wheel selection
- Pinning transitive dependencies
- Model file management (download at build vs runtime)
- Private model registries

### 4. Kubernetes Basics for AI

- Deployment vs StatefulSet for inference
- GPU scheduling and node affinity
- Health checks for model-loaded services
- Horizontal scaling for embedding services
- Resource requests/limits for GPU workloads

### 5. CI/CD Pipelines

- GitHub Actions for AI projects
- Model artifact caching in CI
- Automated evaluation in PR checks
- Staging vs production model promotion
- Rollback strategies for model updates

### 6. Common Pitfalls

| Symptom                | Cause                            | Fix                   |
| ---------------------- | -------------------------------- | --------------------- |
| 15GB Docker image      | Full PyTorch + CUDA in one stage | Multi-stage build     |
| Container crash on GPU | CUDA version mismatch            | Pin nvidia base image |
| Slow CI builds         | Downloading models every run     | Cache model artifacts |
| OOM in K8s             | No resource limits               | Set memory/GPU limits |

---

## 📖 Resources

- NVIDIA Container Toolkit documentation
- Docker multi-stage build best practices
- Kubernetes GPU scheduling guide

---

## ➡️ Next Steps

Continue to **Phase 0.5: ML Foundations** → [Transformer Architecture](../phase_0_5_ml_foundations/00_transformer_architecture.md)
