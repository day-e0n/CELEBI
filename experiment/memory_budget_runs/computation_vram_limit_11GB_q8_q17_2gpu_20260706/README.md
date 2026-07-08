# Computation VRAM Probe

Purpose: baseline Sirius GPU query execution peak VRAM, with fixed-page reuse disabled.

Input: `/mnt/nvme/dataset`
Queries: `q8,q17`
Devices: `0,1`
GPU usage limit per GPU: `11GB`
Sampling interval: `100 ms`

Files:
- `query_vram.csv`: per-query runtime/status/peak process VRAM
- `summary.csv`: mean/p50/p95/max peak VRAM and estimated headroom
- `cases/q<N>.json`: compact child status and error details
