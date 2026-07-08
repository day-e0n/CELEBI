# Computation VRAM Probe

Purpose: baseline Sirius GPU query execution peak VRAM, with fixed-page reuse disabled.

Input: `/mnt/nvme/dataset`
Queries: `q1,q3,q4,q5,q6,q7,q8,q10,q11,q12,q13,q14,q15,q16,q17,q19,q20,q21,q22`
Devices: `0,1`
GPU usage limit per GPU: `8GB`
Sampling interval: `100 ms`

Files:
- `query_vram.csv`: per-query runtime/status/peak process VRAM
- `summary.csv`: mean/p50/p95/max peak VRAM and estimated headroom
- `cases/q<N>.json`: compact child status and error details
