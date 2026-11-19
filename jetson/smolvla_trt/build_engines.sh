#!/usr/bin/env bash
# Build the three TensorRT engines from export.py's ONNX graphs. Engines belong to one GPU and
# one TensorRT version: build them on the machine that runs them (the Jetson).
#
#   ./build_engines.sh MODEL_DIR [fp16|fp32] [ENGINE_DIR]
#
# fp16 (default) writes MODEL_DIR/*.engine; fp32 writes MODEL_DIR/fp32/*.engine, to compare.
# trtexec times each engine after building it; the numbers are in ENGINE_DIR/<part>.build.log.
set -euo pipefail
MODEL=${1:?usage: build_engines.sh MODEL_DIR [fp16|fp32] [ENGINE_DIR]}
PRECISION=${2:-fp16}
if [[ $PRECISION == fp32 ]]; then OUT=${3:-$MODEL/fp32}; else OUT=${3:-$MODEL}; fi
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
case $PRECISION in fp16) FLAGS=(--fp16) ;; fp32) FLAGS=() ;; *) echo "fp16 or fp32" >&2; exit 2 ;; esac
mkdir -p "$OUT"
for part in vision prefix expert; do
  echo "== $part.engine ($PRECISION) from $MODEL/$part.onnx"
  start=$(date +%s)
  "$TRTEXEC" --onnx="$MODEL/$part.onnx" --saveEngine="$OUT/$part.engine" "${FLAGS[@]}" \
             --memPoolSize=workspace:2048M --avgRuns=20 >"$OUT/$part.build.log" 2>&1 \
    || { tail -30 "$OUT/$part.build.log"; exit 1; }
  echo "   built in $(( $(date +%s) - start )) s, $(du -h "$OUT/$part.engine" | cut -f1);" \
       "$(grep -m1 'GPU Compute Time' "$OUT/$part.build.log" | sed 's/.*GPU Compute Time: //')"
done
