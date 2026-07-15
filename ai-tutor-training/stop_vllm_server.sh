#!/usr/bin/env bash
pkill -9 -f uvicorn || true
pkill -9 -f multiprocess.spawn || true
pkill -9 -f VLLM::EngineCore || true
pkill -9 -f vllm_server.py || true