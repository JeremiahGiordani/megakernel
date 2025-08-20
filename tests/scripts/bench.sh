#!/usr/bin/env bash

SGEMM_MC=64 SGEMM_NC=960 SGEMM_KC=640 ./build/mk --bench gemm --m 2048 --n 960 --k 5120 --repeats 20 --warmups 3 --threads 8

# ./build/mk --bench gemm --m 2048 --n 960 --k 5120 --repeats 20 --warmups 3 --threads 8