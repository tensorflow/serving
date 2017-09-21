#!/usr/bin/env bash

cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )

for cpu_id in $(seq 1 $cpus)
do
    bazel-bin/tensorflow_serving/example/syntaxnet_client --server=$1 --interactive=false 2>&1 > log_$cpu_id.txt&
done

wait