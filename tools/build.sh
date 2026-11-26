#!/bin/bash
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Convenience wrapper to build TensorFlow Serving inside Docker.
# Automatically applies known compatibility patches before building.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

function apply_compatibility_patches() {
  echo "== Applying compatibility patches for XLA / GCC..."
  
  # Patch xla/shape.cc to fix noexcept mismatch with newer GCC versions
  local shape_cc="${WORKSPACE_ROOT}/external/local_xla/xla/shape.cc"
  if [[ -f "$shape_cc" ]]; then
    if grep -q 'Shape::Shape(Shape&&) noexcept = default;' "$shape_cc"; then
      sed -i 's/Shape::Shape(Shape&&) noexcept = default;/Shape::Shape(Shape&&) = default;/' "$shape_cc"
      echo "   Patched: shape.cc (move ctor)"
    fi
    if grep -q 'Shape& Shape::operator=(Shape&&) noexcept = default;' "$shape_cc"; then
      sed -i 's/Shape& Shape::operator=(Shape&&) noexcept = default;/Shape& Shape::operator=(Shape&&) = default;/' "$shape_cc"
      echo "   Patched: shape.cc (move assign)"
    fi
  fi

  # Also patch shape.h if it exists to keep declaration/definition consistent
  local shape_h="${WORKSPACE_ROOT}/external/local_xla/xla/shape.h"
  if [[ -f "$shape_h" ]]; then
    sed -i 's/Shape(Shape&&) noexcept;/Shape(Shape&&);/' "$shape_h" 2>/dev/null || true
    sed -i 's/operator=(Shape&&) noexcept;/operator=(Shape&&);/' "$shape_h" 2>/dev/null || true
  fi

  # Recursively find shape.cc if external/ not yet at expected path
  find "${WORKSPACE_ROOT}" -path '*/xla/shape.cc' -not -path '*/.git/*' -print0 2>/dev/null | \
    while IFS= read -r -d '' file; do
      sed -i 's/Shape::Shape(Shape&&) noexcept = default;/Shape::Shape(Shape&&) = default;/' "$file" 2>/dev/null || true
      sed -i 's/Shape& Shape::operator=(Shape&&) noexcept = default;/Shape& Shape::operator=(Shape&&) = default;/' "$file" 2>/dev/null || true
    done
}

function usage() {
  echo "Usage:"
  echo "  $(basename $0) [bazel build flags] <target>"
  echo ""
  echo "Examples:"
  echo "  $(basename $0) -c opt tensorflow_serving/model_servers:tensorflow_model_server"
  echo "  $(basename $0) --config=nativeopt tensorflow_serving/..."
  exit 1
}

# Ensure external/ deps are available before patching
echo "== Fetching dependencies..."
"${SCRIPT_DIR}/run_in_docker.sh" bazel fetch //tensorflow_serving/... 2>/dev/null || true

# Apply patches in the local workspace
apply_compatibility_patches

# Forward everything to run_in_docker.sh for the actual build
echo "== Starting build via run_in_docker.sh..."
exec "${SCRIPT_DIR}/run_in_docker.sh" bazel build "$@"
