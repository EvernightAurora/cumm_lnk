// Copyright 2021 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
namespace tv {

namespace gemm {
namespace constants {

constexpr int kStridedAxis = 0;
constexpr int kContigAxis = 1;
constexpr int kWarpSize = 32;
constexpr int kOptimAccessBit = 128;
constexpr int kOptimAccess = kOptimAccessBit / 8;

} // namespace constants

} // namespace gemm
} // namespace tv