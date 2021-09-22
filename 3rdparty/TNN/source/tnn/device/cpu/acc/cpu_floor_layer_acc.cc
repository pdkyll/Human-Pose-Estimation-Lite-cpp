// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/device/cpu/acc/cpu_unary_layer_acc.h"

#include <cmath>

namespace TNN_NS {

typedef struct floor_operator : unary_operator {
    virtual float operator()(float in) {
        return std::floor(in);
    }
} FLOOR_OP;

DECLARE_UNARY_ACC(Floor, LAYER_FLOOR, FLOOR_OP);

REGISTER_CPU_ACC(Floor, LAYER_FLOOR);

}  // namespace TNN_NS
