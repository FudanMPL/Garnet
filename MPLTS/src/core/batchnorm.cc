/* Copyright 2020 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "taso/ops.h"
using namespace taso;

TensorHandle Graph::batchnorm(const TensorHandle _input,
                              const TensorHandle _scale,
                              const TensorHandle _bias,
                              const TensorHandle _mean,
                              const TensorHandle _var,
                              const float _epsilon)
{
  Op op = model->get_or_create_batchnorm(*_input, *_scale, *_bias,
                                         *_mean, *_var, _epsilon);
  add_edge(_input->op, op, _input->idx, 0);
  add_edge(_scale->op, op, _scale->idx, 1);
  add_edge(_bias->op, op, _bias->idx, 2);
  add_edge(_mean->op, op, _mean->idx, 3);
  add_edge(_var->op, op, _var->idx, 4);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_batchnorm(const Tensor& _input,
                                  const Tensor& _scale,
                                  const Tensor& _bias,
                                  const Tensor& _mean,
                                  const Tensor& _var,
                                  const float _epsilon)
{
  // key is (inputN, inputC, inputH, inputW)
  BatchNormKey key(_input);
  BatchNorm* bnOp;
  if(batchnorm.find(key) != batchnorm.end()) {
    bnOp = batchnorm[key];
  } else {
    bnOp = new BatchNorm(this, _input, _scale, _bias, _mean, _var, _epsilon);
    measure_batchnorm_cost(bnOp);
    batchnorm[key] = bnOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = bnOp;
  return ret;
}

BatchNorm::BatchNorm(Model* _model,
                     const Tensor& _input,
                     const Tensor& _scale,
                     const Tensor& _bias,
                     const Tensor& _mean,
                     const Tensor& _var,
                     const float _epsilon)
: OpBase(_input, _scale, _bias, _mean, _var, _model, OP_BATCHNORM)
{
  epsilon = _epsilon < 0 ? get_min_epsilon() : _epsilon;
  assert(epsilon >= get_min_epsilon());
  assert(_input.numDim == 4);
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

BatchNorm::~BatchNorm(void)
{}

bool BatchNorm::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

bool BatchNorm::get_float_parameter(PMParameter para, float* value)
{
  switch (para) {
    case PM_EPSILON:
    {
      *value = epsilon;
      return true;
    }
    default:
      return OpBase::get_float_parameter(para, value);
  }
}

void BatchNorm::collect_costs(float& exe_time, float& flops,
                              float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  flops += outputSize * 2;
  mem_acc += inputSize;
  num_kernels += 1;
  printf("        cost[BatchNorm]: i(%d %d %d %d) cost(%.4lf) total_cost(%.4lf)\n",
          inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2], inputs[0].dim[3],
          runtime, exe_time);
}

// key is (_input)
BatchNormKey::BatchNormKey(const Tensor& _input)
{
  int idx = 0;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}
