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
#include "taso/cuda_helper.h"
using namespace taso;

float BatchNorm::get_min_epsilon(void)
{
  return CUDNN_BN_MIN_EPSILON;
}

void BatchNorm::map(void)
{
  assert(inputs[0].numDim == 4);
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  int inputN = inputs[0].dim[0];
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, inputC, 1, 1));
#ifdef DO_TRAINING
  checkCUDA(cudaMalloc(&runningMean, sizeof(DATATYPE) * inputC));
  checkCUDA(cudaMalloc(&runningVar, sizeof(DATATYPE) * inputC));
  checkCUDA(cudaMalloc(&saveMean, sizeof(DATATYPE) * inputC));
  checkCUDA(cudaMalloc(&saveVar, sizeof(DATATYPE) * inputC));
  checkCUDA(cudaMalloc(&biasPtr, sizeof(DATATYPE) * inputC));
  checkCUDA(cudaMalloc(&scalePtr, sizeof(DATATYPE) * inputC));
   initialize scale to ones and bias to zeros
  assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
    scalePtr, inputC, 1.0f);
  assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
    biasPtr, inputC, 0.0f);
  assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
    runningMean, inputC, 0.0f);
  assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
    runningVar, inputC, 0.0f);
#endif
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void BatchNorm::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
#ifdef DO_TRAINING
  checkCUDA(cudaFree(runningMean));
  checkCUDA(cudaFree(runningVar));
  checkCUDA(cudaFree(saveMean));
  checkCUDA(cudaFree(saveVar));
  checkCUDA(cudaFree(biasPtr));
  checkCUDA(cudaFree(scalePtr));
#endif
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void BatchNorm::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const float eps = epsilon;
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  //int inputC = inputs[0].dim[1];
#ifdef DO_TRAINING 
  if (model->isTraining) {
    assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
      runningMean, inputC, 0.0f);
    assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
      runningVar, inputC, 0.0f);
    checkCUDNN(cudnnBatchNormalizationForwardTraining(
      model->dnn, mode, &alpha, &beta, inputTensor, inputs[0].data_ptr,
      outputTensor, outputs[0].data_ptr, biasTensor, scalePtr, biasPtr,
      1.0, runningMean, runningVar, eps, saveMean, saveVar));
  } else {
#endif
    checkCUDNN(cudnnBatchNormalizationForwardInference(
      model->dnn, mode, &alpha, &beta, inputTensor, inputs[0].data_ptr,
      outputTensor, outputs[0].data_ptr, biasTensor, inputs[1].data_ptr, inputs[2].data_ptr,
      inputs[3].data_ptr, inputs[4].data_ptr, eps));
#ifdef DO_TRAINING 
  }
#endif
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_batchnorm_cost(BatchNorm* bn)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputN = bn->inputs[0].dim[0];
  int inputC = bn->inputs[0].dim[1];
  int inputH = bn->inputs[0].dim[2];
  int inputW = bn->inputs[0].dim[3];
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, inputC, 1, 1));
#ifdef DO_TRAINING
  assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
    scalePtr, inputC, 0.5f);
  assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
    biasPtr, inputC, 0.5f);
#endif
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
#ifdef DO_TRAINING
    if (isTraining) {
      assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
        runningMean, inputC, 0.0f);
      assign_kernel<<<GET_BLOCKS(inputC), CUDA_NUM_THREADS>>>(
        runningVar, inputC, 0.0f);
      checkCUDNN(cudnnBatchNormalizationForwardTraining(
        dnn, mode, &alpha, &beta, inputTensor, inputPtr,
        outputTensor, outputPtr, biasTensor, scalePtr, biasPtr,
        1.0, runningMean, runningVar, CUDNN_BN_MIN_EPSILON,
        saveMean, saveVar));
    } else {
#endif
      checkCUDNN(cudnnBatchNormalizationForwardInference(
        dnn, mode, &alpha, &beta, inputTensor, inputPtr,
        outputTensor, outputPtr, biasTensor, scalePtr, biasPtr,
        runningMean, runningVar, CUDNN_BN_MIN_EPSILON));
#ifdef DO_TRAINING
    }
#endif
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  bn->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("measure[BatchNorm]: i(%d %d %d %d) cost(%.4lf)\n",
           inputN, inputC, inputH, inputW, bn->runtime);
}

