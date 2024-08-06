/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) 2024 NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.necst.controller.runtime.worker.options;

import org.graalvm.options.OptionKey;

public final class GrOUTOptions {
    public static final OptionKey<Boolean> EnableComputationTimers = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_ENABLE_COMPUTATION_TIMERS);

    public static final OptionKey<String> ExecutionPolicy = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_EXECUTION_POLICY.toString());

    public static final OptionKey<String> DependencyPolicy = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_DEPENDENCY_POLICY.toString());

    public static final OptionKey<String> RetrieveNewStreamPolicy = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_RETRIEVE_STREAM_POLICY.toString());

    public static final OptionKey<String> RetrieveParentStreamPolicy = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_PARENT_STREAM_POLICY.toString());

    public static final OptionKey<Boolean> ForceStreamAttach = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_FORCE_STREAM_ATTACH);

    public static final OptionKey<Boolean> InputPrefetch = new OptionKey<>(GrOUTOptionMap.DEFAULT_INPUT_PREFETCH);

    public static final OptionKey<Integer> NumberOfGPUs = new OptionKey<>(GrOUTOptionMap.DEFAULT_NUMBER_OF_GPUs);

    public static final OptionKey<String> DeviceSelectionPolicy = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_DEVICE_SELECTION_POLICY.toString());

    public static final OptionKey<String> MemAdvisePolicy = new OptionKey<>(
            GrOUTOptionMap.DEFAULT_MEM_ADVISE_POLICY.toString());

    public static final OptionKey<String> BandwidthMatrix = new OptionKey<>(GrOUTOptionMap.DEFAULT_BANDWIDTH_MATRIX);

    public static final OptionKey<Double> DataThreshold = new OptionKey<>(GrOUTOptionMap.DEFAULT_DATA_THRESHOLD);

    public static final OptionKey<String> ExportDAG = new OptionKey<>(GrOUTOptionMap.DEFAULT_EXPORT_DAG);
}
