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
package com.necst.controller;

import com.oracle.truffle.api.Option;
import org.graalvm.options.OptionCategory;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionStability;

@Option.Group(ControllerLanguage.ID)
public final class ControllerOptions {
    @Option(category = OptionCategory.USER, help = "Choose the heuristic that manages how GPU computations are mapped to workers, if multiple workers are available.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> WorkerSelectionPolicy = new OptionKey<>(ControllerOptionMap.DEFAULT_WORKER_SELECTION_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "Dependency policy (no-const, with-const)", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> DependencyPolicy = new OptionKey<>(ControllerOptionMap.DEFAULT_DEPENDENCY_POLICY.toString());

    @Option(category = OptionCategory.USER, help = "List of comma separated ip:port of the workers (es. '192.168.0.42:2425, 192.168.0.11:232425') ", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> WorkersNetInfo = new OptionKey<>(ControllerOptionMap.DEFAULT_WORKERS_INFO);

    @Option(category = OptionCategory.USER, help = "Use or not the Mock worker", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> EnableMockWorker = new OptionKey<>(ControllerOptionMap.DEFAULT_MOCK_WORKER);

    @Option(category = OptionCategory.USER, help = "Enable timers", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> EnableTimers = new OptionKey<>(ControllerOptionMap.DEFAULT_ENABLE_TIMERS);

    @Option(category = OptionCategory.USER, help = "List of integer values representing the steps to be taken", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> VectorStep = new OptionKey<>(ControllerOptionMap.DEFUALT_VECTOR_STEP);

    @Option(category = OptionCategory.USER, help = "Path of the link bandwidth matrix to calculate the minimum transfer time policy", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> LinkBandwidthPath = new OptionKey<>(ControllerOptionMap.DEFAULT_LINK_BANDWIDTH_PATH);
}

