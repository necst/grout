/*
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

import java.io.File;

public class GrOUTOptionMap {
    public static final ExecutionPolicyEnum DEFAULT_EXECUTION_POLICY = ExecutionPolicyEnum.ASYNC;
    public static final DependencyPolicyEnum DEFAULT_DEPENDENCY_POLICY = DependencyPolicyEnum.WITH_CONST;
    public static final RetrieveNewStreamPolicyEnum DEFAULT_RETRIEVE_STREAM_POLICY = RetrieveNewStreamPolicyEnum.REUSE;
    public static final RetrieveParentStreamPolicyEnum DEFAULT_PARENT_STREAM_POLICY = RetrieveParentStreamPolicyEnum.MULTIGPU_DISJOINT;
    public static final DeviceSelectionPolicyEnum DEFAULT_DEVICE_SELECTION_POLICY = DeviceSelectionPolicyEnum.MINMAX_TRANSFER_TIME;
    public static final MemAdviserEnum DEFAULT_MEM_ADVISE_POLICY = MemAdviserEnum.NONE;
    public static final boolean DEFAULT_INPUT_PREFETCH = true; // Value obtained from the input flags;
    public static final boolean DEFAULT_FORCE_STREAM_ATTACH = false;
    public static final boolean DEFAULT_ENABLE_COMPUTATION_TIMERS = false;
    public static final Integer DEFAULT_NUMBER_OF_GPUs = 8;
    public static final String DEFAULT_BANDWIDTH_MATRIX = System.getenv("GRCUDA_HOME") + File.separatorChar +
            "projects" + File.separatorChar + "resources" + File.separatorChar +
            "connection_graph" + File.separatorChar + "datasets" + File.separatorChar + "connection_graph.csv";
    public static final double DEFAULT_DATA_THRESHOLD = 0.1;
    public static final String DEFAULT_EXPORT_DAG = "false";

    public GrOUTOptionMap() {
    }
}
