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
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.necst.controller.runtime.worker;

import com.necst.controller.runtime.Service;
import com.necst.controller.runtime.Timings;
import com.necst.controller.runtime.WorkerStatistics;
import com.necst.controller.runtime.array.BigByteBuffer;
import com.necst.controller.runtime.array.LittleEndianBigByteBufferView;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;

public class MockWorker extends AbstractWorker {
    private final Service service;
    private final String workerId;
    private final String hostname;
    private final Short port;
    private int numActiveComputation = 0; // the current number of active computations within this worker
    AbstractControllerExecutionContext controllerExecutionContext;
    WorkerStatistics workerStatistics = new WorkerStatistics();

    public MockWorker(String hostname, AbstractControllerExecutionContext controllerExecutionContext, int id) {
        super(hostname, id);
        this.workerId = hostname;
        this.hostname = hostname.split(":")[0];
        this.port = Short.parseShort(hostname.split(":")[1]);
        this.controllerExecutionContext = controllerExecutionContext;
        if (controllerExecutionContext!=null)
            this.controllerExecutionContext.controllerStatistics.workerStatistics.add(this.workerStatistics);

        this.service = null;
    }

    public int getNumActiveComputation() {
        return numActiveComputation;
    }

    public void setNumActiveComputation(int numActiveComputation) {
        this.numActiveComputation = numActiveComputation;
    }

    public void buildKernel(String ptx, String kernelName, String parameters) {
        System.out.println("BUILDING KERNEL");
    }

    public void executeKernel(String kernelName, int[] blockSizeArray, int[] gridSizeArray, String[] types, Object[] data) {
        System.out.println("EXECUTING KERNEL");
    }

    public void sendControllerArray(BigByteBuffer bigByteBuffer, long id, long sizeBytes) {
        this.workerStatistics.dataReceivedFromController += bigByteBuffer.size;
        System.out.println("SENDING ARRAY");
    }

    public LittleEndianBigByteBufferView getRemoteArray(long id, long length) {
        System.out.printf("GETTING ARRAY len=%d\n", length);
        this.workerStatistics.dataSentToController += length;
        return new LittleEndianBigByteBufferView(length);
    }

    @Override
    public void p2pTransfer(long arrayID, long sizeBytes, String workerID) {
        System.out.printf("p2pTransfer ARRAY len=%d from workerID=%s\n", arrayID, workerID);

    }

    public Timings close() {
        System.out.println("CLOSING");
        Timings timings = new Timings();
        timings.buildKernelTime.add(42L);
        timings.initContextTime.add(42L);
        timings.invocationKernelTime.add(42L);
        timings.getArrayTime.add(42L);
        timings.getArraySize.add(42);
        timings.sendArraySize.add(42);
        timings.closeContextTime.add(42L);
        timings.sendArrayTime.add(42L);

        return timings;
    }


    public String toString() {
        return "GrWorker(id=" + hostname + ":" + port + ")";
    }

}