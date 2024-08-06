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

import com.necst.controller.ControllerException;
import com.necst.controller.ControllerLogger;
import com.necst.controller.runtime.RuntimeWorkerOptions;
import com.necst.controller.runtime.Service;
import com.necst.controller.runtime.Timings;
import com.necst.controller.runtime.WorkerStatistics;
import com.necst.controller.runtime.array.BigByteBuffer;
import com.necst.controller.runtime.array.LittleEndianBigByteBufferView;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.util.ArrayList;

import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;
import com.oracle.truffle.api.TruffleLogger;

public class Worker extends AbstractWorker {
    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.WORKERSMANAGER_LOGGER);
    private final Service service;
    private final String workerId;
    private final String hostname;
    private final Short port;
    private int numActiveComputation = 0; // the current number of active computations within this worker
    AbstractControllerExecutionContext controllerExecutionContext;
    WorkerStatistics workerStatistics = new WorkerStatistics();

    public Worker(String hostname, AbstractControllerExecutionContext controllerExecutionContext, int id) {
        super(hostname, id);
        this.workerId = hostname;
        this.hostname = hostname.split(":")[0];
        this.port = Short.parseShort(hostname.split(":")[1]);
        this.controllerExecutionContext = controllerExecutionContext;
        this.controllerExecutionContext.controllerStatistics.workerStatistics.add(this.workerStatistics);
        workerStatistics.workerID = this.workerId;

        // TODO: get the worker grcuda configuration
        try {
            if(controllerExecutionContext.enableTimers){
                long start = System.nanoTime();
                this.service = (Service) Naming.lookup("rmi://" + this.hostname + ":" + this.port + "/worker");
                RuntimeWorkerOptions workerOptions = new RuntimeWorkerOptions();
                workerOptions.enableWorkerTimers = true;
                workerOptions.workers = controllerExecutionContext.options.getWorkersInfo();
                service.initContext(workerOptions);
                long time = System.nanoTime() - start;
                workerStatistics.initContextTime.add(time);
            }else{
                this.service = (Service) Naming.lookup("rmi://" + this.hostname + ":" + this.port + "/worker");
                RuntimeWorkerOptions workerOptions = new RuntimeWorkerOptions();
                workerOptions.enableWorkerTimers = false;
                workerOptions.workers = controllerExecutionContext.options.getWorkersInfo();
                service.initContext(workerOptions);
            }

        } catch (NotBoundException | MalformedURLException | RemoteException e) {
            throw new RuntimeException(e);
        }
    }

    public int getNumActiveComputation() {
        return numActiveComputation;
    }

    public void setNumActiveComputation(int numActiveComputation) {
        this.numActiveComputation = numActiveComputation;
    }
    
    @Override
    public void buildKernel(String ptx, String kernelName, String parameters) {
        LOGGER.fine("BUILDING KERNEL: " + kernelName + " on " + workerId);

        try {
            if(controllerExecutionContext.enableTimers){
                long start = System.nanoTime();
                service.buildKernel(ptx, kernelName, parameters);
                long time = System.nanoTime() - start;
                workerStatistics.buildKernelTime.add(time);
            }else{
                service.buildKernel(ptx, kernelName, parameters);
            }
        } catch (RemoteException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void executeKernel(String kernelName, int[] gridSize, int[] blockSize, String[] types, Object[] data) {
        LOGGER.fine("EXECUTING KERNEL: " + kernelName + " on " + workerId);
        try {
            if(controllerExecutionContext.enableTimers){
                long start = System.nanoTime();
                service.execKernel(kernelName, gridSize, blockSize, types, data);
                long time = System.nanoTime() - start;
                workerStatistics.invocationKernelTime.add(time);
            }else{
                service.execKernel(kernelName, gridSize, blockSize, types, data);
            }
        } catch (RemoteException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void sendControllerArray(BigByteBuffer bigByteBuffer, long id, long sizeBytes) {
        LOGGER.fine("SEND ControllerArray(ID: " + id + ") to " + workerId);
        
        long start = -1;
        if(controllerExecutionContext.enableTimers){
            try{
                this.service.syncOnArray(id); // this allow us to record the actual execution time of the get without considering runtime of kernels
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }            
            this.workerStatistics.dataReceivedFromController += bigByteBuffer.size;
            start = System.nanoTime();
        }

        for (int i = 0; i < bigByteBuffer.numInnerBuffers(); i++) {
            try {
                byte[] arr = bigByteBuffer.getBufferGivenInternalIndex(i).array();
                service.sendArray(
                        id,
                        sizeBytes,
                        (long) i * BigByteBuffer.ALLOWED_INTEGER_MAX_VALUE,
                        arr
                );
            } catch (RemoteException e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }

        if(controllerExecutionContext.enableTimers){
            long time = System.nanoTime() - start;
            workerStatistics.sendArrayTime.add(time);
            workerStatistics.sendArraySize.add(sizeBytes);
        }
    }

    @Override
    public LittleEndianBigByteBufferView getRemoteArray(long id, long length) {
        LOGGER.fine("GETTING LittleEndianBigByteBufferView(ID: " + id + ") from " + workerId);

        long start = -1;
        if(controllerExecutionContext.enableTimers){
            try{
                this.service.syncOnArray(id); // this allow us to record the actual execution time of the get without considering runtime of kernels
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
            this.workerStatistics.dataSentToController += length;
            start = System.nanoTime();
        }        
        
        LittleEndianBigByteBufferView array = new LittleEndianBigByteBufferView(length, true);

        int numParts = (int) (length / BigByteBuffer.ALLOWED_INTEGER_MAX_VALUE);
        if (length % BigByteBuffer.ALLOWED_INTEGER_MAX_VALUE!= 0) {
            numParts++;
        }
        byte[] part;
        for (int i = 0; i < numParts; i++) {

            try {
                part = service.getArray(id, i * BigByteBuffer.ALLOWED_INTEGER_MAX_VALUE);
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }

            array.bigByteBuffer.replaceBuffer(i,part);
            
            /*
            for (int j = 0; j < part.length; j++) {
                array.setByte((long) i * Integer.MAX_VALUE + j, part[j]);
            }
            */
        }

        if(controllerExecutionContext.enableTimers){
            long time = System.nanoTime() - start;
            workerStatistics.getArrayTime.add(time);
            workerStatistics.getArraySize.add(length);
        }

        return array;
    }

    @Override
    public void p2pTransfer(long arrayID, long sizeBytes, String destinationWorker) {
        LOGGER.fine("P2P transfer of ID("+ arrayID +") from " + destinationWorker+ " to " + this.workerId);
        long start = -1;
        if(controllerExecutionContext.enableTimers){
            try{
                this.service.syncOnArray(arrayID); // this allows us to record the actual execution time of the get without considering runtime of kernels
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
            start = System.nanoTime();
        }

        try {
            service.p2pTransfer(arrayID, sizeBytes, destinationWorker);
        } catch (RemoteException e) {
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RuntimeException(exceptionAsString);
        }

        if(controllerExecutionContext.enableTimers){
            long time = System.nanoTime() - start;
            workerStatistics.p2pTime.add(time);
            workerStatistics.p2pSize.add(sizeBytes);
        }

    }

    @Override
    public Timings close() {
        try {
            if(controllerExecutionContext.enableTimers){
                long start = System.nanoTime();
                Timings timings = service.closeContext();
                if(timings==null)
                    System.out.println("timings is null");
                long time = System.nanoTime() - start;
                workerStatistics.closeContextTime.add(time);
                return timings;
            }else{
                service.closeContext();
                return null;
            }
        } catch (RemoteException e) {
            throw new ControllerException(e.toString());
        }
    }


    public String toString() {
        return workerId;
    }

}