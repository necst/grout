package com.necst.controller.runtime;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class Servant extends UnicastRemoteObject implements Service {
    private static final long MEGABYTE = 1024L * 1024L;
    private RuntimeWorker runtimeWorker;
    private Timings timings;
    private Boolean enableWorkerTimers;

    public Servant() throws RemoteException {
        super();
    }

    private static void getHeapInfo(String info) {
    //     Runtime runtime = Runtime.getRuntime();

    //     long maxMemory = runtime.maxMemory();
    //     long totalMemory = runtime.totalMemory();
    //     long freeMemory = runtime.freeMemory();
    //     long usedMemory = totalMemory - freeMemory;

    //     double utilization = (double) usedMemory / maxMemory * 100.0;

    //     System.out.printf("\n[%s] UTILIZATION: %.1f%% | USED: %dMB, TOTAL: %dMB, FREE=%dMB, MAX=%dMB \n", info, utilization, usedMemory/MEGABYTE, totalMemory/MEGABYTE, freeMemory/MEGABYTE, maxMemory/MEGABYTE);
    // 
    }

    @Override
    public void initContext(RuntimeWorkerOptions options) throws RemoteException {
        try{
            this.enableWorkerTimers = options.enableComputationTimers;
            System.out.printf("enableWorkerTimers==options.enableComputationTimers==%s\n",  options.enableComputationTimers ? "true" : "false");
            if(enableWorkerTimers){
                timings = new Timings();
                long start = System.nanoTime();
                runtimeWorker = new RuntimeWorker(options);
                long time = System.nanoTime() - start;
                timings.initContextTime.add(time);
            }else{
                getHeapInfo("INIT");
                runtimeWorker = new RuntimeWorker(options);
            }
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
    }

    @Override
    public void syncOnArray(long arrayHash) throws RemoteException{
        try{
        runtimeWorker.syncToGetArray(arrayHash);
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
    }

    @Override
    public void p2pTransfer(long arrayHash, long sizeInBytes, String workerID) throws RemoteException {
        try{
        runtimeWorker.p2pTransfer(arrayHash, sizeInBytes, workerID);
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
    }

    @Override
    public byte[] getArray(long arrayHash, long offset) throws RemoteException {
        byte[] a;
        try{
            if(enableWorkerTimers){                
                long start = System.nanoTime();       
                a = runtimeWorker.getArray(arrayHash, offset);
                long time = System.nanoTime() - start;
                
                timings.getArrayTime.add(time);
                timings.getArraySize.add(a.length);
            }else{
                getHeapInfo("getArray");
                a = runtimeWorker.getArray(arrayHash, offset);
            }
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
        return a;
    }

    @Override
    public void sendArray(long arrayHash, long arraySize, long offset, byte[] array) throws RemoteException {
        try{
            if(enableWorkerTimers){                
                long start = System.nanoTime();
                if (offset == 0) {
                    runtimeWorker.initializeArray(arrayHash, arraySize);
                }
                runtimeWorker.setArray(arrayHash, offset, array);
                long time = System.nanoTime() - start;

                timings.sendArrayTime.add(time);
                timings.sendArraySize.add(array.length);
            }else{
                getHeapInfo("sendArray");
                //System.out.printf("## SETTING_ARR_FROM_CONTROLLER(ID: %d, size: %d, offset: %d)\n", arrayHash, arraySize, offset);
                if (offset == 0) {
                    runtimeWorker.initializeArray(arrayHash, arraySize);
                }
                runtimeWorker.setArray(arrayHash, offset, array);
            }
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
    }

    @Override
    public void buildKernel(String kernelString, String kernelName, String kernelSignature) throws RemoteException {
        try{
            if(enableWorkerTimers){
                long start = System.nanoTime();
                runtimeWorker.buildKernel(kernelString, kernelName, kernelSignature);
                long time = System.nanoTime() - start;
                timings.buildKernelTime.add(time);
            }else{
                getHeapInfo("buildKernel");
                runtimeWorker.buildKernel(kernelString, kernelName, kernelSignature);
            }
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
    }

    @Override
    public void execKernel(String kernelName, int[] blocks, int[] threadsPerBlock, String[] types, Object[] data) throws RemoteException {
        try{
            if(enableWorkerTimers){
                long start = System.nanoTime();
                //getHeapInfo("execKernel");
                runtimeWorker.executeKernel(kernelName, blocks, threadsPerBlock, types, data);
                long time = System.nanoTime() - start;
                timings.invocationKernelTime.add(time);
            }else{
                getHeapInfo("execKernel");
                runtimeWorker.executeKernel(kernelName, blocks, threadsPerBlock, types, data);
            }
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }

    }

    @Override
    public Timings closeContext() throws RemoteException {
            try{
            if(enableWorkerTimers){
                long start = System.nanoTime();
                runtimeWorker.close();
                runtimeWorker = null;
                System.gc();
                getHeapInfo("afterClosing");
                long time = System.nanoTime() - start;
                timings.closeContextTime.add(time);
                return timings;
            }else{
                getHeapInfo("beforeClosing");
                runtimeWorker.close();
                runtimeWorker = null;
                System.gc();
                getHeapInfo("afterClosing");
                return null;
            }
        }catch(Exception e){
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            throw new RemoteException(exceptionAsString);
        }
    }

}