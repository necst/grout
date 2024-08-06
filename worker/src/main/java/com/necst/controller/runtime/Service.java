package com.necst.controller.runtime;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface Service extends Remote {

    void initContext(RuntimeWorkerOptions options) throws RemoteException;

    byte[] getArray(long arrayHash, long offset) throws RemoteException;

    void sendArray(long arrayHash, long arraySize, long offset, byte[] array) throws RemoteException;

    void buildKernel(String kernelString, String kernelName, String kernelSignature) throws RemoteException;

    void execKernel(String kernelName, int[] blocks, int[] threadsPerBlock, String[] types, Object[] data) throws RemoteException;

    void syncOnArray(long arrayHash) throws RemoteException;

    void p2pTransfer(long arrayHash, long sizeInBytes, String workerID) throws RemoteException;

    Timings closeContext() throws RemoteException;
}