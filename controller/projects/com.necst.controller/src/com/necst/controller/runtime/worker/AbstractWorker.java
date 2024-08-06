package com.necst.controller.runtime.worker;

import com.necst.controller.runtime.AbstractNode;
import com.necst.controller.runtime.Timings;
import com.necst.controller.runtime.array.BigByteBuffer;
import com.necst.controller.runtime.array.LittleEndianBigByteBufferView;

public abstract class AbstractWorker extends AbstractNode {
    private final int id;

    public AbstractWorker(String nodeIdentifier, int id) {
        super(nodeIdentifier);
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public abstract void buildKernel(String ptx, String kernelName, String parameters);

    public abstract void executeKernel(String kernelName, int[] blockSizeArray, int[] gridSizeArray, String[] types, Object[] data);

    public abstract void sendControllerArray(BigByteBuffer bigByteBuffer, long id, long sizeBytes);

    public abstract LittleEndianBigByteBufferView getRemoteArray(long id, long length);

    public abstract void p2pTransfer(long arrayID, long sizeBytes, String workerID);

    public abstract Timings close();
}
