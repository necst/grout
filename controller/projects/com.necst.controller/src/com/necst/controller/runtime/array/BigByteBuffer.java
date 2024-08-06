package com.necst.controller.runtime.array;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class BigByteBuffer {

    public long size;
    private ByteBuffer[] buffers;
    public static int ALLOWED_INTEGER_MAX_VALUE = (1<<30); // must be divisible by the max length of supported types & equal to LittleEndian..View

    public BigByteBuffer(long size, boolean empty) {
        this.size = size;
        int chunks = (int) (size / ALLOWED_INTEGER_MAX_VALUE);
        int lastSmallChunkSize = (int) (size % ALLOWED_INTEGER_MAX_VALUE);

        int chunks_num = chunks;
        if (lastSmallChunkSize != 0)
            chunks_num++;

        buffers = new ByteBuffer[chunks_num];
        //System.out.printf("Created ByteBuffer[%d]\n", chunks_num);

        if(!empty){
            for (int i = 0; i < chunks; i++) {
                //System.out.printf("Allocating ByteBuffer[%d] = ByteBuffer.allocate(%d)\n", i, ALLOWED_INTEGER_MAX_VALUE);
                buffers[i] = ByteBuffer.allocate(ALLOWED_INTEGER_MAX_VALUE);
                buffers[i].order(ByteOrder.LITTLE_ENDIAN);
            }

            if (lastSmallChunkSize != 0) {
                //System.out.printf("Allocating ByteBuffer[%d] = ByteBuffer.allocate(%d)\n", chunks_num - 1, lastSmallChunkSize);
                buffers[chunks_num - 1] = ByteBuffer.allocate(lastSmallChunkSize);
                buffers[chunks_num - 1].order(ByteOrder.LITTLE_ENDIAN);
            }
        }
        //System.out.printf("Created a BigByteBuffer with %d chunks_num and %d chunks and lastSmallChunkSize: %d\n", chunks_num, chunks, lastSmallChunkSize);
    }

    public int numInnerBuffers(){
        return this.buffers.length;
    }

    public ByteBuffer getBuffer(long index) {
        int idx = (int) (index/ALLOWED_INTEGER_MAX_VALUE);
        //if(idx < 0)
        //    System.out.printf("Obtained INDEX: %d --> computed buffer[%d]\n", index, idx);
        return buffers[idx];
    }

    public ByteBuffer getBufferGivenInternalIndex(int index){
        return buffers[index];
    }

    public void replaceBuffer(int index, byte[] byteArray){
        buffers[index] = ByteBuffer.wrap(byteArray);
        buffers[index].order(ByteOrder.LITTLE_ENDIAN);
    }
}
