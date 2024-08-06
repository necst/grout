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
package com.necst.controller.runtime.array;

import com.necst.controller.Type;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A non-owning view over native memory provided. No bounds checks are performed.
 */
public class LittleEndianBigByteBufferView {
    public static int ALLOWED_INTEGER_MAX_VALUE = BigByteBuffer.ALLOWED_INTEGER_MAX_VALUE; // must be divisible by the max length of supported types & equal to BigByteBuffer

    public BigByteBuffer getBigByteBuffer() {
        return bigByteBuffer;
    }

    public BigByteBuffer bigByteBuffer;

    private int getInnerIndex(long index){   
        // index MUST be the GLOBAL index of the array of BYTES --> therefore convert the index by multipling it by the size before using this function
        // given an array with P partitions of size ALLOWED_INTEGER_MAX_VALUE
	    int res =  (int) (index - (int)(index/ALLOWED_INTEGER_MAX_VALUE) * ALLOWED_INTEGER_MAX_VALUE);
        //if(res < 0)
        //    System.out.printf("%d - %d/(%d-1) * %d = %d", index, index, ALLOWED_INTEGER_MAX_VALUE, ALLOWED_INTEGER_MAX_VALUE, res);
        //if(index == 4194303*Type.SINT32.getSizeBytes())
        //    System.out.printf("%d - %d/(%d-1)[%d] * %d = %d\n", index, index, ALLOWED_INTEGER_MAX_VALUE, (int)(index/(ALLOWED_INTEGER_MAX_VALUE-1)), ALLOWED_INTEGER_MAX_VALUE, res);
        return res;
    }

    public void setByte(long index, byte value) {
	    bigByteBuffer.getBuffer(index).put(getInnerIndex(index), value);
    }

    public void setChar(long index, char value) {
        bigByteBuffer.getBuffer(index).putChar(getInnerIndex(index), value);
    }

    public void setShort(long index, short value) {
        bigByteBuffer.getBuffer(index).putShort(getInnerIndex(index), value);
    }

    public void setInt(long index, int value) {        
        int idx = (int) (index / (ALLOWED_INTEGER_MAX_VALUE));
        ByteBuffer buffer = bigByteBuffer.getBuffer(index);
        
        //if(index == 4194303*Type.SINT32.getSizeBytes())
        //    System.out.printf("buffer[%d], idx=%d, original_idx=%d # putInt(%d)\n", idx, index, index/Type.SINT32.getSizeBytes(), getInnerIndex(index));
        buffer.putInt(getInnerIndex(index), value);
    }

    public void setLong(long index, long value) {
        bigByteBuffer.getBuffer(index).putLong(getInnerIndex(index), value);
    }

    public void setFloat(long index, float value) {
        bigByteBuffer.getBuffer(index).putFloat(getInnerIndex(index), value);
    }

    public void setDouble(long index, double value) {
        bigByteBuffer.getBuffer(index).putDouble(getInnerIndex(index), value);
    }

    public byte getByte(long index) {
        return bigByteBuffer.getBuffer(index).get(getInnerIndex(index));
    }

    public char getChar(long index) {
        return bigByteBuffer.getBuffer(index).getChar(getInnerIndex(index));
    }

    public short getShort(long index) {
        return bigByteBuffer.getBuffer(index).getShort(getInnerIndex(index));
    }

    public int getInt(long index) {
        ByteBuffer current = bigByteBuffer.getBuffer(index);
        return bigByteBuffer.getBuffer(index).getInt(getInnerIndex(index));
    }

    public long getLong(long index) {
        return bigByteBuffer.getBuffer(index).getLong(getInnerIndex(index));
    }

    public float getFloat(long index) {
        return bigByteBuffer.getBuffer(index).getFloat(getInnerIndex(index));
    }

    public double getDouble(long index) {
        return bigByteBuffer.getBuffer(index).getDouble(getInnerIndex(index));
    }

    public LittleEndianBigByteBufferView(long numBytes) {
        this.bigByteBuffer = new BigByteBuffer(numBytes, false);
    }

    public LittleEndianBigByteBufferView(long numBytes, boolean empty){
        this.bigByteBuffer = new BigByteBuffer(numBytes, empty);
    }
}




