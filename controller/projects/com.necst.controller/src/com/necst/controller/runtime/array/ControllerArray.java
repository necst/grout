package com.necst.controller.runtime.array;

import com.necst.controller.ControllerException;
import com.necst.controller.Type;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;
import com.necst.controller.runtime.computation.arraycomputation.ControllerArrayReadExecution;
import com.necst.controller.runtime.computation.arraycomputation.ControllerArrayWriteExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public class ControllerArray extends AbstractArray implements TruffleObject {

    /** Total number of elements stored in the array. */
    private final long numElements;

    /**
     * Total number of bytes allocated and used to store the array data (includes
     * padding).
     */
    public final long sizeBytes;

    public LittleEndianBigByteBufferView getLittleEndianBigByteBufferView() {
        return littleEndianBigByteBufferView;
    }

    /** Mutable view onto the underlying memory buffer. */
    LittleEndianBigByteBufferView littleEndianBigByteBufferView;

    public ControllerArray(AbstractControllerExecutionContext grOUTExecutionContext, long numElements,
            Type elementType) {
        super(grOUTExecutionContext, elementType);
        this.numElements = numElements;
        this.sizeBytes = numElements * elementType.getSizeBytes();
        // System.out.printf("\n############\nCreating a littleEndianBigByteByfferView
        // of size: %d\n", sizeBytes);
        this.littleEndianBigByteBufferView = new LittleEndianBigByteBufferView(sizeBytes);
        // Register the array in the AsyncGrOUTExecutionContext;
        this.registerArray();
    }

    @Override
    final public long getSizeBytes() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new ControllerException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return sizeBytes;
    }

    @Override
    public long getPointer() {
        return 0;
    }

    public Type getElementType() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new ControllerException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return elementType;
    }

    @Override
    public String toString() {
        if (arrayFreed) {
            return "ByteBufferArray(memory freed)";
        } else {
            return "ByteBufferArray(elementType=" + elementType + ", numElements=" + numElements;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        if (!arrayFreed) {
            freeMemory();
        }
        super.finalize();
    }

    @Override
    public void freeMemory() {
        if (arrayFreed) {
            throw new ControllerException("device array already freed");
        }
        this.littleEndianBigByteBufferView = null;
        arrayFreed = true;
    }

    @Override
    public void updateArray(LittleEndianBigByteBufferView array) {
        this.littleEndianBigByteBufferView = array;
    }

    // Implementation of InteropLibrary

    @ExportMessage
    public long getArraySize() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new ControllerException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return numElements;
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return !arrayFreed && index >= 0 && index < numElements;
    }

    @ExportMessage
    boolean isArrayElementModifiable(long index) {
        return index >= 0 && index < numElements;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

    @ExportMessage
    Object readArrayElement(long index,
            @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile)
            throws InvalidArrayIndexException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new ControllerException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((index < 0) || (index >= numElements)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        try {
            if (this.canSkipSchedulingRead()) {
                // Fast path, skip the DAG scheduling;
                return AbstractArray.readArrayElementNative(this.littleEndianBigByteBufferView, index, this.elementType,
                        elementTypeProfile);
            } else {
                return new ControllerArrayReadExecution(this, index, elementTypeProfile).schedule();
            }
        } catch (UnsupportedTypeException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public Object readNativeView(long index,
            @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) {
        return AbstractArray.readArrayElementNative(this.littleEndianBigByteBufferView, index, this.elementType,
                elementTypeProfile);
    }

    @ExportMessage
    public void writeArrayElement(long index, Object value,
            @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
            @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile)
            throws UnsupportedTypeException, InvalidArrayIndexException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new ControllerException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((index < 0) || (index >= numElements)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        if (this.canSkipSchedulingWrite()) {
            // Fast path, skip the DAG scheduling;
            AbstractArray.writeArrayElementNative(this.littleEndianBigByteBufferView, index, value, elementType,
                    valueLibrary, elementTypeProfile);
        } else {
            new ControllerArrayWriteExecution(this, index, value, valueLibrary, elementTypeProfile).schedule();
        }
    }

    @Override
    public void writeNativeView(long index, Object value, @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
            @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile)
            throws UnsupportedTypeException {
        AbstractArray.writeArrayElementNative(this.littleEndianBigByteBufferView, index, value, elementType,
                valueLibrary, elementTypeProfile);
    }
}
