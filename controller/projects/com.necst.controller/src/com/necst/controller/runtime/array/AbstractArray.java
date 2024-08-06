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
package com.necst.controller.runtime.array;

import com.necst.controller.ControllerException;
import com.necst.controller.MemberSet;
import com.necst.controller.NoneValue;
import com.necst.controller.Type;
import com.necst.controller.runtime.Controller;
import com.necst.controller.runtime.computation.GrOUTComputationalElement;
import com.necst.controller.runtime.computation.arraycomputation.ArrayAccessExecution;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;
import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;
import com.oracle.truffle.api.TruffleLogger;
import com.necst.controller.ControllerLogger;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.Arrays;

/**
 * Simple wrapper around each class that represents device arrays in GrOUT.
 * It can be used to keep track of generic arrays during execution, and monitor
 * dependencies.
 */
@ExportLibrary(InteropLibrary.class)
public abstract class AbstractArray implements TruffleObject {
    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.ARRAY_LOGGER);

    static final AtomicLong NEXT_ID = new AtomicLong(0);

    public long getId() {
        return id;
    }

    final long id = NEXT_ID.getAndIncrement();

    protected static final String POINTER = "pointer";
    protected static final String COPY_FROM = "copyFrom";
    protected static final String COPY_TO = "copyTo";
    protected static final String FREE = "free";
    protected static final String IS_MEMORY_FREED = "isMemoryFreed";
    protected static final String ACCESSED_FREED_MEMORY_MESSAGE = "memory of array freed";
    protected static final MemberSet PUBLIC_MEMBERS = new MemberSet(COPY_FROM, COPY_TO, FREE, IS_MEMORY_FREED);
    protected static final MemberSet MEMBERS = new MemberSet(POINTER, COPY_FROM, COPY_TO, FREE, IS_MEMORY_FREED);

    /**
     * Reference to the underlying CUDA runtime that manages the array memory.
     */
    protected final AbstractControllerExecutionContext grOUTExecutionContext;

    /**
     * Data type of elements stored in the array.
     */
    protected final Type elementType;

    /**
     * True IFF the array has been registered in
     * {@link AbstractControllerExecutionContext}.
     * Used to avoid multiple registration;
     */
    private boolean registeredInContext = false;

    /**
     * Function used to compute if we can skip the scheduling of a computational
     * element for a given array read;
     */
    private final SkipSchedulingInterface skipScheduleRead;

    /**
     * Function used to compute if we can skip the scheduling of a computational
     * element for a given array write;
     */
    private final SkipSchedulingInterface skipScheduleWrite;

    /**
     * List of devices where the array is currently UP-TO-DATE, i.e. it can be
     * accessed without requiring any memory transfer.
     * On pre-Pascal GPUs, arrays are allocated on the currently active GPU. On
     * devices since Pascal, arrays are allocated on the CPU.
     * We identify devices using integers. CPU is -1
     * ({@link Controller#CONTROLLER_NODE}, GPUs start from 0;
     */
    public final Set<String> arrayUpToDateLocations = new HashSet<>();

    /** Flag set when underlying off-heap memory has been freed. */
    protected boolean arrayFreed = false;

    public Type getElementType() {
        return elementType;
    }

    protected AbstractArray(AbstractControllerExecutionContext grOUTExecutionContext, Type elementType) {
        this.grOUTExecutionContext = grOUTExecutionContext;
        this.elementType = elementType;

        this.skipScheduleRead = this::isArrayUpdatedOnController;
        this.skipScheduleWrite = this::isArrayUpdatedOnlyOnController;

        // While creating a new ARRAY all the devices have it up-to-date since none has
        // written to it!
        // this is important, since we avoid sending empty arrays to the workers
        this.addArrayUpToDateLocations(Controller.CONTROLLER_NODE);
        for (String worker : grOUTExecutionContext.workersManager.getWorkersIdentifiers()) {
            this.addArrayUpToDateLocations(worker);
        }
    }

    protected AbstractArray(AbstractArray otherArray) {
        this.grOUTExecutionContext = otherArray.grOUTExecutionContext;
        this.elementType = otherArray.elementType;
        this.skipScheduleRead = otherArray.skipScheduleRead;
        this.skipScheduleWrite = otherArray.skipScheduleWrite;
        this.arrayFreed = otherArray.arrayFreed;

        // Initialize the location of an abstract array, copying the ones specified in
        // the input;
        this.arrayUpToDateLocations.addAll(otherArray.getArrayUpToDateLocations());

        // Registration must be done afterwards;
        this.registeredInContext = false;
    }

    public boolean isMemoryFreed() {
        return arrayFreed;
    }

    /**
     * Register the array in {@link AbstractControllerExecutionContext} so that
     * operations on this array
     * can be monitored by the runtime. Registration must be done with a separate
     * function at the end of concrete Array classes.
     * This is done to avoid leaving the context in an inconsistent state if the
     * concrete constructor throws an exception and fails.
     */
    protected void registerArray() {
        if (!this.registeredInContext) {
            this.grOUTExecutionContext.registerArray(this);
            this.registeredInContext = true;
        }
    }

    public AbstractControllerExecutionContext getGrOUTExecutionContext() {
        return grOUTExecutionContext;
    }

    /**
     * Tracks whether the array is up-to-date on CPU.
     * This happens if the last operation done on the native memory underlying this
     * array is a read/write operation
     * handled by the CPU. If so, we can avoid creating
     * {@link GrOUTComputationalElement}
     * for array reads that are immediately following the last one, as they are
     * performed synchronously and there is no
     * reason to explicitly model them in the {@link ExecutionDAG};
     */
    // FIXME (check if fixed already): Possible error: Array A is up-to-date on CPU
    // and GPU0. There's an ongoing kernel on GPU0 that uses A read-only.
    // If we write A on the CPU, is the scheduling skipped? That's an error.
    // In the case of a read, no problem (a kernel that modifies the data would take
    // exclusive ownership),
    // while in the case of a write we need to check that arrayUpToDateLocations ==
    // CPU
    public boolean isArrayUpdatedOnController() {
        return this.arrayUpToDateLocations.contains(Controller.CONTROLLER_NODE);
    }

    /**
     * Tracks whether the array is up-to-date only on CPU, and not on other devices.
     * This happens if the last operation done on the native memory underlying this
     * array is a read/write operation
     * handled by the CPU. If so, we can avoid creating
     * {@link GrOUTComputationalElement}
     * for array accesses that are immediately following the last one, as they are
     * performed synchronously and there is no
     * reason to explicitly model them in the {@link ExecutionDAG}.
     * To perform a write on the CPU, we need the array to be updated exclusively on
     * the CPU;
     */
    public boolean isArrayUpdatedOnlyOnController() {
        return this.arrayUpToDateLocations.size() == 1
                && this.arrayUpToDateLocations.contains(Controller.CONTROLLER_NODE);
    }

    public Set<String> getArrayUpToDateLocations() {
        return this.arrayUpToDateLocations;
    }

    /**
     * Reset the list of devices where the array is currently up-to-date,
     * and specify a new device where the array is up-to-date.
     * Used when the array is modified by some device: there should never be a
     * situation where
     * the array is not up-to-date on at least one device;
     */
    public void resetArrayUpToDateLocations(String workerId) {
        LOGGER.finer("BEFORE RESET: " + Arrays.toString(arrayUpToDateLocations.toArray()));
        this.arrayUpToDateLocations.clear();
        this.arrayUpToDateLocations.add(workerId);
        LOGGER.finer("AFTER RESET: " + Arrays.toString(arrayUpToDateLocations.toArray()));
    }

    public void addArrayUpToDateLocations(String nodeIdentifier) {
        this.arrayUpToDateLocations.add(nodeIdentifier);
        LOGGER.finer("AFTER ADD: " + Arrays.toString(arrayUpToDateLocations.toArray()));
    }

    /**
     * True if this array is up-to-date for the input device;
     * 
     * @param nodeIdentifier a worker for which we want to check if this array is
     *                       up-to-date;
     * @return if this array is up-to-date with respect to the input device
     */
    public boolean isArrayUpdatedIn(String nodeIdentifier) {
        return this.arrayUpToDateLocations.contains(nodeIdentifier);
    }

    public abstract long getPointer();

    public abstract long getSizeBytes();

    public abstract void freeMemory();

    public abstract void updateArray(LittleEndianBigByteBufferView array);

    /**
     * Access the underlying native memory of the array, as if it were a linear 1D
     * array.
     * It can be used to copy chunks of the array without having to perform repeated
     * checks,
     * and for the low-level implementation of array accesses
     * 
     * @param index              index used to access the array
     * @param elementTypeProfile profiling of the element type, to speed up the
     *                           native view access
     * @return element of the array
     */
    public abstract Object readNativeView(long index,
            @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile);

    /**
     * Static method to read the native view of an array. It can be used to
     * implement the innermost access in {@link AbstractArray#readNativeView};
     * 
     * @param nativeView         native array representation of the array
     * @param index              index used to access the array
     * @param elementType        type of the array, required to know the size of
     *                           each element
     * @param elementTypeProfile profiling of the element type, to speed up the
     *                           native view access
     * @return element of the array
     */
    protected static Object readArrayElementNative(LittleEndianBigByteBufferView nativeView, long index,
            Type elementType,
            @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) {
        switch (elementTypeProfile.profile(elementType)) {
            case CHAR:
                return nativeView.getByte(index * Type.CHAR.getSizeBytes());
            case SINT16:
                return nativeView.getShort(index * Type.SINT16.getSizeBytes());
            case SINT32:
                return nativeView.getInt(index * Type.SINT32.getSizeBytes());
            case SINT64:
                return nativeView.getLong(index * Type.SINT64.getSizeBytes());
            case FLOAT:
                return nativeView.getFloat(index * Type.FLOAT.getSizeBytes());
            case DOUBLE:
                return nativeView.getDouble(index * Type.DOUBLE.getSizeBytes());
        }
        return null;
    }

    /**
     * Access the underlying native memory of the array, as if it were a linear 1D
     * array.
     * It can be used to copy chunks of the array without having to perform repeated
     * checks,
     * and for the low-level implementation of array accesses
     * 
     * @param index              index used to access the array
     * @param value              value to write in the array
     * @param valueLibrary       interop access of the value, required to understand
     *                           its type
     * @param elementTypeProfile profiling of the element type, to speed up the
     *                           native view access
     * @throws UnsupportedTypeException if writing the wrong type in the array
     */
    public abstract void writeNativeView(long index, Object value,
            @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
            @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile)
            throws UnsupportedTypeException;

    /**
     * Static method to write the native view of an array. It can be used to
     * implement the innermost access in {@link AbstractArray#writeNativeView};
     * 
     * @param nativeView         native array representation of the array
     * @param index              index used to access the array
     * @param value              value to write in the array
     * @param elementType        type of the array, required to know the size of
     *                           each element
     * @param valueLibrary       interop access of the value, required to understand
     *                           its type
     * @param elementTypeProfile profiling of the element type, to speed up the
     *                           native view access
     * @throws UnsupportedTypeException if writing the wrong type in the array
     */
    public static void writeArrayElementNative(LittleEndianBigByteBufferView nativeView, long index, Object value,
            Type elementType,
            @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
            @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile)
            throws UnsupportedTypeException {
        try {
            switch (elementTypeProfile.profile(elementType)) {
                case CHAR:
                    nativeView.setByte(index * Type.CHAR.getSizeBytes(), valueLibrary.asByte(value));
                    break;
                case SINT16:
                    nativeView.setShort(index * Type.SINT16.getSizeBytes(), valueLibrary.asShort(value));
                    break;
                case SINT32:
                    nativeView.setInt(index * Type.SINT32.getSizeBytes(), valueLibrary.asInt(value));
                    break;
                case SINT64:
                    nativeView.setLong(index * Type.SINT64.getSizeBytes(), valueLibrary.asLong(value));
                    break;
                case FLOAT:
                    // going via "double" to allow floats to be initialized with doubles
                    nativeView.setFloat(index * Type.FLOAT.getSizeBytes(), (float) valueLibrary.asDouble(value));
                    break;
                case DOUBLE:
                    nativeView.setDouble(index * Type.DOUBLE.getSizeBytes(), valueLibrary.asDouble(value));
                    break;
            }
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[] { value }, "value cannot be coerced to " + elementType);
        }
    }

    /**
     * Check if this array can be accessed by the host for a read without having to
     * schedule a {@link ArrayAccessExecution}.
     * This is possible if the array is up-to-date on the CPU,
     * and the array is not exposed on the default stream while other GPU
     * computations are running (on pre-Pascal devices).
     * 
     * @return if this array can be accessed by the host without scheduling a
     *         computation
     */
    public boolean canSkipSchedulingRead() {
        return this.skipScheduleRead.canSkipScheduling();
    }

    /**
     * Check if this array can be accessed by the host for a write without having to
     * schedule a {@link ArrayAccessExecution}.
     * This is possible if the array is assumed up-to-date only on the CPU,
     * and the array is not exposed on the default stream while other GPU
     * computations are running (on pre-Pascal devices).
     * 
     * @return if this array can be accessed by the host without scheduling a
     *         computation
     */
    public boolean canSkipSchedulingWrite() {
        return this.skipScheduleWrite.canSkipScheduling();
    }

    protected interface SkipSchedulingInterface {
        boolean canSkipScheduling();
    }

    /**
     * By default, we assume that arrays are stored in row-major format ("C"
     * format).
     * This holds true for {@link ControllerArray}s, which are 1D arrays where the
     * storage order does not matter;
     * 
     * @return if the array was stored in column-major format (i.e. "Fortran" or
     *         "F")
     */
    public boolean isColumnMajorFormat() {
        return false;
    }

    // Implementation of InteropLibrary

    @ExportMessage
    boolean isPointer() {
        return true;
    }

    @ExportMessage
    long asPointer() {
        return this.getPointer();
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new ControllerException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return true;
    }

    @ExportMessage
    Object readArrayElement(long index) throws UnsupportedMessageException, InvalidArrayIndexException {
        return null;
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return false;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(boolean includeInternal) {
        return includeInternal ? MEMBERS : PUBLIC_MEMBERS;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
            @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return POINTER.equals(name) || COPY_FROM.equals(name) || COPY_TO.equals(name) || FREE.equals(name)
                || IS_MEMORY_FREED.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
            @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile)
            throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (POINTER.equals(memberName)) {
            return getPointer();
        }
        if (COPY_FROM.equals(memberName)) {
            throw new ControllerException("DeviceArrayCopyFunction(COPY_FROM) currently not supported from readMember");
            // return new DeviceArrayCopyFunction(this,
            // DeviceArrayCopyFunction.CopyDirection.FROM_POINTER);
        }
        if (COPY_TO.equals(memberName)) {
            throw new ControllerException("DeviceArrayCopyFunction(COPY_TO) currently not supported from readMember");
            // return new DeviceArrayCopyFunction(this,
            // DeviceArrayCopyFunction.CopyDirection.TO_POINTER);
        }
        if (FREE.equals(memberName)) {
            return new DeviceArrayFreeFunction();
        }
        if (IS_MEMORY_FREED.equals(memberName)) {
            return isMemoryFreed();
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return COPY_FROM.equals(memberName) || COPY_TO.equals(memberName) || FREE.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
            Object[] arguments,
            @CachedLibrary("this") InteropLibrary interopRead,
            @CachedLibrary(limit = "1") InteropLibrary interopExecute)
            throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }

    /**
     * Retrieve the total number of elements in the array,
     * or the size of the current dimension for matrices and tensors
     *
     * @return the total number of elements in the array
     */
    @ExportMessage
    public abstract long getArraySize();

    // TODO: equals must be smarter than checking memory address, as a MultiDimView
    // should be considered as part of its parent,
    // similarly to what "isLastComputationArrayAccess" is doing.
    // The hash instead should be different. We might also not touch equals, and
    // have another method "isPartOf"

    @ExportLibrary(InteropLibrary.class)
    final class DeviceArrayFreeFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            if (arguments.length != 0) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(0, 0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }
}
