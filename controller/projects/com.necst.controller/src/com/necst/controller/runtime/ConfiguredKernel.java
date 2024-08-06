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
package com.necst.controller.runtime;

import com.necst.controller.ControllerException;
import com.necst.controller.Type;
import com.necst.controller.runtime.array.ControllerArray;
import com.necst.controller.runtime.computation.ComputationArgument;
import com.necst.controller.runtime.computation.KernelExecution;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;


@ExportLibrary(InteropLibrary.class)
public class ConfiguredKernel implements TruffleObject {

    private final Kernel kernel;

    private final KernelConfig config;

    public ConfiguredKernel(Kernel kernel, KernelConfig config) {
        this.kernel = kernel;
        this.config = config;
    }

    @ExportMessage
    boolean isExecutable() {
        return true;
    }

    /*
    KernelArguments createKernelArguments(Object[] args) throws ArityException {
        if (args.length != kernel.getKernelParameters().length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(kernel.getKernelParameters().length, kernel.getKernelParameters().length, args.length);
        }
        return new KernelArguments(args, this.kernel.getKernelParameters());
    }
    */

    /**
     * Parse the input arguments of the kernel and map them to the signature, making sure that the signature is respected
     * @param arguments list of input arguments to the kernel
     * @param booleanAccess used to parse boolean inputs
     * @param int8Access used to parse char inputs
     * @param int16Access used to parse short integer inputs
     * @param int32Access used to parse integer inputs
     * @param int64Access used to parse long integer inputs
     * @param doubleAccess used to parse double and float inputs
     * @return the object that wraps the kernel signature and arguments
     * @throws UnsupportedTypeException if one of the inputs does not respect the signature
     * @throws ArityException if the number of inputs does not respect the signature
     */
    KernelArguments createKernelArguments(Object[] arguments, InteropLibrary booleanAccess,
                                          InteropLibrary int8Access, InteropLibrary int16Access,
                                          InteropLibrary int32Access, InteropLibrary int64Access, InteropLibrary doubleAccess)
            throws UnsupportedTypeException, ArityException {
        if (arguments.length != kernel.getKernelParameters().length) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(kernel.getKernelParameters().length, kernel.getKernelParameters().length, arguments.length);
        }
        KernelArguments kernelArgs = new KernelArguments(arguments, this.kernel.getKernelParameters());
        for (int paramIdx = 0; paramIdx < kernel.getKernelParameters().length; paramIdx++) {
            Object arg = arguments[paramIdx];
            ComputationArgument param = kernel.getKernelParameters()[paramIdx];
            Type paramType = param.getType();
            try {
                // System.out.println(paramIdx+") "+arg.getClass().getName());
                if (param.isPointer()) { //Array
                    if (arg instanceof ControllerArray) {
                        ControllerArray controllerArray = (ControllerArray) arg;
                        if (!param.isSynonymousWithPointerTo(controllerArray.getElementType())) {
                            throw new ControllerException("controller array of " + controllerArray.getElementType() + " cannot be used as pointer argument " + paramType);
                        }
                        // we set the type parameter of the kernelArgs at index paramIdx to a string containing the total length of the controllerArray
                        kernelArgs.setArgument(paramIdx, Long.toString(controllerArray.sizeBytes), controllerArray.getId());
                    } else {
                        // System.out.println("### ARG IS POINTER BUT NOT OF TYPE CONTROLLER ARRAY ###");
                        CompilerDirectives.transferToInterpreter();
                        throw UnsupportedTypeException.create(new Object[]{arg}, "expected DeviceArray type");
                    }
                } else { // by-value argument
                    // in this case directly create primitive types for the constants passed as arguments
                    switch (paramType) {
                        case BOOLEAN: {
                            kernelArgs.setArgument(paramIdx, "constant" , booleanAccess.asBoolean(arg) ? ((byte) 1) : ((byte) 0));
                            break;
                        }
                        case SINT8:
                        case CHAR: {
                            kernelArgs.setArgument(paramIdx, "constant" , int8Access.asByte(arg));
                            break;
                        }
                        case SINT16: {
                            kernelArgs.setArgument(paramIdx, "constant" , int16Access.asShort(arg));
                            break;
                        }
                        case SINT32:
                        case WCHAR: {
                            kernelArgs.setArgument(paramIdx, "constant" , int32Access.asInt(arg));
                            break;
                        }
                        case SINT64:
                        case SLL64:
                            // no larger primitive type than long -> interpret long as unsigned
                        case UINT64:
                        case ULL64: {
                            kernelArgs.setArgument(paramIdx, "constant" , int64Access.asLong(arg));
                            break;
                        }
                        case UINT8:
                        case CHAR8: {
                            int uint8 = int16Access.asShort(arg);
                            if (uint8 < 0 || uint8 > 0xff) {
                                CompilerDirectives.transferToInterpreter();
                                throw createExceptionValueOutOfRange(paramType, uint8);
                            }
                            kernelArgs.setArgument(paramIdx, "constant" , 0xff & uint8);
                            break;
                        }
                        case UINT16:
                        case CHAR16: {
                            int uint16 = int32Access.asInt(arg);
                            if (uint16 < 0 || uint16 > 0xffff) {
                                CompilerDirectives.transferToInterpreter();
                                throw createExceptionValueOutOfRange(paramType, uint16);
                            }
                            kernelArgs.setArgument(paramIdx, "constant" , (short) (0xffff & uint16));
                            break;
                        }
                        case UINT32: {
                            long uint32 = int64Access.asLong(arg);
                            if (uint32 < 0 || uint32 > 0xffffffffL) {
                                CompilerDirectives.transferToInterpreter();
                                throw createExceptionValueOutOfRange(paramType, uint32);
                            }
                            kernelArgs.setArgument(paramIdx, "constant" , 0xffffffffL & uint32);
                            break;
                        }
                        case FLOAT: {
                            kernelArgs.setArgument(paramIdx, "constant" , (float) doubleAccess.asDouble(arg));
                            break;
                        }
                        case DOUBLE: {
                            kernelArgs.setArgument(paramIdx, "constant" , doubleAccess.asDouble(arg));
                            break;
                        }
                        default:
                            CompilerDirectives.transferToInterpreter();
                            throw UnsupportedTypeException.create(new Object[]{arg},
                                    "unsupported by-value parameter type: " + paramType);
                    }
                }
            } catch (UnsupportedMessageException e) {
                CompilerDirectives.transferToInterpreter();
                throw UnsupportedTypeException.create(new Object[]{arg},
                        "expected type " + paramType + " in argument " + arg);
            }
        }
        return kernelArgs;
    }

    private static ControllerException createExceptionValueOutOfRange(Type type, long value) {
        return new ControllerException("value " + value + " is out of range for type " + type);
    }

    @ExportMessage
    @TruffleBoundary
    Object execute(Object[] arguments,
                   @CachedLibrary(limit = "3") InteropLibrary boolAccess,
                   @CachedLibrary(limit = "3") InteropLibrary int8Access,
                   @CachedLibrary(limit = "3") InteropLibrary int16Access,
                   @CachedLibrary(limit = "3") InteropLibrary int32Access,
                   @CachedLibrary(limit = "3") InteropLibrary int64Access,
                   @CachedLibrary(limit = "3") InteropLibrary doubleAccess) throws UnsupportedTypeException, ArityException {
        kernel.incrementLaunchCount();
        try (KernelArguments args = this.createKernelArguments(arguments, boolAccess, int8Access, int16Access, int32Access, int64Access, doubleAccess)) {
            new KernelExecution(this, args).schedule();
        }
        return this;
    }

    public Kernel getKernel() {
        return kernel;
    }

    public KernelConfig getConfig() {
        return config;
    }

    @Override
    public String toString() {
        return "ConfiguredKernel(" + kernel.toString() + "; " + config.toString() + ")";
    }
}
