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
 */
package com.necst.controller.functions;

import com.necst.controller.ControllerException;
import com.necst.controller.MemberSet;
import com.necst.controller.Type;
import com.necst.controller.TypeException;
import com.necst.controller.runtime.array.ControllerArray;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

import java.util.ArrayList;
import java.util.Optional;

@ExportLibrary(InteropLibrary.class)
public final class DeviceArrayFunction extends Function {

    private static final String MAP = "map";

    private static final MemberSet MEMBERS = new MemberSet(MAP);

    private final ControllerExecutionContext controllerExecutionContext;

    public DeviceArrayFunction(ControllerExecutionContext controllerExecutionContext) {
        super("DeviceArray");
        this.controllerExecutionContext = controllerExecutionContext;
    }

    @Override
    @TruffleBoundary
    public Object call(Object[] arguments) throws ArityException, UnsupportedTypeException {
        if (arguments.length < 1) {
            throw ArityException.create(1, 2, arguments.length);
        }
        String typeName = expectString(arguments[0], "first argument of DeviceArray must be string (type name)");
        Type elementType;
        try {
            elementType = Type.fromGrOUTTypeString(typeName);
        } catch (TypeException e) {
            throw new ControllerException(e.getMessage());
        }
        if (arguments.length == 1) {
            return new TypedDeviceArrayFunction(controllerExecutionContext, elementType);
        } else {
            return createArray(arguments, 1, elementType, controllerExecutionContext);
        }
    }

    static Object createArray(Object[] arguments, int start, Type elementType,
            AbstractControllerExecutionContext grOUTExecutionContext) throws UnsupportedTypeException {
        ArrayList<Long> elementsPerDim = new ArrayList<>();
        Optional<Boolean> useColumnMajor = Optional.empty();
        for (int i = start; i < arguments.length; ++i) {
            Object arg = arguments[i];
            if (INTEROP.isString(arg)) {
                if (useColumnMajor.isPresent()) {
                    throw new ControllerException("string option already provided");
                } else {
                    String strArg = expectString(arg, "string argument expected that specifies order ('C' or 'F')");
                    if (strArg.equals("f") || strArg.equals("F")) {
                        useColumnMajor = Optional.of(true);
                    } else if (strArg.equals("c") || strArg.equals("C")) {
                        useColumnMajor = Optional.of(false);
                    } else {
                        throw new ControllerException(
                                "invalid string argument '" + strArg + "', only \"C\" or \"F\" are allowed");
                    }
                }
            } else {
                long n = expectLong(arg, "expected number argument for dimension size");
                if (n < 1) {
                    throw new ControllerException("array dimension less than 1");
                }
                elementsPerDim.add(n);
            }
        }
        if (elementsPerDim.size() == 1) {
            return new ControllerArray(grOUTExecutionContext, elementsPerDim.get(0), elementType);
        } else {
            // TODO: implement MultiDimArray
            throw new ControllerException("Creating MultiDimDeviceArray is not currently supported");
        }
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return MEMBERS;
    }

    @ExportMessage(name = "isMemberReadable")
    @ExportMessage(name = "isMemberInvocable")
    @SuppressWarnings("static-method")
    boolean isMemberExisting(String memberName,
            @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return MAP.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
            @Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile)
            throws UnknownIdentifierException {
        if (MAP.equals(memberProfile.profile(memberName))) {
            return new MapDeviceArrayFunction(controllerExecutionContext);
        }
        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
            Object[] arguments,
            @CachedLibrary("this") InteropLibrary interopRead,
            @CachedLibrary(limit = "1") InteropLibrary interopExecute)
            throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }
}
