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
package com.necst.controller.runtime;

import com.necst.controller.runtime.computation.ComputationArgument;
import com.necst.controller.runtime.computation.ComputationArgumentWithValue;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class KernelArguments implements Closeable{
    /**
     * Associate each input object to the characteristics of its argument, such as its type and if it's constant;
     */

    private final Object[] originalArgs; // input of the kernel, with direct ref to ControllerArray
    private final Object[] data; // "cleaned" input of the kernel ready for remote execution, aka no ControllerArray
    private final String[] types; // either pointer (ControllerArray) or constant (int, float, etc)

    // TODO: KernelArgumentWithValues can probably be deprecated
    private final List<ComputationArgumentWithValue> kernelArgumentWithValues = new ArrayList<>();


    public Object[] getData() {
        return data;
    }

    public String[] getTypes() {
        return types;
    }

    public KernelArguments(Object[] args, ComputationArgument[] kernelArgumentList) {
        this.originalArgs = args;
        assert(args.length == kernelArgumentList.length);
        // Initialize the list of arguments and object references;
        data = new Object[args.length];
        types = new String[args.length];

        //TODO: investigate if we can remove kernelArgumentsWithValues completely
        for (int i = 0; i < args.length; i++) {
            kernelArgumentWithValues.add(new ComputationArgumentWithValue(kernelArgumentList[i], args[i]));
        }
    }

    public void setArgument(int paramIdx, String type, Object obj) {
        data[paramIdx] = obj;
        types[paramIdx] = type;
    }


    public Object[] getOriginalArgs() {
        return originalArgs;
    }

    public Object getOriginalArg(int index) {
        return originalArgs[index];
    }

    public List<ComputationArgumentWithValue> getKernelArgumentWithValues() {
        return kernelArgumentWithValues;
    }

    @Override
    public String toString() {
        return "KernelArgs=" + Arrays.toString(originalArgs);
    }

    @Override
    public void close() {

    }
}
