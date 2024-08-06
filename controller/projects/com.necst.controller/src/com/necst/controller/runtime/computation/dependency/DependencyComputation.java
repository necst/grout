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
package com.necst.controller.runtime.computation.dependency;

import com.necst.controller.runtime.computation.ComputationArgumentWithValue;
import com.necst.controller.runtime.computation.GrOUTComputationalElement;

import java.util.Collection;

/**
 * Defines how data dependencies between {@link GrOUTComputationalElement} are
 * found,
 * e.g. if read-only or scalar argyments should be ignored.
 * It returns the list of arguments that have been found to create side-effects.
 * The function is not guaranteed to be pure,
 * and is allowed update information in the {@link GrOUTComputationalElement}
 */
public abstract class DependencyComputation {

    /**
     * This set contains the input arguments that are considered, at each step, in
     * the dependency computation.
     * The set initially coincides with "argumentSet", then arguments are removed
     * from this set once a new dependency is found.
     * This is conceptually a set, in the sense that every element is unique.
     * Concrete implementations might use other data structures, if required;
     */
    protected Collection<ComputationArgumentWithValue> activeArgumentSet;

    /**
     * Computes if the "other" GrOUTComputationalElement has dependencies w.r.t.
     * this kernel,
     * such as requiring as input a value computed by this kernel;
     * 
     * @param other kernel for which we want to check dependencies, w.r.t. this
     *              kernel
     * @return the list of arguments that the two kernels have in common
     */
    public abstract Collection<ComputationArgumentWithValue> computeDependencies(GrOUTComputationalElement other);

    public Collection<ComputationArgumentWithValue> getActiveArgumentSet() {
        return activeArgumentSet;
    }

}
