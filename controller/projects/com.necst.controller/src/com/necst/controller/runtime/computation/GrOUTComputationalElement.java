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
package com.necst.controller.runtime.computation;

import com.necst.controller.ControllerLogger;
import com.necst.controller.runtime.array.AbstractArray;
import com.necst.controller.runtime.computation.dependency.DependencyComputation;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.runtime.worker.Worker;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Basic class that represents GrOUT computations,
 * and is used to model data dependencies between computations;
 */
public abstract class GrOUTComputationalElement {

    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.COMPUTATION_LOGGER);

    /**
     * This list contains the original set of input arguments that are used to
     * compute dependencies;
     */
    protected final List<ComputationArgumentWithValue> argumentsThatCanCreateDependencies;

    /**
     * Reference to the execution context where this computation is executed;
     */
    protected final AbstractControllerExecutionContext controllerExecutionContext;

    /**
     * Reference to the worker where this GrOUTComputationalElement will be
     * executed;
     */
    protected AbstractWorker worker;

    /**
     * Keep track of whether this computation has already been executed, and
     * represents a "dead" vertex in the DAG.
     * Computations that are already executed will not be considered when computing
     * dependencies;
     */
    private boolean computationFinished = false;

    /**
     * Keep track of whether this computation has already been started, to avoid
     * performing the same computation multiple times;
     */
    private boolean computationStarted = false;

    /**
     * Specify if this computational element represents a computation executed on
     * the CPU,
     * such as an array access (read or write) on an {@link AbstractArray}.
     * CPU computations are assumed synchronous. By default it returns false;
     */
    protected boolean isComputationDoneByCONTROLLER = false;

    private final DependencyComputation dependencyComputation;

    /**
     * Constructor that takes an argument set initializer to build the set of
     * arguments used in the dependency computation
     * 
     * @param controllerExecutionContext execution context in which this
     *                                   computational element will be scheduled
     * @param initializer                the initializer used to build the internal
     *                                   set of arguments considered in the
     *                                   dependency computation
     */
    public GrOUTComputationalElement(AbstractControllerExecutionContext controllerExecutionContext,
            InitializeDependencyList initializer) {
        this.argumentsThatCanCreateDependencies = initializer.initialize();
        // Initialize by making a copy of the original set;
        this.controllerExecutionContext = controllerExecutionContext;
        this.dependencyComputation = controllerExecutionContext.getDependencyBuilder()
                .initialize(this.argumentsThatCanCreateDependencies);
    }

    /**
     * Simplified constructor that takes a list of arguments, and consider all of
     * them in the dependency computation
     * 
     * @param controllerExecutionContext execution context in which this
     *                                   computational element will be scheduled
     * @param args                       the list of arguments provided to the
     *                                   computation. Arguments are expected to be
     *                                   {@link org.graalvm.polyglot.Value}
     */
    public GrOUTComputationalElement(AbstractControllerExecutionContext controllerExecutionContext,
            List<ComputationArgumentWithValue> args) {
        this(controllerExecutionContext, new DefaultExecutionInitializer(args));
    }

    public List<ComputationArgumentWithValue> getArgumentsThatCanCreateDependencies() {
        return argumentsThatCanCreateDependencies;
    }

    /**
     * Return if this computation could lead to dependencies with future
     * computations.
     * If not, this usually means that all of its arguments have already been
     * superseded by other computations,
     * or that the computation didn't have any arguments to begin with;
     * 
     * @return if the computation could lead to future dependencies
     */
    public boolean hasPossibleDependencies() {
        return !this.dependencyComputation.getActiveArgumentSet().isEmpty();
    }

    /**
     * Schedule this computation for future execution by the
     * {@link ControllerExecutionContext}.
     * The scheduling request is separate from the {@link GrOUTComputationalElement}
     * instantiation
     * as we need to ensure that the the computational element subclass has been
     * completely instantiated;
     */
    public Object schedule() throws UnsupportedTypeException {
        return this.controllerExecutionContext.registerExecution(this);
    }

    /**
     * Generic interface to perform the execution of this
     * {@link GrOUTComputationalElement}.
     * The actual execution implementation must be added by concrete computational
     * elements.
     * The execution request will be done by the {@link ControllerExecutionContext},
     * after this computation has been scheduled
     * using {@link GrOUTComputationalElement#schedule()}
     */
    public abstract Object execute() throws UnsupportedTypeException;

    public AbstractWorker getClient() {
        return this.worker;
    }

    public void setClient(AbstractWorker worker) {
        this.worker = worker;
    }

    public boolean isComputationFinished() {
        return computationFinished;
    }

    public boolean isComputationStarted() {
        return computationStarted;
    }

    public void setComputationFinished() {
        this.computationFinished = true;
    }

    public void setComputationStarted() {
        this.computationStarted = true;
    }

    public boolean canBeDistributed() {
        return false;
    }

    /**
     * Retrieve how the dependency computations are computed;
     */
    public DependencyComputation getDependencyComputation() {
        return dependencyComputation;
    }

    /**
     * This function needs to update the location of the ControllerArrays of this
     * computational element to be the one of the worker.
     * 1) If this is an array read:
     * --> nothing to do
     * 2) If this is an array write:
     * --> reset all the up-to-date locations and set the Controller as the only one
     * 3) If this is a kernel execution:
     * --> reset to the assigned worker if the ControllerArray is not marked as
     * constant
     * --> otherwise just add the worker to the list of up-to-date workers
     */
    abstract public void updateLocationOfArrays();

    /**
     * Obtain the list of input arguments for this computation that are arrays;
     * 
     * @return a list of arrays that are inputs for this computation
     */
    public List<AbstractArray> getArrayArguments() {
        // Note: "argumentsThatCanCreateDependencies" is a filter applied to the
        // original inputs,
        // so we have no guarantees that it contains all the input arrays.
        // In practice, "argumentsThatCanCreateDependencies" is already a selection of
        // the input arrays,
        // making the filter below unnecessary.
        // If for whatever reason we have a argumentsThatCanCreateDependencies that does
        // not contain all the input arrays,
        // we need to store the original input list in this class as well, and apply the
        // filter below to that list.
        return this.argumentsThatCanCreateDependencies.stream()
                .filter(ComputationArgument::isArray)
                .map(a -> (AbstractArray) a.getArgumentValue())
                .collect(Collectors.toList());
    }

    /**
     * Computes if the "other" GrOUTComputationalElement has dependencies w.r.t.
     * this kernel,
     * such as requiring as input a value computed by this kernel;
     * 
     * @param other kernel for which we want to check dependencies, w.r.t. this
     *              kernel
     * @return the list of arguments that the two kernels have in common
     */
    public Collection<ComputationArgumentWithValue> computeDependencies(GrOUTComputationalElement other) {
        return this.dependencyComputation.computeDependencies(other);
    }

    /**
     * The default initializer will simply store all the arguments,
     * and consider each of them in the dependency computations;
     */
    private static class DefaultExecutionInitializer implements InitializeDependencyList {
        private final List<ComputationArgumentWithValue> args;

        DefaultExecutionInitializer(List<ComputationArgumentWithValue> args) {
            this.args = args;
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            return args;
        }
    }
}
