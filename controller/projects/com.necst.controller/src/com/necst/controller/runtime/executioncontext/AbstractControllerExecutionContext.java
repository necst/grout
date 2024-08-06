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
package com.necst.controller.runtime.executioncontext;

import com.necst.controller.ControllerException;
import com.necst.controller.ControllerLogger;
import com.necst.controller.ControllerOptionMap;
import com.necst.controller.runtime.ControllerStatistics;
import com.necst.controller.runtime.Kernel;
import com.necst.controller.runtime.array.AbstractArray;
import com.necst.controller.runtime.computation.GrOUTComputationalElement;
import com.necst.controller.runtime.computation.dependency.DefaultDependencyComputationBuilder;
import com.necst.controller.runtime.computation.dependency.DependencyComputationBuilder;
import com.necst.controller.runtime.computation.dependency.WithConstDependencyComputationBuilder;
import com.necst.controller.runtime.worker.WorkersManager;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.HashSet;
import java.util.Set;

/**
 * Abstract class that defines how {@link GrOUTComputationalElement} are
 * registered and scheduled for execution.
 * It monitors the state of GrOUT execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public abstract class AbstractControllerExecutionContext {

    public ControllerStatistics controllerStatistics = new ControllerStatistics();

    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.EXECUTIONCONTEXT_LOGGER);

    /**
     * Reference to the {@link WorkersManager} that takes care of
     * scheduling computations on different workers;
     */
    public WorkersManager workersManager;

    /**
     * Set that contains all the arrays allocated so far.
     */
    protected final Set<AbstractArray> arraySet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels declared so far.
     */
    protected final Set<Kernel> kernelSet = new HashSet<>();

    /**
     * Reference to the computational DAG that represents dependencies between
     * computations;
     */
    protected final ExecutionDAG dag;

    /**
     * Reference to how dependencies between computational elements are computed
     * within this execution context;
     */
    private final DependencyComputationBuilder dependencyBuilder;

    /**
     * True if we consider that an argument can be "const" in the scheduling;
     */
    private final boolean isConstAware;

    public ControllerOptionMap options;

    public final boolean enableTimers;

    public AbstractControllerExecutionContext(ControllerOptionMap options) {
        this.options = options;
        // Compute the dependency policy to use;
        switch (options.getDependencyPolicy()) {
            case WITH_CONST:
                this.isConstAware = true;
                this.dependencyBuilder = new WithConstDependencyComputationBuilder();
                break;
            case NO_CONST:
                this.isConstAware = false;
                this.dependencyBuilder = new DefaultDependencyComputationBuilder();
                break;
            default:
                LOGGER.severe(
                        () -> "Cannot create a GrOUTExecutionContext. The selected dependency policy is not valid: "
                                + options.getDependencyPolicy());
                throw new ControllerException(
                        "selected dependency policy is not valid: " + options.getDependencyPolicy());
        }

        LOGGER.info("Record timings: " + options.getEnableTimers());
        this.enableTimers = options.getEnableTimers();

        LOGGER.info("Connecting to the specified clients");
        this.workersManager = new WorkersManager(options, this);

        LOGGER.info("Creating the ExecutionDAG");
        this.dag = new ExecutionDAG(options.getDependencyPolicy());

    }

    /**
     * Register this computation for future execution by the
     * {@link AbstractControllerExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data
     * dependencies;
     */
    abstract public Object registerExecution(GrOUTComputationalElement computation) throws UnsupportedTypeException;

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
    }

    public void registerKernel(Kernel kernel) {
        kernelSet.add(kernel);
    }

    public ExecutionDAG getDag() {
        return dag;
    }

    public DependencyComputationBuilder getDependencyBuilder() {
        return dependencyBuilder;
    }

    public boolean isConstAware() {
        return isConstAware;
    }

    abstract public Kernel buildKernel(String code, String kernelName, String signature);

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    public void cleanup() {
    }
}
