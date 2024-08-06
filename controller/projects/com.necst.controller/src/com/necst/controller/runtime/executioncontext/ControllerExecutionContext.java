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

import com.necst.controller.ControllerOptionMap;
import com.necst.controller.runtime.ControllerStatistics;
import com.necst.controller.runtime.Kernel;
import com.necst.controller.runtime.array.AbstractArray;
import com.necst.controller.runtime.computation.GrOUTComputationalElement;
import com.necst.controller.runtime.worker.WorkersManager;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.Format;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

/**
 * Class used to monitor the state of GrOUT execution, keep track of memory
 * allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public class ControllerExecutionContext extends AbstractControllerExecutionContext {

    /**
     * Set that contains all the arrays allocated so far.
     */
    protected final Set<AbstractArray> arraySet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels declared so far.
     */
    protected final Set<Kernel> kernelSet = new HashSet<>();

    public ControllerExecutionContext(ControllerOptionMap controllerOptions) {
        super(controllerOptions);
    }

    /**
     * Register this computation for future execution by the
     * {@link ControllerExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data
     * dependencies;
     */
    public Object registerExecution(GrOUTComputationalElement computation) throws UnsupportedTypeException {
        // Add the new computation to the DAG
        LOGGER.fine("DAG appending -- " + computation);
        ExecutionDAG.DAGVertex vertex = dag.append(computation);

        // Assign the worker to this specific computation
        LOGGER.fine("Assigning worker -- " + computation);
        if (computation.canBeDistributed()) {
            if (this.enableTimers) {
                long start = System.nanoTime();
                workersManager.assignWorker(vertex);
                long time = System.nanoTime() - start;
                this.controllerStatistics.overheadRetrieveClient.add(time);
            } else {
                workersManager.assignWorker(vertex);
            }
        }

        // Start the computation;
        if (computation.getClient() == null)
            LOGGER.info("[CONTROLLER] Executing -- " + computation);
        else
            LOGGER.info("[" + computation.getClient() + "] Executing -- " + computation);
        return executeComputation(vertex);
    }

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    public void cleanup() {
        LOGGER.info("CLEANUP");
        String results_path = "";

        if (options.getEnableTimers()) {
            try {
                Format formatter = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss");
                Date currentDate = new Date();
                results_path = "./results/" + formatter.format(currentDate);
                Files.createDirectories(Paths.get(results_path));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            this.controllerStatistics.toJson(results_path, "/controller.json");
        }

        workersManager.cleanup(results_path);

    }

    private Object executeComputation(ExecutionDAG.DAGVertex vertex) throws UnsupportedTypeException {

        // Obtain the ControllerArrays necessary to start this computation
        // This will implicitly sync all the workers parents of the current computation
        // with respect to the Controller
        // After this call we know that all the parents have finished executing since
        // they returned the computed arrays
        LOGGER.fine("Obtaining the arrays necessary for the computation");
        /*
         * The basic idea of this function is to retrieve inside the Controller the
         * necessary arrays to start the
         * current vertex. In the case of reads/writes this is the only step needed, in
         * the case of KernelExecutions
         * we will need to move those arrays also the workers, but this will be done in
         * the updateLocationsOfArrays()
         * function call later on.
         * 
         * We have three types of vertex:
         * 1) ArrayRead: we will retrieve (if necessary) the arrays to the controller
         * 2) ArrayWrite: we will retrieve (if necessary) the arrays to the controller
         * 3) KernelExecution:
         * a) if all input parameters are up-to-date in the scheduled worker
         * --> do nothing
         * b) for all the input parameters that are not up-to-date in the scheduled
         * worker
         * --> move those inside the controller (later on updateLocationOfArrays will
         * send those to the worker)
         */
        long start = -1;
        if (this.enableTimers) {
            start = System.nanoTime();
        }

        workersManager.obtainNecessaryControllerArrays(vertex);

        // Perform the computation;
        LOGGER.fine("Set computation started");
        vertex.getComputation().setComputationStarted();

        LOGGER.fine("Updating location of the arrays");
        /*
         * This function is implemented in each type of computationalElement.
         * ArrayRead: it will add the controller to the up-to-date locations of the
         * DeviceArray
         * ArrayWrite: it will reset the up-to-date locations of the DeviceArray,
         * only the controller has the updated version
         * KernelExecution: it will send all the necessary arrays for the selected
         * worker.
         * Those that needs to be sent have already been sent to the controller in the
         * previous
         * step obtainNecessaryControllerArrays(). The others are already up-to-date on
         * the worker.
         */
        vertex.getComputation().updateLocationOfArrays();

        long time = -1;
        if (this.enableTimers) {
            time = System.nanoTime() - start;
            if (vertex.getComputation().canBeDistributed()) {
                // controller read/write on a deviceArray
                controllerStatistics.beforeExecutingKernels.add(time);
            } else {
                // remote kernel execution
                controllerStatistics.beforeExecutingArrayOperations.add(time);
            }
        }

        // Execute the computation
        LOGGER.fine("Executing the actual computational element");
        return vertex.getComputation().execute();
    }

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
    }

    public void registerKernel(Kernel kernel) {
        kernelSet.add(kernel);
    }

    public Kernel buildKernel(String code, String kernelName, String signature) {
        Kernel kernel = new Kernel(this, code, kernelName, signature);
        workersManager.buildKernelOnWorkers(code, kernelName, signature);
        return kernel;
    }
}
