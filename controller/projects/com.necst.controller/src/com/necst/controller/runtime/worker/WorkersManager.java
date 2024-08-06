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
package com.necst.controller.runtime.worker;

import com.necst.controller.ControllerLogger;
import com.necst.controller.ControllerOptionMap;
import com.necst.controller.runtime.AbstractNode;
import com.necst.controller.runtime.Controller;
import com.necst.controller.runtime.Timings;
import com.necst.controller.runtime.WorkerStatistics;
import com.necst.controller.runtime.array.AbstractArray;
import com.necst.controller.runtime.array.ControllerArray;
import com.necst.controller.runtime.array.LittleEndianBigByteBufferView;
import com.necst.controller.runtime.computation.GrOUTComputationalElement;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.necst.controller.runtime.worker.policy.*;
import com.oracle.truffle.api.TruffleLogger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.Format;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import java.util.Iterator;

public class WorkersManager {

    public final HashMap<String, AbstractWorker> workersMap = new HashMap<>();

    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.WORKERSMANAGER_LOGGER);

    /**
     * Track the active computations each client has;
     */
    protected final Map<AbstractWorker, Set<ExecutionDAG.DAGVertex>> activeComputationsPerClient = new HashMap<>();

    /**
     * Handle for all the policies to assign a client to a new computation;
     */
    private final AbstractWorkerSelectionPolicy workerSelectionPolicy;

    AbstractControllerExecutionContext executionContext;

    ControllerOptionMap controllerOptionMap;

    public WorkersManager(ControllerOptionMap controllerOptionMap,
            AbstractControllerExecutionContext executionContext) {
        this.executionContext = executionContext;
        this.controllerOptionMap = controllerOptionMap;

        // LOGGER.info("WorkerNetInfo: " + controllerOptionMap.getWorkersInfo());
        generateWorkers(controllerOptionMap, controllerOptionMap.getWorkersInfo());

        switch (controllerOptionMap.getWorkerSelectionPolicy()) {
            case ROUND_ROBIN:
                this.workerSelectionPolicy = new RoundRobinPolicy(this);
                break;
            case MIN_TRANSFER_SIZE:
                this.workerSelectionPolicy = new MinTransferSizePolicy(this);
                break;
            case K_STEPS:
                this.workerSelectionPolicy = new KSteps(this);
                break;
            case VECTOR_STEP:
                this.workerSelectionPolicy = new VectorStep(this, controllerOptionMap.getVectorStep());
                break;
            case MIN_TRANSFER_TIME:
                this.workerSelectionPolicy = new MinTransferTimePolicy(this,
                        controllerOptionMap.getLinkBandwidthPath());
                break;
            default:
                this.workerSelectionPolicy = new RoundRobinPolicy(this);
        }

    }

    public void buildKernelOnWorkers(String code, String name, String signature) {
        for (AbstractWorker worker : workersMap.values()) {
            worker.buildKernel(code, name, signature);
        }
    }

    private void generateWorkers(ControllerOptionMap controllerOptionMap, List<String> workersNetInfo) {
        int id = 0;
        for (String workerNetInfo : workersNetInfo) {
            AbstractWorker worker;
            if (controllerOptionMap.getEnableMockWorker()) { // we want mock workers
                worker = new MockWorker(workerNetInfo, this.executionContext, id);
                id++;
                this.workersMap.put(workerNetInfo, worker);
            } else {
                worker = new Worker(workerNetInfo, this.executionContext, id);
                id++;
                this.workersMap.put(workerNetInfo, worker);
            }
        }
    }

    /**
     * Assign a {@link Worker} to the input computation, based on its dependencies.
     *
     * @param vertex an input computation for which we want to assign a stream
     */
    public void assignWorker(ExecutionDAG.DAGVertex vertex) {
        AbstractWorker worker = this.workerSelectionPolicy.retrieveClient(vertex);

        vertex.getComputation().setClient(worker);
        // Update the computation counter;
        addActiveComputation(vertex);
    }

    public void obtainNecessaryControllerArrays(ExecutionDAG.DAGVertex vertex) {
        // iterate over all the input ControllerArrays for the current vertex
        for (AbstractArray arr : vertex.getComputation().getArrayArguments()) {
            if (vertex.getComputation().canBeDistributed()) { // this vertex is a KernelExecution
                String scheduledWorker = vertex.getComputation().getClient().getNodeIdentifier();
                if (!arr.isArrayUpdatedIn(scheduledWorker)) { // in this case the current ControllerArrays is not
                                                              // up-to-date on the worker
                    if (arr.isArrayUpdatedOnlyOnController()) { // if the array is only up-to-date on the controller
                        LOGGER.finer("[TRANSFER] ControllerArray(" + arr.getId()
                                + ") up-to-date only on controller: sending controller --> scheduled worker");
                        ControllerArray controllerArray = (ControllerArray) arr;
                        this.getWorker(scheduledWorker).sendControllerArray(
                                controllerArray.getLittleEndianBigByteBufferView().getBigByteBuffer(),
                                controllerArray.getId(), controllerArray.getSizeBytes());
                    } // otherwise the array is already up-to-date on the controller
                    else { // p2p transfer between workers
                        LOGGER.finer(
                                "[TRANSFER] ControllerArray(" + arr.getId() + "): p2p transfer (worker --> worker)");
                        Set<String> potentialWorkers = arr.getArrayUpToDateLocations(); // this may contain the
                                                                                        // Controller.CONTROLLER_NODE
                        LOGGER.finer("[TRANSFER] ControllerArray(" + arr.getId() + "): p2p uptodate locations: "
                                + potentialWorkers.toString());
                        Iterator<String> workerIterator = potentialWorkers.iterator();
                        String selectedWorkerForP2P = workerIterator.next();
                        while (selectedWorkerForP2P.equals(Controller.CONTROLLER_NODE)) {
                            if (workerIterator.hasNext() == false) {
                                throw new RuntimeException(
                                        "Something went wrong iterating over the list of candidate workers");
                            }
                            LOGGER.finer("Current selected worker: " + selectedWorkerForP2P);
                            selectedWorkerForP2P = workerIterator.next();
                        }
                        this.executionContext.controllerStatistics.numP2Ptransfers++;
                        this.getWorker(scheduledWorker).p2pTransfer(arr.getId(), arr.getSizeBytes(),
                                selectedWorkerForP2P);
                        // --> transfer the data from selectedWorker to scheduledWorker via p2p
                        // directives
                    }
                }
            } else { // this vertex is a read/write on a ControllerArray
                if (!arr.isArrayUpdatedOnController()) {
                    // if the array is not up-to-date on the controller we retrieve it from one
                    // worker
                    LOGGER.fine("[TRANSFER] ControllerArray(" + arr.getId() + "): not up-to-date on the controller");
                    Set<String> potentialWorkers = arr.getArrayUpToDateLocations();
                    AbstractWorker selectedWorker = workersMap.get(potentialWorkers.iterator().next());
                    updateControllerArrayFromRemoteWorker(selectedWorker, arr);
                } else {
                    LOGGER.fine(
                            "[NO TRANSFER] ControllerArray(" + arr.getId() + "): already up-to-date on the controller");
                }
            }
        }
        // at this point in the controller we have all the ControllerArrays necessary to
        // start the current vertex
    }

    private void updateControllerArrayFromRemoteWorker(AbstractWorker worker, AbstractArray array) {
        LOGGER.fine(
                "[UPDATING] ControllerArray(" + array.getId() + ") from WORKER(" + worker.getNodeIdentifier() + ")");
        LittleEndianBigByteBufferView byteArray = worker.getRemoteArray(array.getId(), array.getSizeBytes());
        array.updateArray(byteArray);
    }

    /**
     * Obtain the set of Clients that needs to be synced before starting the
     * computations;
     *
     * @param computationsToSync a set of computations to sync
     * @return the set of Client that have to be synchronized
     */
    protected Set<AbstractWorker> getParentClients(Collection<GrOUTComputationalElement> computationsToSync) {
        return computationsToSync.stream().map(GrOUTComputationalElement::getClient).collect(Collectors.toSet());
    }

    public int getNumActiveComputationsOnClient(Worker worker) {
        return worker.getNumActiveComputation();
    }

    protected void addActiveComputation(ExecutionDAG.DAGVertex vertex) {
        AbstractWorker worker = vertex.getComputation().getClient();
        // Start tracking the stream if it wasn't already tracked;
        if (!activeComputationsPerClient.containsKey(worker)) {
            activeComputationsPerClient.put(worker, new HashSet<>());
        }
        // Associate the computation to the stream;
        activeComputationsPerClient.get(worker).add(vertex);
    }

    /**
     * Reset the association between workers and computations. All computations are
     * finished, and all workers are free;
     */
    protected void resetActiveComputationState() {
        activeComputationsPerClient.keySet().forEach(
                s -> activeComputationsPerClient.get(s).forEach(v -> v.getComputation().setComputationFinished()));
        // Streams don't have any active computation;
        activeComputationsPerClient.clear();
        // All streams are free;
    }

    public List<AbstractWorker> getWorkersList() {
        return new ArrayList<>(this.workersMap.values());
    }

    public List<String> getWorkersIdentifiers() {
        ArrayList<String> res = new ArrayList<>();
        for (AbstractWorker abstractWorker : this.workersMap.values()) {
            res.add(abstractWorker.getNodeIdentifier());
        }
        return res;
    }

    public AbstractWorker getWorker(String workerId) {
        return this.workersMap.get(workerId);
    }

    public AbstractWorkerSelectionPolicy getClientPolicy() {
        return workerSelectionPolicy;
    }

    /**
     * Cleanup and deallocate the streams managed by this manager;
     */
    public void cleanup(String path) {
        // close the connection with each worker retrieving the results
        if (executionContext.enableTimers) {
            for (AbstractWorker worker : this.getWorkersList()) {
                Timings workerTimings = worker.close();
                workerTimings.toJson(path + "/", worker.getNodeIdentifier() + ".json");
                LOGGER.fine("Worker " + worker + " closed with timings: " + workerTimings);
            }
        } else {
            for (AbstractWorker worker : this.getWorkersList()) {
                worker.close();
                LOGGER.fine("Worker " + worker + " closed");
            }
        }
        this.activeComputationsPerClient.clear();
        this.workerSelectionPolicy.cleanup();
    }
}
