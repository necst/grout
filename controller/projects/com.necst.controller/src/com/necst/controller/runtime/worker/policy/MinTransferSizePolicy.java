package com.necst.controller.runtime.worker.policy;

import com.necst.controller.ControllerLogger;
import com.necst.controller.runtime.Controller;
import com.necst.controller.runtime.array.AbstractArray;
import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.runtime.worker.WorkersManager;
import com.oracle.truffle.api.TruffleLogger;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MinTransferSizePolicy extends AbstractWorkerSelectionPolicy {
    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.WORKER_SELECTION_POLICY_LOGGER);

    AbstractWorkerSelectionPolicy fallbackPolicy;
    float fallbackThreshold = 0.9f; // X% of the data needs to be on a worker in order to be considered for minTransferSize

    public MinTransferSizePolicy(WorkersManager workersManager) {
        super(workersManager);
        fallbackPolicy = new RoundRobinPolicy(workersManager);
    }

    private void computeDataSizeOnWorkers(ExecutionDAG.DAGVertex vertex, HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers) {
        List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();
        for (AbstractArray a : arguments) {
            for (String location : a.getArrayUpToDateLocations()) {
                if (!location.equals(Controller.CONTROLLER_NODE)) {
                    AbstractWorker worker = workersManager.getWorker(location);
                    bytesAlreadyPresentOnWorkers.put(worker, bytesAlreadyPresentOnWorkers.get(worker) + a.getSizeBytes());
                }
            }
        }
    }

    private boolean findIfAnyDeviceHasEnoughData(HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers, ExecutionDAG.DAGVertex vertex) {
        // Total size of the input arguments;
        long totalSize = vertex.getComputation().getArrayArguments().stream().map(AbstractArray::getSizeBytes).reduce(0L, Long::sum);

        // True if at least one device already has at least X% of the data required by the computation;
        for(AbstractWorker workerID : bytesAlreadyPresentOnWorkers.keySet()) {
            if((float) bytesAlreadyPresentOnWorkers.get(workerID) / totalSize > this.fallbackThreshold) {
                return true;
            }
        }
        return false;
    }

    private AbstractWorker selectDeviceWithMostData(HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers) {
        // Find device with maximum available data;
        // probably can be optimized
        AbstractWorker workerWithMaximumAvailableData = (AbstractWorker) bytesAlreadyPresentOnWorkers.keySet().toArray()[0];
        for (Map.Entry<AbstractWorker, Long> entry: bytesAlreadyPresentOnWorkers.entrySet()) {
            if (entry.getValue() > bytesAlreadyPresentOnWorkers.get(workerWithMaximumAvailableData)) {
                workerWithMaximumAvailableData = entry.getKey();
            }
        }
        return workerWithMaximumAvailableData;
    }

    @Override
    public AbstractWorker retrieveClient(ExecutionDAG.DAGVertex vertex) {
        // Map that tracks the size, in bytes, of data that is already present on each worker;
        HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers = new HashMap<>();
        this.workersManager.getWorkersList().forEach(worker -> bytesAlreadyPresentOnWorkers.put(worker, 0L)); // initialize the map

        // Compute the amount of data on each worker, and if any worker has any data at all;
        computeDataSizeOnWorkers(vertex, bytesAlreadyPresentOnWorkers);
        LOGGER.fine("retrieveClient("+vertex.getComputation()+")");
        bytesAlreadyPresentOnWorkers.forEach((key, value) -> LOGGER.fine(key.getNodeIdentifier() + " --> " + value ));

        // If no device has at least X% of data available, it's not worth optimizing data locality (exploration preferred to exploitation);
        if (findIfAnyDeviceHasEnoughData(bytesAlreadyPresentOnWorkers, vertex)) {
            // Find device with maximum available data;
            AbstractWorker abst = selectDeviceWithMostData(bytesAlreadyPresentOnWorkers);
            LOGGER.fine("Found a worker with enough data: " + abst.getNodeIdentifier());
            return abst;
        } else {
            // No data is present on any GPU: select the device with round-robin;
            AbstractWorker abst = fallbackPolicy.retrieveClient(vertex);
            LOGGER.fine("Fallback to round-robin" + abst.getNodeIdentifier());
            return abst;
        }
    }


    @Override
    public void cleanup() {

    }
}
