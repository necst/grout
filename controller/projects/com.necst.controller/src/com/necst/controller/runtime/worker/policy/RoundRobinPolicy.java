package com.necst.controller.runtime.worker.policy;

import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.runtime.worker.WorkersManager;

/**
 * This WorkerSelectionPolicy iterates over the workers in a round-robin fashion.
 */
public class RoundRobinPolicy extends AbstractWorkerSelectionPolicy {

    int curr = 0;
    int size;

    public RoundRobinPolicy(WorkersManager workersManager) {
        super(workersManager);
        size = workersManager.getWorkersList().size();
    }

    @Override
    public AbstractWorker retrieveClient(ExecutionDAG.DAGVertex vertex) {
        AbstractWorker worker = workersManager.getWorkersList().get(curr);
        curr = (curr+1)%size;
        return worker;
    }

    @Override
    public void cleanup() {

    }
}
