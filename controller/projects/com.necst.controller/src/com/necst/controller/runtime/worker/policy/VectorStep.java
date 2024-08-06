package com.necst.controller.runtime.worker.policy;

import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.runtime.worker.WorkersManager;

/**
 * This WorkerSelectionPolicy iterates over the workers in a round-robin fashion remaining on the same worker for K steps.
 */
public class VectorStep extends AbstractWorkerSelectionPolicy {
    //protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.WORKER_SELECTION_POLICY_LOGGER);
    int vectorStep_index;
    int worker_index;
    int elapsed_steps;
    int[] vectorStep;

    public VectorStep(WorkersManager workersManager, int[] vectorStep) {
        super(workersManager);

        vectorStep_index =0;
        elapsed_steps=1;
        worker_index=0;

        this.vectorStep = vectorStep;
    }

    @Override
    public AbstractWorker retrieveClient(ExecutionDAG.DAGVertex vertex) {
        AbstractWorker abstractWorker =  workersManager.getWorkersList().get(worker_index);
        if(elapsed_steps == vectorStep[vectorStep_index]) {
            vectorStep_index = (vectorStep_index+1)%vectorStep.length; // switch to the next step in the VectorStep
            elapsed_steps=1; // reset the taken step within a vectorStep value
            worker_index = (worker_index+1)%workersManager.getWorkersList().size(); // switch to the next worker
            // we need to change the current_index
        }else{
            elapsed_steps++;
        }
        return abstractWorker;
    }

    @Override
    public void cleanup() {

    }
}
