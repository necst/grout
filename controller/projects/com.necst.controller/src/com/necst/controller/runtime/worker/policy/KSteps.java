package com.necst.controller.runtime.worker.policy;

import java.util.Map;

import com.necst.controller.ControllerLogger;
import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.runtime.worker.WorkersManager;
import com.oracle.truffle.api.TruffleLogger;

/**
 * This WorkerSelectionPolicy iterates over the workers in a round-robin fashion remaining on the same worker for K steps.
 */
public class KSteps extends AbstractWorkerSelectionPolicy {
    //protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.WORKER_SELECTION_POLICY_LOGGER);
    private int curr = 0;
    private final int size;
    int steps;
    private int curr_steps = 1;
    private int cycles = 0; 

    public KSteps(WorkersManager workersManager) {
        super(workersManager);
        size = workersManager.getWorkersList().size();
        Map<String, String> map = System.getenv();
        steps = Integer.parseInt(map.getOrDefault("KSTEPS", "1")); // defaults to round-robin
        if(steps == 1){
            //LOGGER.warning("KSteps policy defaulting to Round-Robin (K=1)");
        }
        else if(steps < 1){
            steps = 1;
            //LOGGER.warning("KSteps policy defaulting to Round-Robin (K=1)");
        }
        
    }

    @Override
    public AbstractWorker retrieveClient(ExecutionDAG.DAGVertex vertex) {
        AbstractWorker worker = workersManager.getWorkersList().get(curr);
        
        if(curr_steps<steps){
            curr_steps++;
        }else{
            curr_steps=1;
            curr = (curr+1)%size; 
        }
        
        return worker;
    }

    @Override
    public void cleanup() {

    }
}
