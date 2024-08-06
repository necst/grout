package com.necst.controller.runtime.worker.policy;

import com.necst.controller.ControllerOptionMap;
import com.necst.controller.ControllerOptions;
import com.necst.controller.runtime.worker.WorkersManager;
import com.necst.controller.runtime.worker.options.DependencyPolicyEnum;
import com.necst.controller.utils.OptionValuesMockBuilder;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class KStepsTest {

    @Test
    public void retrieveClient() {
        ControllerOptionMap controllerOptionMap = new ControllerOptionMap(new OptionValuesMockBuilder()
                .add(ControllerOptions.DependencyPolicy, DependencyPolicyEnum.WITH_CONST.toString())
                .add(ControllerOptions.WorkerSelectionPolicy, WorkerSelectionPolicyEnum.MIN_TRANSFER_SIZE.toString())
                .add(ControllerOptions.EnableMockWorker, true)
                .add(ControllerOptions.WorkersNetInfo, "WORKER:1111, WORKER:2222, WORKER:3333")
                .build());

        WorkersManager workersManager = new WorkersManager(controllerOptionMap, null);
        KSteps kSteps = new KSteps(workersManager);
        kSteps.steps = 3;

        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:1111");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:1111");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:1111");

        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:2222");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:2222");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:2222");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:2222");

        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:3333");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:3333");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:3333");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:3333");
        assertEquals(kSteps.retrieveClient(null).getNodeIdentifier(),"WORKER:3333");
    }
}