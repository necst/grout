package com.necst.controller.utils;

import com.necst.controller.ControllerOptionMap;
import com.necst.controller.ControllerOptions;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.necst.controller.runtime.worker.options.DependencyPolicyEnum;
import com.necst.controller.runtime.worker.policy.WorkerSelectionPolicyEnum;

/**
 * Mock class to test the GrOUTExecutionContextTest, it has a null CUDARuntime;
 */
public class ControllerExecutionContextMock extends ControllerExecutionContext {

    public ControllerExecutionContextMock() {
        super(new ControllerOptionMap(new OptionValuesMockBuilder()
                .add(ControllerOptions.DependencyPolicy, DependencyPolicyEnum.WITH_CONST.toString())
                .add(ControllerOptions.WorkerSelectionPolicy, WorkerSelectionPolicyEnum.MIN_TRANSFER_SIZE.toString())
                .add(ControllerOptions.EnableMockWorker, true)
                .add(ControllerOptions.WorkersNetInfo, "WORKER:1111, WORKER:2222, WORKER:3333")
                .build()));
    }

}
