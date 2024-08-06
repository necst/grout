package com.necst.controller.runtime.worker.policy;

import com.necst.controller.ControllerOptionMap;
import com.necst.controller.ControllerOptions;
import com.necst.controller.Type;
import com.necst.controller.runtime.array.ControllerArray;
import com.necst.controller.runtime.computation.ComputationArgument;
import com.necst.controller.runtime.computation.ComputationArgumentWithValue;
import com.necst.controller.runtime.computation.dependency.DependencyPolicyEnum;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.necst.controller.runtime.executioncontext.ExecutionDAG;

import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.utils.KernelExecutionMock;
import com.necst.controller.utils.OptionValuesMockBuilder;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;

@Ignore("NOT ready for test")
public class MinTransferTimePolicyTest {

    @Test
    public void retrieveClient() {
        // create the ControllerExecutionContext

        ControllerOptionMap controllerOptionMap = new ControllerOptionMap(new OptionValuesMockBuilder()
                .add(ControllerOptions.DependencyPolicy, DependencyPolicyEnum.WITH_CONST.toString())
                .add(ControllerOptions.WorkerSelectionPolicy, WorkerSelectionPolicyEnum.MIN_TRANSFER_TIME.toString())
                .add(ControllerOptions.EnableMockWorker, true)
                .add(ControllerOptions.EnableTimers, false)
                .add(ControllerOptions.WorkersNetInfo, "WORKER:1111, WORKER:2222, WORKER:3333")
                .build());

        // TODO: fix problem with truffle logger
        ControllerExecutionContext controllerExecutionContext = new ControllerExecutionContext(controllerOptionMap);

        // create the KernelExecution
        ArrayList<ComputationArgumentWithValue> args = new ArrayList<>();
        ControllerArray in_1 = new ControllerArray(controllerExecutionContext, 100, Type.SINT32);
        ControllerArray in_2 = new ControllerArray(controllerExecutionContext, 100, Type.SINT32);
        ControllerArray out = new ControllerArray(controllerExecutionContext, 10000, Type.SINT32);

        in_1.addArrayUpToDateLocations("WORKER:1111");
        in_1.addArrayUpToDateLocations("WORKER:2222");
        in_1.addArrayUpToDateLocations("WORKER:3333");

        in_2.addArrayUpToDateLocations("WORKER:1111");
        in_2.addArrayUpToDateLocations("WORKER:2222");

        out.addArrayUpToDateLocations("WORKER:2222");
        out.addArrayUpToDateLocations("WORKER:3333");

        args.add(new ComputationArgumentWithValue(new ComputationArgument(0, "IN-1", Type.NFI_POINTER, ComputationArgument.Kind.POINTER_IN), new ControllerArray(controllerExecutionContext, 100, Type.SINT32)));
        args.add(new ComputationArgumentWithValue(new ComputationArgument(1, "IN-2", Type.NFI_POINTER, ComputationArgument.Kind.POINTER_IN), new ControllerArray(controllerExecutionContext, 100, Type.SINT32)));
        args.add(new ComputationArgumentWithValue(new ComputationArgument(2, "OUT", Type.NFI_POINTER, ComputationArgument.Kind.POINTER_INOUT), new ControllerArray(controllerExecutionContext, 10000, Type.SINT32)));
        KernelExecutionMock kernelExecution = new KernelExecutionMock(controllerExecutionContext, args, "kernel(IN-1, IN-2, OUT)");

        try {
            controllerExecutionContext.registerExecution(kernelExecution);
        } catch (UnsupportedTypeException e) {
            throw new RuntimeException(e);
        }

        ExecutionDAG.DAGVertex vertex = controllerExecutionContext.getDag().getVertices().get(0); // there is only the previously defined vertex
        MinTransferTimePolicy minTransferTimePolicy = new MinTransferTimePolicy(controllerExecutionContext.workersManager, "link_bandwidth.csv");

        AbstractWorker selectedWorker = minTransferTimePolicy.retrieveClient(vertex);

        System.out.println(selectedWorker.getNodeIdentifier());

        assert selectedWorker.getNodeIdentifier().equals("WORKER:2222");
    }
}