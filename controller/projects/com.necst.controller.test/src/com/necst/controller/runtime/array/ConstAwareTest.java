package com.necst.controller.runtime.array;

import com.necst.controller.ControllerOptionMap;
import com.necst.controller.ControllerOptionMapGen;
import com.necst.controller.ControllerOptions;
import com.necst.controller.Type;
import com.necst.controller.runtime.computation.dependency.DependencyPolicyEnum;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.necst.controller.utils.OptionValuesMockBuilder;
import org.graalvm.options.OptionValues;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.graalvm.polyglot.Value;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@Ignore("Class not ready for tests")
public class ConstAwareTest {

        @Test
        public void readOnlyInputParameter() {
                Context context = Context.newBuilder()
                                .allowAllAccess(true)
                                .allowExperimentalOptions(true)
                                .allowPolyglotAccess(PolyglotAccess.ALL)
                                .option("controller.WorkerSelectionPolicy",
                                                ControllerOptionMap.DEFAULT_WORKER_SELECTION_POLICY.toString())
                                .option("controller.DependencyPolicy",
                                                ControllerOptionMap.DEFAULT_DEPENDENCY_POLICY.toString())
                                .option("controller.WorkersNetInfo", "192.168.0.1:1111, 192.168.0.2:2222")
                                .option("controller.EnableMockWorker", "true")
                                .build();

                ControllerExecutionContext executionContext = new ControllerExecutionContext(
                                new ControllerOptionMap(
                                                new OptionValuesMockBuilder()
                                                                .add(ControllerOptions.DependencyPolicy,
                                                                                DependencyPolicyEnum.WITH_CONST
                                                                                                .toString())
                                                                .build()));
                int SIZE = 100;

                ControllerArray arr1 = new ControllerArray(executionContext, SIZE, Type.SINT32);
                ControllerArray arr2 = new ControllerArray(executionContext, SIZE, Type.SINT32);

                // TODO: in order to test the correct propagation we should do something similar
                // to GrOUT
                // DeviceArrayLocationMockTest.complexFrontierWithSyncMockTest(), we need to
                // implement lots of Mock classes

                context.close();

        }

}
