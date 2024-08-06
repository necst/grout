package com.necst.controller.runtime.executioncontext;

import com.necst.controller.ControllerOptionMap;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public class TimingsTest {

    @Test
    public void testTimingsSave() {
        Context context = Context.newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .allowPolyglotAccess(PolyglotAccess.ALL)
                .option("controller.WorkerSelectionPolicy", ControllerOptionMap.DEFAULT_WORKER_SELECTION_POLICY.toString())
                .option("controller.DependencyPolicy", ControllerOptionMap.DEFAULT_DEPENDENCY_POLICY.toString())
                .option("controller.WorkersNetInfo", "192.168.0.1:1111, 192.168.0.2:2222")
                .option("controller.EnableMockWorker", "true")
                .build();


        Value buildKernel = context.eval("controller", "buildkernel");
        assertNotNull(buildKernel);

        System.out.println("closing the context");
        context.close();

        context.close();

    }
}