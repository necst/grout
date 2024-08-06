package com.necst.controller.runtime.array;

import com.necst.controller.ControllerOptionMap;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.graalvm.polyglot.Value;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ControllerArrayTest {

    @Test
    public void naiveTest(){
        int SIZE = 100;
        Context context = Context.newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .allowPolyglotAccess(PolyglotAccess.ALL)
                .option("controller.WorkerSelectionPolicy", ControllerOptionMap.DEFAULT_WORKER_SELECTION_POLICY.toString())
                .option("controller.DependencyPolicy", ControllerOptionMap.DEFAULT_DEPENDENCY_POLICY.toString())
                .option("controller.WorkersNetInfo", "192.168.0.1:1111, 192.168.0.2:2222")
                .option("controller.EnableMockWorker", "true")
                .build();
        Value controllerArray = context.eval("controller", String.format("int[%d]", SIZE));

        for(int i=0; i<SIZE; i++)
            controllerArray.setArrayElement(i, i);

        assertTrue(controllerArray.hasArrayElements());
        assertEquals(SIZE,  controllerArray.getArraySize());
        for(int i=0; i<SIZE; i++)
            assertEquals(i,  controllerArray.getArrayElement(i).asInt());

        context.close();
    }


}
