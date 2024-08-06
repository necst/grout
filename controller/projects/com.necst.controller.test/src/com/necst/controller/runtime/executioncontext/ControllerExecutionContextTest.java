package com.necst.controller.runtime.executioncontext;

import com.necst.controller.ControllerOptionMap;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.graalvm.polyglot.Value;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class ControllerExecutionContextTest{
    /**
     CUDA C++ source code of incrementing kernel.
     **/
    private static final String INCREMENT_KERNEL_SOURCE = "                   \n" +
            "__global__ void inc_kernel(int *out_arr, const int *in_arr, int num_elements) {     \n" +
            "  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;    \n" +
            "       idx += gridDim.x * blockDim.x) {                                         \n" +
            "    out_arr[idx] = in_arr[idx] + 1;                                     \n" +
            "  }                                                                             \n" +
            "}\n";

    @Rule
    public TestName name = new TestName();

    @Before
    public void displayTestName(){
        System.out.print("\n\n\n######## " + name.getMethodName()+" ########\n");
    }

    @Test
    public void testSimpleKernelLaunch() {
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

        Value kernelFun = buildKernel.execute(INCREMENT_KERNEL_SOURCE, "inc_kernel", "pointer, pointer, sint32");
        assertNotNull(kernelFun);
        assertTrue(kernelFun.canExecute());
        assertEquals(0, kernelFun.getMember("launchCount").asInt());
        assertNotNull(kernelFun.getMember("ptx").asString());

        //TODO: check that it actually creates 10 integer values --> 40 bytes
        Value arrOut = context.eval("controller", "int[10]");
        Value arrIn = context.eval("controller", "int[10]");
        for(int i=0; i<10; i++) {
           arrIn.setArrayElement(i,i);
           arrOut.setArrayElement(i, 0);
        }

        kernelFun.execute(1,1).execute(arrOut, arrIn, 10);

        System.out.println(arrOut);
        System.out.println(arrOut.getArraySize());
        // from here it seems that it has a size of 10 bytes
        for(int i=0; i<10; i++) {
            System.out.println(arrOut.getArrayElement(i));
        }
    }


    @Test
    public void avoidTransferIfAlreadyUptodate() {
        int SIZE = 100;
        Context context = Context.newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .allowPolyglotAccess(PolyglotAccess.ALL)
                .option("controller.WorkerSelectionPolicy", "round-robin")
                .option("controller.DependencyPolicy", "with-const")
                .option("controller.WorkersNetInfo", "192.168.0.1:1111, 192.168.0.2:2222")
                .option("controller.EnableMockWorker", "true")
                .build();


        Value buildKernel = context.eval("controller", "buildkernel");
        assertNotNull(buildKernel);

        Value kernelFun = buildKernel.execute(INCREMENT_KERNEL_SOURCE, "inc_kernel", "pointer, const pointer, sint32");

        Value arrOut = context.eval("controller", "int["+SIZE+"]");
        Value arrIn = context.eval("controller", "int["+SIZE+"]");
        for(int i=0; i<SIZE; i++) {
            arrIn.setArrayElement(i,i);
            arrOut.setArrayElement(i, 0);
        }

        Value fakeOut = context.eval("controller", "int["+SIZE+"]");
        Value fakeIn = context.eval("controller", "int["+SIZE+"]");

        /*
            BEFORE:
                arrOut = [CONTROLLER]
                arrIn = [CONTROLLER]
         */
        kernelFun.execute(1,100).execute(arrOut, arrIn, SIZE);
        /*
            AFTER:
                arrOut = [WORKER-1]
                arrIn = [CONTROLLER, WORKER-1] --> this is CONST
         */

        /*
             the previous kernel triggers a data movement from the controller to the worker
             for both arrOut and arrIn
         */


        // this is used to skip to the next worker given the round-robin policy and two workers in total
        kernelFun.execute(1,100).execute(fakeOut, fakeIn, SIZE);



        /*
            This kernel call finds that all input parameters are already up-to-date on the
            scheduled worker, therefore no data movement is performed
         */
        kernelFun.execute(1,100).execute(arrOut, arrIn, SIZE);

        context.close();

    }

}