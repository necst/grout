package com.necst.controller.functions;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotAccess;
import org.graalvm.polyglot.Value;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;

import static org.junit.Assert.assertNotNull;

public class DistributeArrayFunctionTest {

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
    public void invokeDistributeArray() {
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


        Value distributeArray = context.eval("controller", "DistributeArray");
        assertNotNull(distributeArray);

        Value buildKernel = context.eval("controller", "buildkernel");
        Value kernelFun = buildKernel.execute(INCREMENT_KERNEL_SOURCE, "inc_kernel", "pointer, pointer, sint32");

        Value arr = context.eval("controller", "int["+SIZE+"]");
        for(int i=0; i<SIZE; i++) {
            arr.setArrayElement(i,i);
        }

        distributeArray.execute(arr);

        /*
            Given that ARR is already on all of the workers, this function should run directly without causing any data
            movements. Given that the input arrays are not "const" the arrays up-to-date locations should be updated
            to contain only the assigned worker.
         */
        kernelFun.execute(1,1).execute(arr, arr, SIZE);



        context.close();

    }
}