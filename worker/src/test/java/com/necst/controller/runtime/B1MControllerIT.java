package com.necst.controller.runtime;

import org.junit.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.rmi.Naming;
import java.rmi.RemoteException;

import static java.lang.Thread.sleep;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class B1MControllerIT {
    private static Process process;
    private static Process process2;

    private static final int port = 1090;
    private static final int port2 = 1091;
    private static final int size = 100;
    private static final int P = 16;
    private static final int dim = 8;
    private static final int gridDim = 32;

    private static final String SQUARE_KERNEL = "" +
            "extern \"C\" __global__ void square(float* x, float* y, int n) { \n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        y[i] = x[i] * x[i];\n" +
            "    }\n" +
            "}\n";

    private static final String REDUCE_KERNEL = "" +
            "// From https://devblogs.nvidia.com/faster-parallel-reductions-kepler/\n" +
            "\n" + "__inline__ __device__ float warp_reduce(float val) {\n" +
            "    int warp_size = 32;\n" + "    for (int offset = warp_size / 2; offset > 0; offset /= 2)\n" +
            "        val += __shfl_down_sync(0xFFFFFFFF, val, offset);\n" +
            "    return val;\n" + "}\n" + "\n" + "__global__ void reduce(float *x, float *y, float* z, int N) {\n" +
            "    int warp_size = 32;\n" + "    float sum = float(0);\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {\n" +
            "        sum += x[i] - y[i];\n" + "    }\n" +
            "    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;\n" +
            "    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster\n" +
            "        atomicAdd(z, sum); // The first thread in the warp updates the output;\n" +
            "}";

    @BeforeClass
    public static void setUp() {
        try {
            process = Runtime.getRuntime().exec("./script.sh " + port);
            process2 = Runtime.getRuntime().exec("./script2.sh " + port2);
            //necessary to wait for the workers to start
            sleep(1000);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void B1M() {
        float res_tot = 0.0f;

        //create the arrays
        float[] x;
        float[] x1;
        float[] y;
        float[] y1;

        //create a support vector
        float[] res;

        //connect to worker
        Service service = null;
        try {
            String server = "rmi://localhost:" + port + "/worker";
            service = (Service) Naming.lookup(server);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assertNotNull(service);

        //build the context
        try {
            RuntimeWorkerOptions options = new RuntimeWorkerOptions();
            service.initContext(options);
        } catch (RemoteException e) {
            e.printStackTrace();
        }

        //build the kernels;
        try {
            service.buildKernel(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
            service.buildKernel(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");
        } catch (RemoteException e) {
            e.printStackTrace();
        }

        //create partitions
        ByteBuffer bufx;
        ByteBuffer bufx1;
        ByteBuffer bufy;
        ByteBuffer bufy1;
        ByteBuffer bufres;

        x = new float[size];
        x1 = new float[size];
        y = new float[size];
        y1 = new float[size];
        res = new float[1];

        //initialize the arrays
        for (int k = 0; k < size; k++) {
            x[k] = 1.0f / (k + 1);
            y[k] = 2.0f / (k + 1);
            res[0] = 0;
        }
        bufx = ByteBuffer.allocate(x.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufx1 = ByteBuffer.allocate(x1.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufy = ByteBuffer.allocate(y.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufy1 = ByteBuffer.allocate(y1.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufres = ByteBuffer.allocate(res.length * 4).order(ByteOrder.LITTLE_ENDIAN);

        bufx.asFloatBuffer().put(x);
        bufx1.asFloatBuffer().put(x1);
        bufy.asFloatBuffer().put(y);
        bufy1.asFloatBuffer().put(y1);
        bufres.asFloatBuffer().put(res);


        //send data to the workers
        for (int i = 0; i < P; i++) {
            try {
                service.sendArray(i, size * 4, 0, bufx.array());
                service.sendArray(i + P, size * 4, 0, bufx1.array());
                service.sendArray(i + 2 * P, size * 4, 0, bufy.array());
                service.sendArray(i + 3 * P, size * 4, 0, bufy1.array());
                service.sendArray(i + 4 * P, 4, 0, bufres.array());
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
        }

        //run the kernels
        for (int i = 0; i < P; i++) {
            try {
                service.execKernel(
                        "square",
                        new int[]{dim},
                        new int[]{gridDim},
                        new String[]{"pointer", "pointer", "int"},
                        new Object[]{(long) i, (long) (i + P), size}
                );
            } catch (RemoteException e) {
                e.printStackTrace();
            }

            try {
                service.execKernel(
                        "square",
                        new int[]{dim},
                        new int[]{gridDim},
                        new String[]{"pointer", "pointer", "int"},
                        new Object[]{(long) (i + 2 * P), (long) (i + 3 * P), size}
                );
            } catch (RemoteException e) {
                e.printStackTrace();
            }

            try {
                service.execKernel(
                        "reduce",
                        new int[]{dim},
                        new int[]{gridDim},
                        new String[]{"pointer", "pointer", "pointer", "int"},
                        new Object[]{(long) (i + P), (long) (i + 3 * P), (long) (i + 4 * P), size}
                );
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        }

        float acc = cpuValidation(size, P);

        for (int i = 0; i < P; i++) {
            float val = 0;

            try {
                byte[] arrayByte = service.getArray(i + 4 * P, 0);
                val = ByteBuffer.wrap(arrayByte, 0, 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get();
            } catch (RemoteException e) {
                e.printStackTrace();
            }
            if (!Float.isNaN(val))
                res_tot += val;
        }


        for (int i = 0; i < size; i++) {
            assertEquals(acc, res_tot, 1e-3);
        }

        try {
            service.closeContext();
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void B1M2workers() {
        float res_tot = 0.0f;

        //create the arrays
        float[] x;
        float[] x1;
        float[] y;
        float[] y1;

        //create a support vector
        float[] res;

        //connect to worker
        Service service = null;
        Service service2 = null;
        try {
            String server = "rmi://localhost:" + port + "/worker";
            String server2 = "rmi://localhost:" + port2 + "/worker";
            service = (Service) Naming.lookup(server);
            service2 = (Service) Naming.lookup(server2);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assertNotNull(service);
        assertNotNull(service2);

        //build the context
        try {
            service.initContext(new RuntimeWorkerOptions());
            service2.initContext(new RuntimeWorkerOptions());
        } catch (RemoteException e) {
            e.printStackTrace();
        }

        //build the kernels;
        try {
            service.buildKernel(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
            service.buildKernel(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");
            service2.buildKernel(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
        } catch (RemoteException e) {
            e.printStackTrace();
        }

        //create partitions
        ByteBuffer bufx = null;
        ByteBuffer bufx1 = null;
        ByteBuffer bufy = null;
        ByteBuffer bufy1 = null;
        ByteBuffer bufres = null;

        x = new float[size];
        x1 = new float[size];
        y = new float[size];
        y1 = new float[size];
        res = new float[1];

        //initialize the arrays
        for (int k = 0; k < size; k++) {
            x[k] = 1.0f / (k + 1);
            y[k] = 2.0f / (k + 1);
            res[0] = 0;
        }
        bufx = ByteBuffer.allocate(x.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufx1 = ByteBuffer.allocate(x1.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufy = ByteBuffer.allocate(y.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufy1 = ByteBuffer.allocate(y1.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bufres = ByteBuffer.allocate(res.length * 4).order(ByteOrder.LITTLE_ENDIAN);

        bufx.asFloatBuffer().put(x);
        bufx1.asFloatBuffer().put(x1);
        bufy.asFloatBuffer().put(y);
        bufy1.asFloatBuffer().put(y1);
        bufres.asFloatBuffer().put(res);

        //send data to the workers
        for (int i = 0; i < P; i++) {
            try {
                service.sendArray(i, size * 4, 0, bufx.array());
                service.sendArray(i + P, size * 4, 0, bufx1.array());
                service2.sendArray(i + 2 * P, size * 4, 0, bufy.array());
                service2.sendArray(i + 3 * P, size * 4, 0, bufy1.array());
                service.sendArray(i + 4 * P, 4, 0, bufres.array());
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }
        }

        //run the kernels
        for (int i = 0; i < P; i++) {
            try {
                service.execKernel(
                        "square",
                        new int[]{dim},
                        new int[]{gridDim},
                        new String[]{"pointer", "pointer", "int"},
                        new Object[]{(long) i, (long) i + P, size}
                );
            } catch (RemoteException e) {
                e.printStackTrace();
            }

            try {
                service2.execKernel(
                        "square",
                        new int[]{dim},
                        new int[]{gridDim},
                        new String[]{"pointer", "pointer", "int"},
                        new Object[]{(long) i + 2 * P, (long) i + 3 * P, size}
                );
            } catch (RemoteException e) {
                e.printStackTrace();
            }

            try {
                byte[] y1from2 = service2.getArray(i + 3 * P, 0);
                service.sendArray(i + 3 * P, size * 4, 0, y1from2);
            } catch (RemoteException e) {
                throw new RuntimeException(e);
            }

            try {
                service.execKernel(
                        "reduce",
                        new int[]{dim},
                        new int[]{gridDim},
                        new String[]{"pointer", "pointer", "pointer", "int"},
                        new Object[]{(long) i + P, (long) i + 3 * P, (long) i + 4 * P, size}
                );
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        }

        float acc = cpuValidation(size, P);

        for (int i = 0; i < P; i++) {
            float val = 0;
            try {
                byte[] arrayByte = service.getArray(i + 4 * P, 0);
                val = ByteBuffer.wrap(arrayByte, 0, 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get();
            } catch (RemoteException e) {
                e.printStackTrace();
            }
            if (!Float.isNaN(val))
                res_tot += val;
        }


        for (int i = 0; i < size; i++) {
            assertEquals(acc, res_tot, 1e-3);
        }

        try {
            service.closeContext();
        } catch (RemoteException e) {
            e.printStackTrace();
        }

    }

    public float cpuValidation(int size, int P) {
        float[] xHost = new float[size * P];
        float[] yHost = new float[size * P];
        float acc = 0.0f;

        for (int k = 0; k < size * P; k++) {
            xHost[k] = 1.0f / ((k % size) + 1);
            yHost[k] = 2.0f / ((k % size) + 1);
        }

        for (int k = 0; k < size * P; k++) {
            xHost[k] = xHost[k] * xHost[k];
            yHost[k] = yHost[k] * yHost[k];
            xHost[k] -= yHost[k];
        }

        for (int k = 0; k < size * P; k++) {
            acc += xHost[k];
        }

        return acc;
    }

    @AfterClass
    public static void tearDown() {
        process.destroy();
        process2.destroy();
    }
}
