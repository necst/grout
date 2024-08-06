package com.necst.controller.runtime;

import org.junit.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.rmi.RemoteException;

import static org.junit.Assert.*;

@Ignore("Class not ready for tests")
public class RuntimeWorkerTest {
    private RuntimeWorker worker;

    @Before
    public void setUp() {
        worker = new RuntimeWorker(new RuntimeWorkerOptions());
    }


    @Test
    public void runtimeWorker() {
        assertNotNull(worker);
        assertNotNull(worker.getContext());
        assertNotNull(worker.getKernelBuilder());
        assertEquals(0, worker.getDeviceArraysMap().size());
        assertEquals(0, worker.getKernelsMap().size());
    }

    @Test
    public void setArray() {
        setArrayHelper(worker);

        assertEquals(1, worker.getDeviceArraysMap().size());
        assertNotNull(worker.getDeviceArraysMap().get(1L));
        assertEquals(800, worker.getDeviceArraysMap().get(1L).getArraySize());

        for (int i = 0; i < 100; i++) {
            byte[] bytes = new byte[8];
            for (int j = 0; j < 8; j++) {
                bytes[j] = worker.getDeviceArraysMap().get(1L).getArrayElement(i * 8 + j).asByte();
            }
            assertEquals(1.69, ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(0), 0.0001);
        }

//        for (int i = 0; i < 100; i++) {
//            assertEquals(1.69, worker.getDeviceArraysMap().get(1L).getArrayElement(i * 8).asDouble(), 0.0001);
//        }

//        for (long i = 0; i < 2L * Integer.MAX_VALUE; i++) {
//            assertEquals((i + 1) % Integer.MAX_VALUE, worker.getDeviceArraysMap().get(2L).getArrayElement(i).asInt());
//        }
    }

    @Test
    public void buildKernel() {
        buildKernelHelper(worker);

        assertEquals(2, worker.getKernelsMap().size());
        assertNotNull(worker.getKernelsMap().get("squareRoot"));
    }

    @Test
    public void exec() {
        execHelper(worker);

        assertEquals(2, worker.getKernelsMap().size());
        assertNotNull(worker.getKernelsMap().get("squareRoot"));
        assertEquals(1, worker.getDeviceArraysMap().size());
        assertNotNull(worker.getDeviceArraysMap().get(1L));
        assertEquals(800, worker.getDeviceArraysMap().get(1L).getArraySize());
//        assertNotNull(worker.getDeviceArraysMap().get(2L));
//        assertEquals(2L * Integer.MAX_VALUE, worker.getDeviceArraysMap().get(2L).getArraySize());

        for (int i = 0; i < 100; i++) {
            byte[] bytes = new byte[8];
            for (int j = 0; j < 8; j++) {
                bytes[j] = worker.getDeviceArraysMap().get(1L).getArrayElement(i * 8 + j).asByte();
            }

            assertEquals(1.3, ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(0), 0.0001);
        }
//        for (long i = 0; i < 2L * Integer.MAX_VALUE; i++) {
//            assertEquals((i + 1) * 3, worker.getDeviceArraysMap().get(2L).getArrayElement(i).asInt());
//        }

    }

    @Test
    public void getArray() {
        execHelper(worker);

        double[] array = new double[100];
//        int[] array21 = null;
//        int[] array22 = null;

        for (int i = 0; i < 100; i++) {
                array[i] = ByteBuffer.wrap(worker.getArray(1, 0)).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(i);
        }

//        try {
//            array21 = ByteBuffer.wrap(worker.getArray(2, 0)).asIntBuffer().array();
//            array22 = ByteBuffer.wrap(worker.getArray(2, Integer.MAX_VALUE)).asIntBuffer().array();
//        } catch (RemoteException e) {
//            throw new RuntimeException(e);
//        }

        assertNotNull(array);

        assertEquals(100, array.length);
        assertEquals(worker.getDeviceArraysMap().get(1L).getArraySize(), array.length * 8);
        for (int i = 0; i < 100; i++) {
            assertEquals(1.3, array[i], 0.0001);
        }
//        for (int i = 0; i < 100; i++) {
//            assertEquals(worker.getDeviceArraysMap().get(1L).getArrayElement(i * 8).asDouble(), array[i], 0.0001);
//        }

//        assertEquals(2L * Integer.MAX_VALUE, (long) array21.length + array22.length);
//        assertEquals(worker.getDeviceArraysMap().get(2L).getArraySize(), (long) array21.length + array22.length);
//        for (int i = 0; i < Integer.MAX_VALUE; i++) {
//            assertEquals((i + 1) * 3, array21[i]);
//            assertEquals((i + Integer.MAX_VALUE + 1) * 3, array22[i]);
//        }
//        for (int i = 0; i < Integer.MAX_VALUE; i++) {
//            assertEquals(worker.getDeviceArraysMap().get(2L).getArrayElement(i).asInt(), array21[i]);
//            assertEquals(worker.getDeviceArraysMap().get(2L).getArrayElement(i + Integer.MAX_VALUE).asInt(), array22[i]);
//        }

    }

    @Test
    public void close() {
        closeHelper(worker);
        worker.getArray(10, 0);
        worker = new RuntimeWorker(new RuntimeWorkerOptions());
    }

    @After
    public void tearDown() {
        closeHelper(worker);
    }

    /////////////
    // helpers //
    /////////////

    public void setArrayHelper(RuntimeWorker worker) {
//        double[] array = new double[100];
//        int[] array21 = new int[Integer.MAX_VALUE - 1];
//        int[] array22 = new int[Integer.MAX_VALUE - 1];
//        for (int i = 0; i < 100; i++) {
//            array[i] = 1.69;
//        }
//        for (int j = 0; j < Integer.MAX_VALUE - 1; j++) {
//            array21[j] = j + 1;
//            array22[j] = j + 1;
//        }

        ByteBuffer buffer1 = ByteBuffer.allocate(100 * 8).order(ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < 100; i++) {
            buffer1.putDouble(i * 8, 1.69);
        }

        byte[] arrayByte1 = buffer1.array();

//        ByteBuffer buffer21 = ByteBuffer.allocateDirect(Integer.MAX_VALUE - 1);
//        IntBuffer intBuffer1 = buffer21.asIntBuffer();
//        ByteBuffer buffer22 = ByteBuffer.allocateDirect(Integer.MAX_VALUE - 1);
//        IntBuffer intBuffer2 = buffer22.asIntBuffer();
//        intBuffer1.put(array21);
//        intBuffer2.put(array22);

//        byte[] arrayByte21 = buffer21.array();
//        byte[] arrayByte22 = buffer22.array();

            worker.initializeArray(1, 800);
//            worker.initializeArray(2, 2L * (Integer.MAX_VALUE - 1));
            worker.setArray(1, 0, arrayByte1);
//            worker.setArray(2, 0, arrayByte21);
//            worker.setArray(2, Integer.MAX_VALUE - 1, arrayByte22);
        
    }

    public void buildKernelHelper(RuntimeWorker worker) {
        setArrayHelper(worker);
        String kernelStringSquareRoot = "__global__ void squareRoot(double *a) {\n" +
                "   int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                "   if(i<100)" +
                "       a[i] = sqrt(a[i]);\n" +
                "}";
        String kernelStringProduct = "__global__ void product(int *a, int b) {\n" +
                "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                "    a[i] = a[i] * b;\n" +
                "}";
            worker.buildKernel(kernelStringSquareRoot, "squareRoot", "pointer");
            worker.buildKernel(kernelStringProduct, "product", "pointer, sint32");

    }

    public void execHelper(RuntimeWorker worker) {
        buildKernelHelper(worker);
            worker.executeKernel("squareRoot", new int[]{1}, new int[]{100}, new String[]{"pointer"}, new Object[]{1L});
//        try {
//            worker.executeKernel("product", new int[]{1}, new int[]{5}, new Object[]{"pointer", "int"}, new Object[]{2, 3});
//        } catch (RemoteException e) {
//            e.printStackTrace();
//        }
    }

    public void closeHelper(RuntimeWorker worker) {
            worker.close();
            worker = null;
    }
}