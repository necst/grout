package com.necst.controller.runtime;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import java.net.MalformedURLException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.util.HashMap;
import java.util.Map;

public class RuntimeWorker {
    private static long MAX_ARRAY_OFFSET = (1 << 30);
    private final Value kernelBuilder;
    private final Value deviceArrayBuilder;
    private Context context;
    private Map<Long, Value> deviceArraysMap;
    private Map<Long, String> deviceArraysType;
    private Map<String, Value> kernelsMap;
    private Map<String, Service> remoteWorkers;

    public RuntimeWorker(RuntimeWorkerOptions options) {
        Context.Builder builder = Context.newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE");

        if (options.executionPolicy != null)
            builder.option("grcuda.ExecutionPolicy", options.executionPolicy);
        if (options.inputPrefetch != null)
            builder.option("grcuda.InputPrefetch", String.valueOf(options.inputPrefetch));
        if (options.retrieveNewStreamPolicy != null)
            builder.option("grcuda.RetrieveNewStreamPolicy", options.retrieveNewStreamPolicy);
        if (options.retrieveParentStreamPolicy != null)
            builder.option("grcuda.RetrieveParentStreamPolicy", options.retrieveParentStreamPolicy);
        if (options.dependencyPolicy != null)
            builder.option("grcuda.DependencyPolicy", options.dependencyPolicy);
        if (options.deviceSelectionPolicy != null)
            builder.option("grcuda.DeviceSelectionPolicy", options.deviceSelectionPolicy);
        if (options.forceStreamAttach != null)
            builder.option("grcuda.ForceStreamAttach", String.valueOf(options.forceStreamAttach));
        if (options.enableComputationTimers != null)
            builder.option("grcuda.EnableComputationTimers", String.valueOf(options.enableComputationTimers));
        if (options.memAdvisePolicy != null)
            builder.option("grcuda.MemAdvisePolicy", options.memAdvisePolicy);
        if (options.numGpus != null)
            builder.option("grcuda.NumberOfGPUs", String.valueOf(options.numGpus));
        if (options.bandwidthMatrix != null)
            builder.option("grcuda.BandwidthMatrix", System.getenv("GRCUDA_HOME") + "/projects/resources/connection_graph/datasets/connection_graph.csv");

        context = builder.build();

        deviceArraysMap = new HashMap<>();
        deviceArraysType = new HashMap<>();
        kernelsMap = new HashMap<>();
        kernelBuilder = context.eval("grcuda", "buildkernel");
        deviceArrayBuilder = context.eval("grcuda", "CU").getMember("DeviceArray");

        remoteWorkers = new HashMap<>();
        for (String worker : options.workers) {
            try {
                String hostname = worker.split(":")[0];
                short port = Short.parseShort(worker.split(":")[1]);
                Service toAdd = (Service) Naming.lookup("rmi://" + hostname + ":" + port + "/worker");
                remoteWorkers.put(worker, toAdd);
            } catch (NotBoundException | MalformedURLException | RemoteException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void initializeArray(long arrayHash, long arraySize) {
        //System.out.printf("     initializeArray(ID:%d, size:%d )\n", arrayHash, arraySize);
        if (deviceArraysMap.get(arrayHash) == null) { // create only if not already present
            Value vector;
            //System.out.println("creating array of size " + arraySize);
            if (arraySize % 8 == 0) {
                //System.out.println("creating array of double");
                arraySize /= 8;
                vector = deviceArrayBuilder.execute("double", arraySize);
                deviceArraysType.put(arrayHash, "double");
            } else if (arraySize % 4 == 0) {
                //System.out.println("creating array of float");
                arraySize /= 4;
                vector = deviceArrayBuilder.execute("float", arraySize);
                deviceArraysType.put(arrayHash, "float");
            } else {
                //System.out.println("creating array of char");
                vector = deviceArrayBuilder.execute("char", arraySize);
                deviceArraysType.put(arrayHash, "char");
            }
            deviceArraysMap.put(arrayHash, vector);
        }
    }

    public void setArray(long arrayHash, long offset, byte[] data) throws NullPointerException {
        //System.out.printf("     setArray(ID: %d, offset: %d)\n", arrayHash, offset);

        Value deviceArray = deviceArraysMap.get(arrayHash);
        if (deviceArray == null) {
            throw new NullPointerException("ARRAY NOT FOUND");
        }

        long global_offset = offset;
        long local_offset;
        int size_in_byte = data.length;
        ByteBuffer dataBuf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        if (deviceArraysType.get(arrayHash).equals("double")) {
            //System.out.println("setting element every 8 bytes");
            local_offset = global_offset/8; // the offset based on double
            int size_in_double = size_in_byte/8;
            int byte_index = 0;
            for (int local_index = 0; local_index < size_in_double ; local_index++) {
                // .setArrayElement(double_index, dataBuf.getDouble(byte_index));
                deviceArray.setArrayElement(local_offset + local_index, dataBuf.getDouble(byte_index));
                byte_index+=8;
            }
        } else if (deviceArraysType.get(arrayHash).equals("float")) {
            //System.out.println("setting element every 4 bytes");
            local_offset = global_offset/4; // the offset based on double
            int size_in_float = size_in_byte/4;
            int byte_index = 0;
            for (int local_index = 0; local_index < size_in_float ; local_index++) {
                deviceArray.setArrayElement(local_offset + local_index, dataBuf.getFloat(byte_index));
                byte_index+=4;
            }
        } else {
            //System.out.println("setting element every 1 byte");
            for (int j = 0; j < size_in_byte; j++) {
                deviceArray.setArrayElement(global_offset + j, data[j]);
            }
        }
    }

    public void syncToGetArray(long arrayHash) {
        if (deviceArraysMap.get(arrayHash) != null) {
            Byte a = deviceArraysMap.get(arrayHash).getArrayElement(0).asByte();
        }
    }

    public byte[] getArray(long arrayHash, long offset) throws NullPointerException {
        //System.out.println("returning the arrray to the controller");

        Value deviceArray = deviceArraysMap.get(arrayHash);
        if (deviceArray == null)
            throw new NullPointerException("DEVICE ARRAY NOT FOUND");

        // generate the byte[] to send back
        long size = deviceArray.getArraySize(); // size in double;
        byte[] data;

        if (deviceArraysType.get(arrayHash).equals("double")) {
            //System.out.println("getting element every 8 bytes");

            int bufferSize = (int) Math.min(size * 8 - offset, MAX_ARRAY_OFFSET); // size in bytes

            ByteBuffer dataBuf = ByteBuffer.allocate(bufferSize).order(ByteOrder.LITTLE_ENDIAN);
            
            long local_index = 0;
            long local_offset = offset/8;
            for(int global_index=0; global_index < bufferSize; global_index+=8){
                double val = deviceArray.getArrayElement(local_offset + local_index).asDouble();
                dataBuf.putDouble(global_index, val);
                local_index++;
            }

            data = dataBuf.array();

        } else if (deviceArraysType.get(arrayHash).equals("float")) {
            //System.out.println("getting element every 4 bytes");

            int bufferSize = (int) Math.min(size * 4 - offset, MAX_ARRAY_OFFSET); // size in bytes

            ByteBuffer dataBuf = ByteBuffer.allocate(bufferSize).order(ByteOrder.LITTLE_ENDIAN);
            
            long local_index = 0;
            long local_offset = offset/4;
            for(int global_index=0; global_index < bufferSize; global_index+=4){
                float val = deviceArray.getArrayElement(local_offset + local_index).asFloat();
                dataBuf.putFloat(global_index, val);
                local_index++;
            }

            data = dataBuf.array();

        } else {
            //System.out.println("getting element every 1 byte");
            
            data = new byte[(int) Math.min(size - offset, MAX_ARRAY_OFFSET)];
            for (int i = 0; i < data.length; i++) {
                data[i] = deviceArray.getArrayElement((long) (i) + offset).asByte();
            }
        }

        return data;
    }

    public void buildKernel(String kernelString, String kernelName, String kernelSignature) {
        //System.out.printf("\nBUILDING KERNEL:\n kernelName: %s \n kernelSignature: %s \n kernelString: \n %s \n\n", kernelName, kernelSignature, kernelString);
        Value kernel = kernelBuilder.execute(kernelString, kernelName, kernelSignature);
        kernelsMap.put(kernelName, kernel);
    }

    public void executeKernel(String kernelName, int[] blocks, int[] threadsPerBlock, String[] types, Object[] data) {
        Object[] args = new Object[types.length];

        for (int i = 0; i < args.length; i++) {
            if (!types[i].equals("constant")) {
                if (!deviceArraysMap.containsKey((long) data[i])) {
                    initializeArray((long) data[i], Long.parseLong(types[i]));
                }
                args[i] = deviceArraysMap.get((long) data[i]);
            } else {
                args[i] = data[i];
            }
        }

        //System.out.println("blocks: "+Arrays.toString(blocks)+" threads: "+Arrays.toString(threadsPerBlock));
        kernelsMap.get(kernelName).execute(blocks, threadsPerBlock).execute(args);

    }

    public void p2pTransfer(long arrayID, long sizeInBytes, String workerID) throws RemoteException {
        //System.out.printf("p2pTransfer getting ID(%d) from worker=%s\n", arrayID, workerID);
        Service remoteService = remoteWorkers.get(workerID);
        if (remoteService == null)
            throw new RemoteException("REMOTE SERVICE NOT FOUND");

        initializeArray(arrayID, sizeInBytes);
        Value array = deviceArraysMap.get(arrayID);
        if (array == null) {
            throw new NullPointerException("ARRAY NOT FOUND");
        }

        int numParts = (int) (sizeInBytes / MAX_ARRAY_OFFSET); // compute the number of parts that i NEED to receive
        if (sizeInBytes % MAX_ARRAY_OFFSET != 0) { // increment by one if it is not a perfect multiple
            numParts++;
        }
        byte[] part; // temporary array to store each segment

        long global_offset, local_offset;
        int size_in_byte; 

        for(int i=0; i < numParts; i++){
            global_offset = (long) i * MAX_ARRAY_OFFSET; // the offset when dealing with bytes

            part = remoteService.getArray(arrayID, global_offset); // get the part in bytes
            size_in_byte = part.length;
            ByteBuffer dataBuf = ByteBuffer.wrap(part).order(ByteOrder.LITTLE_ENDIAN);

            if (deviceArraysType.get(arrayID).equals("double")) {
                //System.out.println("setting element every 8 bytes");
                local_offset = global_offset/8; // the offset based on double
                int size_in_double = size_in_byte/8;
                int byte_index = 0;
                for (int local_index = 0; local_index < size_in_double ; local_index++) {
                    // .setArrayElement(double_index, dataBuf.getDouble(byte_index));
                    array.setArrayElement(local_offset + local_index, dataBuf.getDouble(byte_index));
                    byte_index+=8;
                }
            } else if (deviceArraysType.get(arrayID).equals("float")) {
                //System.out.println("setting element every 4 bytes");
                local_offset = global_offset/4; // the offset based on double
                int size_in_float = size_in_byte/4;
                int byte_index = 0;
                for (int local_index = 0; local_index < size_in_float ; local_index++) {
                    array.setArrayElement(local_offset + local_index, dataBuf.getFloat(byte_index));
                    byte_index+=4;
                }
            } else {
                //System.out.println("setting element every 1 byte");
                for (int j = 0; j < part.length; j++) {
                    array.setArrayElement(global_offset + j, part[j]);
                }
            }
        }

    }

    public void close() {
        for (Value v : deviceArraysMap.values()) {
            v.invokeMember("free");
        }

        deviceArraysMap.clear();
        kernelsMap.clear();
        context.close();
    }

    ///////////////
    // utilities //
    ///////////////

    public Context getContext() {
        return context;
    }

    public Value getKernelBuilder() {
        return kernelBuilder;
    }

    public Map<Long, Value> getDeviceArraysMap() {
        return deviceArraysMap;
    }

    public Map<String, Value> getKernelsMap() {
        return kernelsMap;
    }
}
