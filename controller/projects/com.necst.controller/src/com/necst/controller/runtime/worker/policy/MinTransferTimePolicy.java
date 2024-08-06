package com.necst.controller.runtime.worker.policy;

import com.necst.controller.ControllerLogger;
import com.necst.controller.runtime.Controller;
import com.necst.controller.runtime.array.AbstractArray;
import com.necst.controller.runtime.executioncontext.ExecutionDAG;
import com.necst.controller.runtime.worker.AbstractWorker;
import com.necst.controller.runtime.worker.WorkersManager;
import com.oracle.truffle.api.TruffleLogger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import static java.lang.Math.max;

/**
 * This MinTransferTimePolicy selects the worker that requires less time to update all required data for computation.
 */
public class MinTransferTimePolicy extends AbstractWorkerSelectionPolicy {
    protected static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.WORKER_SELECTION_POLICY_LOGGER);
    private final double[][] linkBandwidth = new double[workersManager.getWorkersList().size() + 1][workersManager.getWorkersList().size() + 1];
    private final float fallbackThreshold = 0.9f; // X% of the data needs to be on a worker in order to be considered for minTransferSize
    AbstractWorkerSelectionPolicy fallbackPolicy;

    public MinTransferTimePolicy(WorkersManager workersManager, String bandwidthMatrixPath) {
        super(workersManager);
        fallbackPolicy = new RoundRobinPolicy(workersManager);

        List<List<String>> records = new ArrayList<>();
        // Read each line in the CSV and store each line into a string array, splitting strings on ",";
        try (BufferedReader br = new BufferedReader(new FileReader(bandwidthMatrixPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Read each line, and reconstruct the bandwidth matrix.
        // Given N Workers and 1 CPU, we have a [Worker + 1][Worker+ 1] symmetric matrix.
        // Each line is "start_id", "end_id", "bandwidth";
        for (int il = 1; il < records.size(); il++) {
            int startDevice = Integer.parseInt(records.get(il).get(0));
            int endDevice = Integer.parseInt(records.get(il).get(1));
            // Skip invalid entries, and ignore Workers with ID larger than the number of Workers to use;
            if (startDevice >= -1 && startDevice < workersManager.getWorkersList().size() && endDevice >= -1 && endDevice < workersManager.getWorkersList().size()) {
                // Approximate to the floor, to smooth random bandwidth fluctuations in data transfer;
                double bandwidth = Math.floor(Double.parseDouble(records.get(il).get(2)));
                if (startDevice != -1) {
                    // Worker-Worker interconnection;
                    this.linkBandwidth[startDevice][endDevice] = bandwidth;
                } else {
                    // -1 identifies CPU-Worker interconnection, store it in the last spot;
                    this.linkBandwidth[workersManager.getWorkersList().size()][endDevice] = bandwidth;
                    this.linkBandwidth[endDevice][workersManager.getWorkersList().size()] = bandwidth;
                }
            }
        }
        // Interconnections are supposedly symmetric. Enforce this behavior by averaging results.
        // In other words, B[i][j] = B[j][j] <- (B[i][j] + B[j][i]) / 2.
        // Ignore the diagonal, and the last column and row (it represents the CPU and is already symmetric by construction);
        for (int i = 0; i < this.linkBandwidth.length - 1; i++) {
            for (int j = i; j < this.linkBandwidth.length - 1; j++) {
                double averageBandwidth = (this.linkBandwidth[i][j] + this.linkBandwidth[j][i]) / 2;
                this.linkBandwidth[i][j] = averageBandwidth;
                this.linkBandwidth[j][i] = averageBandwidth;
            }
        }
    }

    private void computeDataSizeOnWorkers(ExecutionDAG.DAGVertex vertex, HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers) {
        List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();
        for (AbstractArray a : arguments) {
            for (String location : a.getArrayUpToDateLocations()) {
                if (!location.equals(Controller.CONTROLLER_NODE)) {
                    AbstractWorker worker = workersManager.getWorker(location);
                    bytesAlreadyPresentOnWorkers.put(worker, bytesAlreadyPresentOnWorkers.get(worker) + a.getSizeBytes());
                }
            }
        }
    }

    public double computeBandwidth(int target, Set<Integer> upToDateLocations) {
        // Hypotheses: we consider the max bandwidth towards the device target.
        // Initialization: min possible value, bandwidth = 0 GB/sec;
        double bandwidth = 0;
        // Check that data is updated at least in some location. This is a precondition that must hold;
        if (upToDateLocations == null || upToDateLocations.isEmpty()) {
            throw new IllegalStateException("data is not updated in any location, when estimating bandwidth for device=" + target);
        }
        // If array an already present in device target, the transfer bandwidth to it is infinity.
        // We don't need to transfer it, so its transfer time will be 0;
        if (upToDateLocations.contains(target)) {
            bandwidth = Double.POSITIVE_INFINITY;
        } else {
            // Otherwise we consider the bandwidth to move array a to device target,
            // from each possible location containing the array a;
            List<Integer> upToDateLocationsList = new ArrayList<>(upToDateLocations);
            for (Integer from : upToDateLocationsList) {
                // The matrix is symmetric, loading [target][fromDevice] is faster than [fromDevice][target];
                bandwidth = max(linkBandwidth[target][from], bandwidth);
            }
        }
        return bandwidth;
    }

    private boolean computeTransferTimes(ExecutionDAG.DAGVertex vertex, HashMap<AbstractWorker, Double> argumentTransferTime) {
        List<AbstractArray> arguments = vertex.getComputation().getArrayArguments();

        // True if there's at least a Worker with some data already available;
        boolean isAnyDataPresentOnWorkers = false;

        for (AbstractWorker w : workersManager.getWorkersList()) {
            argumentTransferTime.put(w, 0.0);
        }

        // For each input array, consider how much time it takes to transfer it from every other device;
        for (AbstractArray a : arguments) {
            boolean ctrlNode = false;
            Set<Integer> upToDateLocations = new HashSet<>();
            for (String w : a.getArrayUpToDateLocations()) {
                if (w.equals(Controller.CONTROLLER_NODE)) {
                    upToDateLocations.add(linkBandwidth.length - 1);
                    ctrlNode = true;
                } else {
                    upToDateLocations.add(workersManager.getWorker(w).getId());
                }
            }
            if (upToDateLocations.size() > 1 || (upToDateLocations.size() == 1 && !ctrlNode)) {
                isAnyDataPresentOnWorkers = true;
            }
            // Check all available Workers and compute the tentative transfer time for each of them.
            // to find the device where transfer time is minimum;
            // Add estimated transfer time;
            argumentTransferTime.replaceAll((w, v) -> v + a.getSizeBytes() / computeBandwidth(w.getId(), upToDateLocations));
        }
        return isAnyDataPresentOnWorkers;
    }

    List<AbstractWorker> findWorkersWithEnoughData(HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers, ExecutionDAG.DAGVertex vertex) {
        // List of devices with enough data;
        List<AbstractWorker> workersWithEnoughData = new ArrayList<>();
        // Total size of the input arguments;
        long totalSize = 0;
        for (AbstractArray a : vertex.getComputation().getArrayArguments()) {
            totalSize += a.getSizeBytes();
        }

        // True if at least one device already has at least X% of the data required by the computation;
        for (AbstractWorker w : bytesAlreadyPresentOnWorkers.keySet()) {
            if ((double) bytesAlreadyPresentOnWorkers.get(w) / totalSize > fallbackThreshold) {
                workersWithEnoughData.add(w);
            }
        }
        return workersWithEnoughData;
    }

    private AbstractWorker findWorkerWithLowestTransferTime(List<AbstractWorker> workers, HashMap<AbstractWorker, Double> argumentTransferTime) {
        // The best device is the one with minimum transfer time;
        AbstractWorker deviceWithMinimumTransferTime = workers.get(0);
        for (AbstractWorker w : workers) {
            if (argumentTransferTime.get(w) < argumentTransferTime.get(deviceWithMinimumTransferTime)) {
                deviceWithMinimumTransferTime = w;
            }
        }
        return deviceWithMinimumTransferTime;
    }


    @Override
    public AbstractWorker retrieveClient(ExecutionDAG.DAGVertex vertex) {
        // Map that tracks the size, in bytes, of data that is already present on each worker;
        HashMap<AbstractWorker, Long> bytesAlreadyPresentOnWorkers = new HashMap<>();
        this.workersManager.getWorkersList().forEach(worker -> bytesAlreadyPresentOnWorkers.put(worker, 0L));

        // Compute the amount of data on each worker, and if any worker has any data at all;
        computeDataSizeOnWorkers(vertex, bytesAlreadyPresentOnWorkers);
        LOGGER.info("retrieveClient(" + vertex.getComputation() + ")");

        List<AbstractWorker> workersWithEnoughData = findWorkersWithEnoughData(bytesAlreadyPresentOnWorkers, vertex);

        HashMap<AbstractWorker, Double> argumentTransferTime = new HashMap<>();
        if (computeTransferTimes(vertex, argumentTransferTime)) {
            //argumentTransferTime.forEach((key, value) -> LOGGER.info(key.getNodeIdentifier() + " --> " + value + " units of time"));
            // get minimum transfer time worker
            return findWorkerWithLowestTransferTime(workersWithEnoughData, argumentTransferTime);
        } else {
            // No data is present on any Worker: select the device with round-robin;
            return fallbackPolicy.retrieveClient(vertex);
        }
    }


    @Override
    public void cleanup() {

    }
}
