package com.necst.controller.runtime;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class ControllerStatistics {
    public int numRemoteArraysRetrieval = 0;

    /*
        A double hop happens when we have a kernelExecution that has as input parameters
        an array not up-to-date on the worker where it is scheduled to be executed.
        This will trigger a data movement from the worker1-->controller-->worker2.
     */
    public int numDoubleHops = 0;
    public long doubleHopsSize = 0;
    public int numP2Ptransfers = 0;

    // NOTE: "-1" means that it was an operation done on the controller
    public ArrayList<Long> overheadRetrieveClient = new ArrayList<>();

    /*
        For kernels, this value include all the necessary data-transfers to start the current operation
        including double-hops to the controller. Near zero means that the array is already up-to-date on
        the controller.
     */
    public ArrayList<Long> beforeExecutingKernels = new ArrayList<>();

    /*
        For arrays read/write operations this is the time required to recover (if necessary) the array
        from a remote worker. Near zero means that the required array is already up-to-date on the controller.
     */
    public ArrayList<Long> beforeExecutingArrayOperations = new ArrayList<>();

    public ArrayList<WorkerStatistics> workerStatistics= new ArrayList<>();

    public void toJson(String path, String fileName){
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        try (FileWriter writer = new FileWriter(path+fileName)) {
            gson.toJson(this, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

