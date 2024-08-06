package com.necst.controller.runtime;

import java.util.ArrayList;

public class WorkerStatistics {
    public String workerID;

    public long dataSentToController = 0;
    public long dataReceivedFromController = 0;

    public ArrayList<Long> initContextTime = new ArrayList<>();

    public ArrayList<Long> getArrayTime = new ArrayList<>();
    public ArrayList<Long> getArraySize = new ArrayList<>();

    public ArrayList<Long> sendArrayTime = new ArrayList<>();
    public ArrayList<Long> sendArraySize = new ArrayList<>();

    public ArrayList<Long> p2pTime = new ArrayList<>();
    public ArrayList<Long> p2pSize = new ArrayList<>();

    public ArrayList<Long> buildKernelTime = new ArrayList<>();
    public ArrayList<Long> invocationKernelTime = new ArrayList<>();
    public ArrayList<Long> closeContextTime = new ArrayList<>();
}
