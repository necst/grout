package com.necst.controller.runtime;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class Timings implements Serializable {
    public List<Long> initContextTime = new ArrayList<>();
    public List<Long> getArrayTime = new ArrayList<>();
    public List<Integer> getArraySize = new ArrayList<>();
    public List<Long> sendArrayTime = new ArrayList<>();
    public List<Integer> sendArraySize = new ArrayList<>();
    public List<Long> buildKernelTime = new ArrayList<>();
    public List<Long> invocationKernelTime = new ArrayList<>();
    public List<Long> closeContextTime = new ArrayList<>();

    public long getTotalInitContextTime() {
        long totalTime = 0;
        for (long time : initContextTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public long getTotalGetArrayTime() {
        long totalTime = 0;
        for (long time : getArrayTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public long getTotalSendArrayTime() {
        long totalTime = 0;
        for (long time : sendArrayTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public long getTotalBuildKernelTime() {
        long totalTime = 0;
        for (long time : buildKernelTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public long getTotalInvocationKernelTime() {
        long totalTime = 0;
        for (long time : invocationKernelTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public long getTotalCloseContextTime() {
        long totalTime = 0;
        for (long time : closeContextTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public long getTotalTime() {
        long totalTime = 0;
        for (long time : initContextTime) {
            totalTime += time;
        }
        for (long time : getArrayTime) {
            totalTime += time;
        }
        for (long time : sendArrayTime) {
            totalTime += time;
        }
        for (long time : buildKernelTime) {
            totalTime += time;
        }
        for (long time : invocationKernelTime) {
            totalTime += time;
        }
        for (long time : closeContextTime) {
            totalTime += time;
        }
        return totalTime;
    }

    public void toJson(String path, String fileName){
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        try (FileWriter writer = new FileWriter(path+fileName)) {
            gson.toJson(this, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String toString() {
        return "Timings (" + getTotalTime() / 1000000000.0 +
                " sec) {\n\tinitContextTime=" + getTotalInitContextTime() / 1000000000.0 +
                " sec,\n\tgetArrayTime=" + getTotalGetArrayTime() / 1000000000.0 +
                " sec,\n\tsendArrayTime=" + getTotalSendArrayTime() / 1000000000.0 +
                " sec,\n\tbuildKernelTime=" + getTotalBuildKernelTime() / 1000000000.0 +
                " sec,\n\tinvocationKernelTime=" + getTotalInvocationKernelTime() / 1000000000.0 +
                " sec,\n\tcloseContextTime=" + getTotalCloseContextTime() / 1000000000.0 +
                " sec\n}";
    }
}
