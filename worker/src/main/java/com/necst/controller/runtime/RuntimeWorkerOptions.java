package com.necst.controller.runtime;

import java.io.Serializable;
import java.util.ArrayList;

public class RuntimeWorkerOptions implements Serializable {
    // SYNC ASYNC
    public String executionPolicy = "async";
    // True False
    public Boolean inputPrefetch = false;
    // reuse always-new
    public String retrieveNewStreamPolicy = "reuse";
    // same-as-parent disjoint multigpu-early-disjoint multigpu-disjoint
    public String retrieveParentStreamPolicy = "multigpu-disjoint";
    // no-const with-const
    public String dependencyPolicy = "with-const";
    // single-gpu round-robin stream-aware min-transfer-size minmin-transfer-time minmax-transfer-time
    public String deviceSelectionPolicy = "minmax-transfer-time";
    // True False
    public Boolean forceStreamAttach = false;
    // True False
    public Boolean enableComputationTimers = false;
    // integer
    public Integer numGpus = 2;
    // read-mostly preferred none
    public String memAdvisePolicy = "none";
    // path of csv file
    public String bandwidthMatrix = "/home/ubuntu/grcuda/projects/resources/connection_graph/datasets/connection_graph.csv";

    public ArrayList<String> workers;

    public Boolean enableWorkerTimers = false;

    public RuntimeWorkerOptions() {
    }
}