package com.necst.controller;

import com.oracle.truffle.api.TruffleLogger;

public class ControllerLogger {

    public static final String DEFAULT_LOGGER_LEVEL= "INFO";

    public static final String CONTROLLER_LOGGER = "com.necst.controller";

    public static final String WORKER_SELECTION_POLICY_LOGGER = "com.necst.controller.worker.policy";

    public static final String WORKERSMANAGER_LOGGER = "com.necst.controller.runtime.worker";

    public static final String FUNCTIONS_LOGGER = "com.necst.controller.functions";

    public static final String NODES_LOGGER = "com.necst.controller.nodes";

    public static final String PARSER_LOGGER = "com.necst.controller.parser";

    public static final String RUNTIME_LOGGER = "com.necst.controller.runtime";

    public static final String ARRAY_LOGGER = "com.necst.controller.runtime.array";

    public static final String COMPUTATION_LOGGER = "com.necst.controller.runtime.computation";

    public static final String EXECUTIONCONTEXT_LOGGER = "com.necst.controller.runtime.executioncontext";

    public static final String STREAM_LOGGER = "com.necst.controller.runtime.stream";

    public static TruffleLogger getLogger(String name) {
        return TruffleLogger.getLogger(ControllerLanguage.ID, name);
    }
}
