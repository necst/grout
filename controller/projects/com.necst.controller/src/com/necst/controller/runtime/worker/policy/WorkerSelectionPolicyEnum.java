package com.necst.controller.runtime.worker.policy;

public enum WorkerSelectionPolicyEnum {
    ROUND_ROBIN("round-robin"),
    K_STEPS("k-steps"),
    MIN_TRANSFER_SIZE("min-transfer-size"),
    VECTOR_STEP("vector-step"),
    MIN_TRANSFER_TIME("min-transfer-time");

    private final String name;

    WorkerSelectionPolicyEnum(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}
