package com.necst.controller.runtime.worker.options;

public enum MemAdviserEnum {
    ADVISE_READ_MOSTLY("read-mostly"),
    ADVISE_PREFERRED_LOCATION("preferred"),
    NONE("none");

    private final String name;

    MemAdviserEnum(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}