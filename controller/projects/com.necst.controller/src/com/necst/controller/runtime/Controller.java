package com.necst.controller.runtime;

public class Controller extends AbstractNode {
    public static final String CONTROLLER_NODE = "CONTROLLER";

    public Controller() {
        super(CONTROLLER_NODE);
    }

    @Override
    public String toString() {
        return "CPU(id=" + nodeIdentifier + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Controller that = (Controller) o;
        return nodeIdentifier == that.nodeIdentifier;
    }
}
