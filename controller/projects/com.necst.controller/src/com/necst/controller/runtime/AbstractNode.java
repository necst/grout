package com.necst.controller.runtime;

import java.util.Objects;

/**
 * Abstract device representation, used to distinguish between CPU and GPU devices inside the GrOUT scheduler.
 */
public abstract class AbstractNode {
    protected final String nodeIdentifier;

    public AbstractNode(String nodeIdentifier) {
        this.nodeIdentifier = nodeIdentifier;
    }

    public String getNodeIdentifier() {
        return nodeIdentifier;
    }



    @Override
    public String toString() {
        return "Device(id=" + nodeIdentifier + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AbstractNode that = (AbstractNode) o;
        return Objects.equals(nodeIdentifier, that.nodeIdentifier);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nodeIdentifier);
    }
}
