package com.necst.controller.utils;

import com.necst.controller.NoneValue;
import com.necst.controller.runtime.computation.*;
import com.necst.controller.runtime.executioncontext.AbstractControllerExecutionContext;

import java.util.List;
import java.util.stream.Collectors;

public class KernelExecutionMock extends GrOUTComputationalElement {

    /**
     * Simulate an execution by forcing a wait that last the given number of
     * milliseconds;
     */
    private final int durationMs;

    private final String name;

    public KernelExecutionMock(AbstractControllerExecutionContext grOUTExecutionContext,
            List<ComputationArgumentWithValue> args) {
        this(grOUTExecutionContext, args, "kernel");
    }

    public KernelExecutionMock(AbstractControllerExecutionContext grOUTExecutionContext,
            List<ComputationArgumentWithValue> args, String name) {
        this(grOUTExecutionContext, args, name, 0);
    }

    public KernelExecutionMock(AbstractControllerExecutionContext grOUTExecutionContext,
            List<ComputationArgumentWithValue> args, String name, int durationMs) {
        super(grOUTExecutionContext, args);
        this.name = name;
        this.durationMs = durationMs;
    }

    public String getName() {
        return name;
    }

    @Override
    public Object execute() {
        if (this.durationMs > 0) {
            try {
                Thread.sleep(this.durationMs);
            } catch (InterruptedException e) {
                System.out.println("ERROR; failed to pause " + this + " for " + this.durationMs + " msec");
                e.printStackTrace();
            }
        }
        return NoneValue.get();
    }

    @Override
    public void updateLocationOfArrays() {

    }

    @Override
    public String toString() {
        return this.getName() + ": args={" +
                this.argumentsThatCanCreateDependencies.stream().map(Object::toString).collect(Collectors.joining(", "))
                +
                "}" + ";";
    }
}