/*
 * Copyright (c) 2024 NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.necst.controller.runtime.computation;

import com.necst.controller.NoneValue;
import com.necst.controller.runtime.ConfiguredKernel;
import com.necst.controller.runtime.Kernel;
import com.necst.controller.runtime.KernelArguments;
import com.necst.controller.runtime.KernelConfig;
import com.necst.controller.runtime.array.ControllerArray;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class used to track the single execution of a {@link ConfiguredKernel}.
 * The execution will be provided to the {@link ControllerExecutionContext} and
 * scheduled accordingly.
 */
public class KernelExecution extends GrOUTComputationalElement {

    private final Kernel kernel;
    private final ConfiguredKernel configuredKernel;
    private final KernelConfig config;
    private final KernelArguments args;

    public KernelExecution(ConfiguredKernel configuredKernel, KernelArguments args) {
        super(
                configuredKernel.getKernel().getGrOUTExecutionContext(),
                new KernelExecutionInitializer(args));
        this.configuredKernel = configuredKernel;
        this.kernel = configuredKernel.getKernel();
        this.config = configuredKernel.getConfig();
        this.args = args;
    }

    @Override
    public void updateLocationOfArrays() {
        LOGGER.fine("[KERNEL] updateLocationOfArrays()");
        // argumentsThatCanCreateDependencies are just the arrays inside of this.args
        for (ComputationArgumentWithValue o : this.argumentsThatCanCreateDependencies) {
            // Ignore non-array arguments.
            if (o.getArgumentValue() instanceof ControllerArray) {
                ControllerArray a = (ControllerArray) o.getArgumentValue();
                if (!a.isArrayUpdatedIn(this.worker.getNodeIdentifier())) {
                    // If the argument is read-only, add the location of this ComputationalElement
                    // to the array;
                    if (controllerExecutionContext.isConstAware() && o.isConst()) {
                        LOGGER.fine("[KERNEL] Adding WORKER(" + this.worker.getNodeIdentifier()
                                + ") to ControllerArray(" + a.getId() + ") up-to-date locations");
                        a.addArrayUpToDateLocations(this.worker.getNodeIdentifier());
                    } else {
                        // the argument is NOT ready only.
                        // Clear the list of up-to-date locations: only the current device has the
                        // updated array;
                        LOGGER.fine("[KERNEL] ControllerArray(" + a.getId() + ") up-to-date only on WORKER ("
                                + this.worker.getNodeIdentifier() + ")");
                        a.resetArrayUpToDateLocations(this.worker.getNodeIdentifier());
                    }
                } else {
                    if (!o.isConst() && a.getArrayUpToDateLocations().size() > 1) {
                        // the array was present in the worker, but the array is not const, therefore
                        // reset the location of the array
                        // needed in the case of "CLEAN" arrays, that have not yet been modified by
                        // anyone
                        LOGGER.fine("[KERNEL] CLEAN -- ControllerArray(" + a.getId() + ") up-to-date only on WORKER ("
                                + this.worker.getNodeIdentifier() + ")");
                        a.resetArrayUpToDateLocations(this.worker.getNodeIdentifier());
                    }
                }
            }
        }
    }

    @Override
    public Object execute() {
        LOGGER.fine("KernelExeuction.execute() -- " + this);
        // sendRequiredArrays();
        this.worker.executeKernel(kernel.getKernelName(), config.getGridSizeArray(), config.getBlockSizeArray(),
                args.getTypes(), args.getData());
        return NoneValue.get();
    }

    /*
     * private void sendRequiredArrays(){
     * LOGGER.info("KernelExeuction.execute().sendRequiredArrays() -- " + this);
     * for (int i = 0; i < args.getData().length; i++) {
     * LOGGER.info("args.getData("+i+")");
     * if (args.getTypes()[i].equals("pointer")) {
     * LOGGER.info("IS A POINTER");
     * ControllerArray array = (ControllerArray) args.getOriginalArg(i);
     * LOGGER.info(Arrays.toString(array.arrayUpToDateLocations.toArray()));
     * if (!array.isArrayUpdatedIn(this.worker.getNodeIdentifier())) {
     * LOGGER.info("SENDING ARRAY TO THE WORKER");
     * this.worker.sendControllerArray(array.getLittleEndianBigByteBufferView().
     * getBigByteBuffer(), array.getId(), array.getSizeBytes());
     * }else{
     * LOGGER.info("ARRAY IS ALREADY UPDATED IN THE WORKER");
     * }
     * }
     * }
     * }
     */

    public KernelArguments getArgs() {
        return args;
    }

    @Override
    public boolean canBeDistributed() {
        return true;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder
                .append(kernel.getKernelName())
                .append("<<<")
                .append(Arrays.toString(this.config.getGridSizeArray()))
                .append(", ")
                .append(Arrays.toString(this.config.getBlockSizeArray()))
                .append(">>>")
                .append("(");
        for (int i = 0; i < args.getOriginalArgs().length; i++) {
            if (args.getTypes()[i].equals("constant")) {
                stringBuilder.append(args.getOriginalArgs()[i]);
            } else {
                ControllerArray arr = (ControllerArray) args.getOriginalArgs()[i];
                stringBuilder.append("arr(").append(arr.getId()).append(")");
            }
            if (i != args.getOriginalArgs().length - 1)
                stringBuilder.append(", ");
        }

        stringBuilder.append(")");
        return stringBuilder.toString();
    }

    static class KernelExecutionInitializer implements InitializeDependencyList {
        private final KernelArguments args;

        KernelExecutionInitializer(KernelArguments args) {
            this.args = args;
        }

        @Override
        public List<ComputationArgumentWithValue> initialize() {
            // TODO: what about scalars? We cannot treat them in the same way, as they are
            // copied and not referenced
            // There should be a semantic to manually specify scalar dependencies? For now
            // we have to skip them;
            return this.args.getKernelArgumentWithValues().stream()
                    .filter(ComputationArgument::isArray).collect(Collectors.toList());
        }
    }

    public Kernel getKernel() {
        return this.kernel;
    }

    public KernelConfig getConfig() {
        return this.config;
    }
}
