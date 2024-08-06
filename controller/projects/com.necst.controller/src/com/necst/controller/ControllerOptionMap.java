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
package com.necst.controller;

import com.necst.controller.runtime.computation.dependency.DependencyPolicyEnum;
import com.necst.controller.runtime.worker.policy.WorkerSelectionPolicyEnum;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.StopIterationException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownKeyException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;

import java.io.File;
import java.util.*;

@ExportLibrary(InteropLibrary.class)
public class ControllerOptionMap implements TruffleObject {

    /**
     * Store options using the option name and its value;
     */
    private final HashMap<String, Object> optionsMap;
    /**
     * Store a mapping between GrOUT's Truffle options and their names, as strings.
     * OptionKeys are assumed to be immutable, so this map must be read-only as
     * well;
     */
    private final HashMap<OptionKey<?>, String> optionNames;

    private static final TruffleLogger LOGGER = TruffleLogger.getLogger(ControllerLanguage.ID,
            ControllerLogger.CONTROLLER_LOGGER);

    public static final WorkerSelectionPolicyEnum DEFAULT_WORKER_SELECTION_POLICY = WorkerSelectionPolicyEnum.ROUND_ROBIN;
    public static final DependencyPolicyEnum DEFAULT_DEPENDENCY_POLICY = DependencyPolicyEnum.WITH_CONST;
    public static final String DEFAULT_WORKERS_INFO = "";
    public static final boolean DEFAULT_MOCK_WORKER = false;
    public static final boolean DEFAULT_ENABLE_TIMERS = false;
    public static final String DEFUALT_VECTOR_STEP = "[1, 2, 3]";
    public static final String DEFAULT_LINK_BANDWIDTH_PATH = System.getenv("GRCUDA_DIST_HOME") + File.separator
            + "link_bandwidth.csv";

    public ControllerOptionMap(OptionValues options) {
        optionsMap = new HashMap<>();
        optionNames = new HashMap<>();

        // Store the name and value of each option;
        // Map each OptionKey to its name, to retrieve values inside GrOUT;
        options.getDescriptors().forEach(o -> {
            optionsMap.put(o.getName(), options.get(o.getKey()));
            optionNames.put(o.getKey(), o.getName());
        });

        // Parse individual options;

        // --> Worker selection policy;
        optionsMap.replace(optionNames.get(ControllerOptions.WorkerSelectionPolicy),
                parseWorkerSelectionPolicy(options.get(ControllerOptions.WorkerSelectionPolicy)));

        // --> Dependency policy;
        optionsMap.replace(optionNames.get(ControllerOptions.DependencyPolicy),
                parseDependecyPolicy(options.get(ControllerOptions.DependencyPolicy)));

        // --> Workers network info
        optionsMap.replace(optionNames.get(ControllerOptions.WorkersNetInfo),
                parseWorkersNetInfo(options.get(ControllerOptions.WorkersNetInfo)));

        optionsMap.replace(optionNames.get(ControllerOptions.VectorStep),
                parseVectorStep(options.get(ControllerOptions.VectorStep)));

        // optionsMap.replace(optionNames.get(ControllerOptions.EnableMockWorker),
        // parseMockWorker(options.get(ControllerOptions.EnableMockWorker)));

        // optionsMap.replace(optionNames.get(ControllerOptions.EnableTimers),
        // parseEnableTimers(options.get(ControllerOptions.EnableTimers)));

    }

    public boolean parseEnableTimers(String enableTimers) {
        return (boolean) Boolean.parseBoolean(enableTimers);
    }

    public boolean parseMockWorker(String mockWorker) {
        return (boolean) Boolean.parseBoolean(mockWorker);
    }

    private static ArrayList<String> parseWorkersNetInfo(String workersNetInfo) {
        return new ArrayList<>(Arrays.asList(workersNetInfo.replaceAll(" ", "").split(",")));
    }

    private static int[] parseVectorStep(String vectorStep) {
        String[] parts = vectorStep.substring(1, vectorStep.length() - 1).split(", ");
        int[] result = new int[parts.length];

        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }

        return result;
    }

    public int[] getVectorStep() {
        return (int[]) getOptionValueFromOptionKey(ControllerOptions.VectorStep);
    }

    public ArrayList<String> getWorkersInfo() {
        return (ArrayList<String>) getOptionValueFromOptionKey(ControllerOptions.WorkersNetInfo);
    }

    /**
     * Obtain the option value starting from the OptionKey;
     */
    private Object getOptionValueFromOptionKey(OptionKey<?> optionKey) {
        return optionsMap.get(optionNames.get(optionKey));
    }

    // Enforces immutability;
    public HashMap<String, Object> getOptions() {
        return new HashMap<>(optionsMap);
    }

    private static WorkerSelectionPolicyEnum parseWorkerSelectionPolicy(String policyString) {
        if (Objects.equals(policyString, WorkerSelectionPolicyEnum.ROUND_ROBIN.toString()))
            return WorkerSelectionPolicyEnum.ROUND_ROBIN;
        else if (Objects.equals(policyString, WorkerSelectionPolicyEnum.MIN_TRANSFER_SIZE.toString()))
            return WorkerSelectionPolicyEnum.MIN_TRANSFER_SIZE;
        else if (Objects.equals(policyString, WorkerSelectionPolicyEnum.VECTOR_STEP.toString()))
            return WorkerSelectionPolicyEnum.VECTOR_STEP;
        else if (Objects.equals(policyString, WorkerSelectionPolicyEnum.K_STEPS.toString()))
            return WorkerSelectionPolicyEnum.K_STEPS;
        else if (Objects.equals(policyString, WorkerSelectionPolicyEnum.MIN_TRANSFER_TIME.toString()))
            return WorkerSelectionPolicyEnum.MIN_TRANSFER_TIME;
        else {
            LOGGER.warning("Warning: unknown device selection policy=" + policyString + "; using default="
                    + DEFAULT_WORKER_SELECTION_POLICY);
            return DEFAULT_WORKER_SELECTION_POLICY;
        }
    }

    public WorkerSelectionPolicyEnum getWorkerSelectionPolicy() {
        return (WorkerSelectionPolicyEnum) getOptionValueFromOptionKey(ControllerOptions.WorkerSelectionPolicy);
    }

    public boolean getEnableMockWorker() {
        return (boolean) getOptionValueFromOptionKey(ControllerOptions.EnableMockWorker);
    }

    public boolean getEnableTimers() {
        return (boolean) getOptionValueFromOptionKey(ControllerOptions.EnableTimers);
    }

    private static DependencyPolicyEnum parseDependecyPolicy(String policyString) {
        if (Objects.equals(policyString, DependencyPolicyEnum.NO_CONST.toString()))
            return DependencyPolicyEnum.NO_CONST;
        else if (Objects.equals(policyString, DependencyPolicyEnum.WITH_CONST.toString()))
            return DependencyPolicyEnum.WITH_CONST;
        else {
            LOGGER.warning("Warning: unknown dependency policy=" + policyString + "; using default="
                    + DEFAULT_DEPENDENCY_POLICY);
            return DEFAULT_DEPENDENCY_POLICY;
        }
    }

    public DependencyPolicyEnum getDependencyPolicy() {
        return (DependencyPolicyEnum) getOptionValueFromOptionKey(ControllerOptions.DependencyPolicy);
    }

    public String getLinkBandwidthPath() {
        return (String) getOptionValueFromOptionKey(ControllerOptions.LinkBandwidthPath);
    }

    // Implement InteropLibrary;

    @ExportMessage
    public final boolean hasHashEntries() {
        return true;
    }

    @ExportMessage
    public final Object readHashValue(Object key) throws UnknownKeyException, UnsupportedMessageException {
        Object value;
        if (key instanceof String) {
            value = this.optionsMap.get(key);
        } else {
            throw UnsupportedMessageException.create();
        }
        if (value == null)
            throw UnknownKeyException.create(key);
        return value.toString();
    }

    @ExportMessage
    public final long getHashSize() {
        return optionsMap.size();
    }

    @ExportMessage
    public final boolean isHashEntryReadable(Object key) {
        return key instanceof String && this.optionsMap.containsKey(key);
    }

    @ExportMessage
    public Object getHashEntriesIterator() {
        return new EntriesIterator(optionsMap.entrySet().iterator());
    }

    @ExportLibrary(InteropLibrary.class)
    public static final class EntriesIterator implements TruffleObject {
        private final Iterator<Map.Entry<String, Object>> iterator;

        private EntriesIterator(Iterator<Map.Entry<String, Object>> iterator) {
            this.iterator = iterator;
        }

        @SuppressWarnings("static-method")
        @ExportMessage
        public boolean isIterator() {
            return true;
        }

        @ExportMessage
        public boolean hasIteratorNextElement() {
            try {
                return iterator.hasNext();
            } catch (NoSuchElementException e) {
                return false;
            }
        }

        @ExportMessage
        public GrOUTOptionTuple getIteratorNextElement() throws StopIterationException {
            if (hasIteratorNextElement()) {
                Map.Entry<String, Object> entry = iterator.next();
                return new GrOUTOptionTuple(entry.getKey(), entry.getValue().toString());
            } else {
                throw StopIterationException.create();
            }
        }
    }

    @ExportLibrary(InteropLibrary.class)
    public static class GrOUTOptionTuple implements TruffleObject {

        private final int SIZE = 2;
        private final String[] entry = new String[SIZE];

        public GrOUTOptionTuple(String key, String value) {
            entry[0] = key;
            entry[1] = value;
        }

        @ExportMessage
        static boolean hasArrayElements(GrOUTOptionTuple tuple) {
            return true;
        }

        @ExportMessage
        public final boolean isArrayElementReadable(long index) {
            return index == 0 || index == 1;
        }

        @ExportMessage
        public final Object readArrayElement(long index) throws InvalidArrayIndexException {
            if (index == 0 || index == 1) {
                return entry[(int) index];
            } else {
                throw InvalidArrayIndexException.create(index);
            }
        }

        @ExportMessage
        public final long getArraySize() {
            return SIZE;
        }
    }

}
