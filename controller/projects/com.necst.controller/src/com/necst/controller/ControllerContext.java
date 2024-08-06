/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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

import com.necst.controller.functions.BuildKernelFunction;
import com.necst.controller.functions.DeviceArrayFunction;
import com.necst.controller.functions.DistributeArrayFunction;
import com.necst.controller.runtime.executioncontext.ControllerExecutionContext;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.TruffleLanguage.Env;
import com.oracle.truffle.api.TruffleLogger;

import java.net.MalformedURLException;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public final class ControllerContext {
    private static final String ROOT_NAMESPACE = "CU";

    private static final TruffleLogger LOGGER = ControllerLogger.getLogger(ControllerLogger.CONTROLLER_LOGGER);

    private final ControllerOptionMap controllerOptionMap;

    private final Env env;
    private final ControllerExecutionContext controllerExecutionContext;
    private final Namespace rootNamespace;
    private final ArrayList<Runnable> disposables = new ArrayList<>();
    private final AtomicInteger moduleId = new AtomicInteger(0);

    // this is used to look up pre-existing call targets for "map" operations, see
    // MapArrayNode
    private final ConcurrentHashMap<Class<?>, CallTarget> uncachedMapCallTargets = new ConcurrentHashMap<>();

    public ControllerContext(Env env) throws MalformedURLException, NotBoundException, RemoteException {
        this.env = env;

        this.controllerOptionMap = new ControllerOptionMap(env.getOptions());
        this.controllerExecutionContext = new ControllerExecutionContext(this.controllerOptionMap);

        Namespace namespace = new Namespace(ROOT_NAMESPACE);
        namespace.addNamespace(namespace);
        namespace.addFunction(new DeviceArrayFunction(this.controllerExecutionContext));
        namespace.addFunction(new BuildKernelFunction(this.controllerExecutionContext));
        namespace.addFunction(new DistributeArrayFunction(this.controllerExecutionContext));

        this.rootNamespace = namespace;
    }

    public Env getEnv() {
        return env;
    }

    public ControllerExecutionContext getControllerExecutionContext() {
        return controllerExecutionContext;
    }

    public Namespace getRootNamespace() {
        return rootNamespace;
    }

    public void addDisposable(Runnable disposable) {
        disposables.add(disposable);
    }

    public void disposeAll() {
        for (Runnable runnable : disposables) {
            runnable.run();
        }
    }

    public int getNextModuleId() {
        return moduleId.incrementAndGet();
    }

    public ConcurrentHashMap<Class<?>, CallTarget> getMapCallTargets() {
        return uncachedMapCallTargets;
    }

    /**
     * Compute the maximum number of concurrent threads that can be spawned by
     * GrOUT.
     * This value is usually smaller or equal than the number of logical CPU threads
     * available on the machine.
     *
     * @return the maximum number of concurrent threads that can be spawned by GrOUT
     */
    public int getNumberOfThreads() {
        return Runtime.getRuntime().availableProcessors();
    }

    public void cleanup() {
        this.controllerExecutionContext.cleanup();
    }

    public ControllerOptionMap getControllerOptions() {
        return this.controllerOptionMap;
    }
}
