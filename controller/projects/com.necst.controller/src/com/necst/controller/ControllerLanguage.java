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
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.necst.controller;


import com.necst.controller.nodes.ExpressionNode;
import com.necst.controller.nodes.ControllerRootNode;
import com.necst.controller.parser.ParserAntlr;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.options.OptionDescriptors;

import java.net.MalformedURLException;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.util.logging.Logger;


/**
 * Controller Truffle language that provides autonomous distribution of CUDA kernels to all the Graal languages.
 */
@TruffleLanguage.Registration(id = ControllerLanguage.ID, name = "controller", version = "0.1", internal = false, contextPolicy = TruffleLanguage.ContextPolicy.SHARED)
public final class ControllerLanguage extends TruffleLanguage<ControllerContext> {

    public static final String ID = "controller";

    public static final TruffleLogger LOGGER = TruffleLogger.getLogger(ID, "com.necst.controller");

    @Override
    protected ControllerContext createContext(Env env) {
        if (!env.isNativeAccessAllowed()) {
            throw new ControllerException("cannot create CUDA context without native access");
        }
        try {
            return new ControllerContext(env);
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        } catch (NotBoundException e) {
            throw new RuntimeException(e);
        } catch (RemoteException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected CallTarget parse(ParsingRequest request) {
        ExpressionNode expression = new ParserAntlr().parse(request.getSource());
        ControllerRootNode newParserRoot = new ControllerRootNode(this, expression);
        return Truffle.getRuntime().createCallTarget(newParserRoot);
    }

    public static ControllerLanguage getCurrentLanguage() {
        return TruffleLanguage.getCurrentLanguage(ControllerLanguage.class);
    }

    public static ControllerContext getCurrentContext() {
        return getCurrentContext(ControllerLanguage.class);
    }

    @Override
    protected void disposeContext(ControllerContext cxt) {
        cxt.disposeAll();
    }

    @Override
    public OptionDescriptors getOptionDescriptors() {
        return ControllerLanguage.getOptionDescriptorsStatic();
    }

    /**
     * We make the list of option descriptors available statically, so it can be used when mocking the language, without having to create a context;
     * @return the list of option descriptors, with default values available;
     */
    public static OptionDescriptors getOptionDescriptorsStatic() {
        return new ControllerOptionsOptionDescriptors();
    }

    @Override
    protected boolean isThreadAccessAllowed(Thread thread, boolean singleThreaded) {
        return true;
    }

    @Override
    protected void finalizeContext(ControllerContext context) {
        context.cleanup();
    }
}
