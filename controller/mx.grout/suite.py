# Copyright (c) 2024, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

suite = {
    # --------------------------------------------------------------------------------------------------------------
    #
    #  METADATA
    #
    # --------------------------------------------------------------------------------------------------------------
    "mxversion": "5.190.1",
    "name": "controller",
    "versionConflictResolution": "latest",

    "version": "1.0.0",
    "release": False,
    "groupId": "com.necst.controller",

    "developer": {
        "name": "GrOUT Developers",
        "organization": "GrOUT Developers",
    },


    # --------------------------------------------------------------------------------------------------------------
    #
    #  DEPENDENCIES
    #
    # --------------------------------------------------------------------------------------------------------------
    "imports": {
        "suites": [
            {
                "name": "truffle",
                "version": "84541b16ae8a8726a0e7d76c7179d94a57ed84ee",
                "subdir": True,
                "urls": [
                    {"url": "https://github.com/oracle/graal", "kind": "git"},
                ]
            },
        ],
    },

    # --------------------------------------------------------------------------------------------------------------
    #
    #  REPOS
    #
    # --------------------------------------------------------------------------------------------------------------
    "repositories": {
    },

    "defaultLicense": "BSD-3",

    # --------------------------------------------------------------------------------------------------------------
    #
    #  LIBRARIES
    #
    # --------------------------------------------------------------------------------------------------------------
    "libraries": {
        "gson" : {
            "maven" : {
                "groupId" : "com.google.code.gson",
                "artifactId" : "gson",
                "version" : "2.10.1",
            },
            "sha1": "b3add478d4382b78ea20b1671390a858002feb6c",
            "license" : ["Apache 2.0"]
        },
    },

    # --------------------------------------------------------------------------------------------------------------
    #
    #  PROJECTS
    #
    # --------------------------------------------------------------------------------------------------------------
    "externalProjects": {
    },


    "projects": {
        "com.necst.controller.parser.antlr": {
            "subDir": "projects",
            "buildEnv": {
                "ANTLR_JAR": "<path:truffle:ANTLR4_COMPLETE>",
                "PARSER_PATH": "<src_dir:com.necst.controller>/com/necst/controller/parser/antlr",
                "OUTPUT_PATH": "<src_dir:com.necst.controller>/com/necst/controller/parser/antlr",
                "PARSER_PKG": "com.necst.controller.parser.antlr",
                "POSTPROCESSOR": "<src_dir:com.necst.controller.parser.antlr>/postprocessor.py",
            },
            "dependencies": [
                "truffle:ANTLR4_COMPLETE",
            ],
            "native": True,
            "vpath": True,
        },
        "com.necst.controller": {
            "subDir": "projects",
            "license": ["BSD-3"],
            "sourceDirs": ["src"],
            "javaCompliance": "8+",
            "annotationProcessors": ["truffle:TRUFFLE_DSL_PROCESSOR"],
            "dependencies": [
                "truffle:TRUFFLE_API",
                "sdk:GRAAL_SDK",
                "truffle:ANTLR4",
                "gson",
            ],
            "buildDependencies": ["com.necst.controller.parser.antlr"],
            "checkstyleVersion": "8.8",
        },
        "com.necst.controller.test": {
            "subDir": "projects",
            "sourceDirs": ["src"],
            "dependencies": [
                "com.necst.controller",
                "mx:JUNIT",
                "truffle:TRUFFLE_TEST",
                "gson",
            ],
            "checkstyle": "com.necst.controller",
            "javaCompliance": "8+",
            "annotationProcessors": ["truffle:TRUFFLE_DSL_PROCESSOR"],
            "workingSets": "Truffle,CUDA",
            "testProject": True,
        },
    },

    "licenses": {
        "BSD-3": {
            "name": "3-Clause BSD License",
            "url": "http://opensource.org/licenses/BSD-3-Clause",
        },
        "Apache 2.0": {
            "name": "Apache 2.0 License",
            "url": "http://www.apache.org/licenses/LICENSE-2.0.txt",
        },
    },

    # --------------------------------------------------------------------------------------------------------------
    #
    #  DISTRIBUTIONS
    #
    # --------------------------------------------------------------------------------------------------------------
    "distributions": {
        "CONTROLLER": {
            "dependencies": [
                "com.necst.controller",
            ],
            "distDependencies": [
                "truffle:TRUFFLE_API",
                "sdk:GRAAL_SDK",
            ],
            "sourcesPath": "controller.src.zip",
            "description": "Controller",
            "javaCompliance": "8+",
        },

        "CONTROLLER_UNIT_TESTS": {
            "description": "Controller unit tests",
            "dependencies": [
                "com.necst.controller.test",
            ],
            "exclude": ["mx:JUNIT"],
            "distDependencies": [
                "CONTROLLER",
                "truffle:TRUFFLE_TEST"
            ],
            "sourcesPath": "controller.tests.src.zip",
            "testDistribution": True,
            "javaCompliance": "8+",
        },
    },
}
