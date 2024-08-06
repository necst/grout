# GrOUT


## Using GrOUT in GraalVM

GrOUT can be used in the binaries of the GraalVM languages (`lli`, `graalpython`, `js`, `R`, and `ruby`).
The JAR file containing GrOUT must be appended to the classpath or copied into `jre/languages/grout` (Java 8) or `languages/grout` (Java 11) of the GraalVM installation. 
Note that `--jvm` and `--polyglot` must be specified in both cases as well.

The following example shows how to create a GPU kernel and two device arrays in JavaScript (NodeJS) and invoke the kernel:

```JavaScript
// build kernel from CUDA C/C++ source code
const kernelSource = `
__global__ void increment(int *arr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] += 1;
  }
}`
const cu = Polyglot.eval('controller', 'CU') // get GrOUT namespace object
const incKernel = cu.buildkernel(
  kernelSource, // CUDA kernel source code string
  'increment', // kernel name
  'pointer, sint32') // kernel signature

// allocate device array
const numElements = 100
const deviceArray = cu.DeviceArray('int', numElements)
for (let i = 0; i < numElements; i++) {
  deviceArray[i] = i // ... and initialize on the host
}
// launch kernel in grid of 1 block with 128 threads
incKernel(1, 128)(deviceArray, numElements)

// print elements from updated array
for (const element of deviceArray) {
  console.log(element)
}
```

```console
$GRAALVM_DIR/bin/node --polyglot --jvm example.js
1
2
...
100
```

### Additional examples

Additional examples (e.g., starting pre-compiled kernels, Python code and many others) can be found in the [GrCUDA](https://github.com/necst/grcuda) project repository.

The APIs of GrOUT are cross-compatible with the ones of GrCUDA, change `grcuda` to `controller` to utilize the same existing code.


## Installation

To use GrOUT, you need to obtain the Worker and Controller JAR.

### Installation from source files

1. First, download GraalVM 22.1 as above.

  ```console
  wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.1.0/graalvm-ce-java11-linux-amd64-22.1.0.tar.gz
  tar xfz graalvm-ce-java11-linux-amd64-22.1.0.tar.gz
  rm graalvm-ce-java11-linux-amd64-22.1.0.tar.gz
  export GRAALVM_DIR=~/graalvm-ce-java11-22.1.0
  ```

2. To build GrOUT, you also need a custom JDK that is used to build GraalVM.

  ```console
  wget https://github.com/graalvm/labs-openjdk-11/releases/download/jvmci-22.1-b01/labsjdk-ce-11.0.15+2-jvmci-22.1-b01-linux-amd64.tar.gz
  tar xfz labsjdk-ce-11.0.15+2-jvmci-22.1-b01-linux-amd64.tar.gz
  rm labsjdk-ce-11.0.15+2-jvmci-22.1-b01-linux-amd64.tar.gz
  export JAVA_HOME=~/labsjdk-ce-11.0.15-jvmci-22.1-b01
  ```
  
3. GrOUT requires the [mx build tool](https://github.com/graalvm/mx).
Clone the mx repository and add the directory into `$PATH`, such that the `mx` can be invoked from
the command line.
We checkout the commit corresponding to the current GraalVM release.

  ```console
  git clone https://github.com/graalvm/mx.git
  cd mx
  git checkout 722b86b8ef87fbb297f7e33ee6014bbbd3f4a3a8
  cd ..
  ```

5. Last but not least, build GrOUT

  ```console
  cd <directory containing this README>
  ./controller.sh
  ./worker.sh
  ```

## Run

1. Start the workers on the desired GPU nodes
  ```bash
java -cp ./target/worker-1.0-SNAPSHOT.jar com.necst.controller.runtime.Worker $PORT
  ```

2. Set their IP addresses in the host Context:
  ```java
context = Context.newBuilder()
  .allowAllAccess(true)
  .allowExperimentalOptions(true)
  .option("controller.WorkersNetInfo", "IP_1:$PORT, IP_2:$PORT")
  .build();
  ```
3. Start the host code



## Publications

If you use GrOUT in your research, please cite the following publication(s). Thank you!

```
@INPROCEEDINGS{10596350,
  author={Dio Lavore, Ian Di and Maffi, Davide and Arnaboldi, Marco and Delamare, Arnaud and Bonetta, Daniele and Santambrogio, Marco D.},
  booktitle={2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)}, 
  title={GrOUT: Transparent Scale-Out to Overcome UVM's Oversubscription Slowdowns}, 
  year={2024},
  pages={696-705},
  doi={10.1109/IPDPSW63119.2024.00132}}
```
