kissofdeath (**JAIDE** v40) – Teljes dokumentáció (Magyar)

> Forrás: [https://deepwiki.com/kollarsandor/kissofdeath](https://deepwiki.com/kollarsandor/kissofdeath) > Generálva: **2026**-04-03

---

 Projektáttekintés

A **JAIDE** (v40) egy gyökérszintű nagy nyelvi modell (**LLM**) rendszer, amelyet a **KGRU** (Kalmár László – Gábor Dénes – Riesz Frigyes egység) architekturális filozofia alapján terveztek. Ez egy vertikálisan integrált stack, amely egyedi neurális architektúrát, dedikált relációs motort, hardveres gyorsítási kerneleket és átfogó formális verifikációs csomagot foglal magában.

A projekt elsősorban Zig nyelven íródott (0.13.0 vagy 0.14.1 verzió), a teljesítménykritikus kernelekhez Futhark nyelvet, a formális specifikációkhoz Lean 4, Cryptol és **SAW** eszközöket használ.

 Rendszer céljai és filozofiája

A **JAIDE** célja, hogy túllépjen a standard transformer architektúrákon az alábbi elemek segítségével:

**RSF** (Relational Signal Flow): Egy olyan neurális rétegkészlet, amely a relációs adatmozgást helyezi előtérbe.

**MGT** (Morpheme-Guided Tokenization): Egy tokenizáló, amely morfológiai struktúrát és morfémabontást alkalmaz a szimpla bájt-pár helyett.

**SFD** (Stochastic Fractal Descent): Egyedi optimalizációs stratégia a paraméterfrissítéshez, pillanatbecslés alapján.

Formális rigor: Minden alapkomponenst matematikai bizonyítékok támasztanak alá a magas kockázatú következtetési feladatokban való biztonság és helyesség garantálása érdekében.

A **KGRU** név magyar úttörőknek állít emléket: Kalmár László (logika/verifikáció), Gábor Dénes (információelmélet/jelterjedés) és Riesz Frigyes (funkcionálanalízis/kvantumlogika).

 Magas szintű architektúra

A rendszer három fő síkra oszlik: az ML Pipeline, a Core Relational Engine és az Acceleration Layer.

Az adatáramlás sorban: nyers szöveg → **MGT** tokenizálás → **RSF** rétegek → **SFD** optimalizáló → **SSI** index → **NSIR** relációs gráf → Ranker visszakeresés.

 Könyvtárstruktúra

| Könyvtár | Cél | Kulcsentitások |
|---|---|---|
| src/core/ | Alapprimitívek | Tensor, Memory, IO |
| src/processor/ | Neurális architektúra | RSF, LayerCore, OFTB |
| src/core_relational/ | Következtetési motor | NSIR, ZRuntime, ReasoningOrchestrator |
| src/hw/ | Hardveres gyorsítás | Futhark, RTL (Clash), CUDA |
| src/verification/ | Formális bizonyítékok | Lean 4, Viper, SAW, Agda |
| src/api/ | Telepítés | InferenceServer |

---

 Bevezetés: Build és konfiguráció

 Eszközlánc és követelmények

A **JAIDE** a Zig programozási nyelven épül, és az **LLVM**-et használja formális verifikációhoz és bitkód generáláshoz.

Alapfüggőségek: Zig 0.13.0, **LLVM** 17, Futhark (**GPU** kernelekhez), **SAW** 0.9.0, Cryptol 2.13.0 és Z3 a formális verifikációs pipeline-hoz.

A src/build.sh szkript automatizálja a szükséges eszközláncok telepítését Linux (apt/dnf/pacman) és macOS (brew) környezetekben.

 Build célok

| Cél neve | Bináris neve | Cél |
|---|---|---|
| jaide | jaide | Standard CPU/GPU interaktív és betanítási mód |
| jaide-gpu | jaide-gpu | Dedikált GPU-optimalizált futtatási útvonal |
| jaide-distributed | jaide-distributed | Több csomópontos betanítás koordináció |
| jaide-inference-server | jaide-inference-server | HTTP API modell kiszolgáláshoz |

Build beállítások: **GPU** gyorsítás -Dgpu=true-val, optimalizálási szintek -Doptimize-zal (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall).

 Működési módok és **CLI** zászlók

Betanítási mód (--mode train): --dataset, --epochs, --batch-size, --learning-rate, --dim / --layers.

Következtetési mód (--mode infer): --model, --prompt, --max-tokens.

 Felhőalapú telepítés (Modal)

A **JAIDE** Python szkriptek átfogó készletét tartalmazza a Modal felhőplatformra való telepítéshez, **NVIDIA** **B200** **GPU**-kat célozva. Az egyedi Ubuntu 22.04 kép **CUDA** 12.4-et, Python 3.11-et és Zig 0.13.0 eszközláncot tartalmaz, csomópontonként akár 8 **B200** **GPU**-val és **256** GB **RAM**-mal.

Telepítési munkafolyamat: (1) modal_setup.sh inicializálja a tokent és köteteket, (2) modal_distributed_train.py futtatja a betanítást, (3) modal_inference.py lehetővé teszi a felhőbeli tesztelést.

---

 Rendszerarchitektúra – Áttekintés

A **JAIDE** (v40) egy nagy teljesítményű **LLM** infrastruktúra, amely hagyományos lineáris szekvenciafeldolgozásról fraktális, relációs gráfalapú megközelítésre vált.

 Magas szintű adatáramlás

## Bevitel: A nyers szöveget az MGT (Morfémavezérelt Tokenizáló) dolgozza fel.

## Feldolgozás: A tokenek embeddingekké alakulnak, majd RSF rétegeken haladnak át. ## Optimalizálás: A gradienseket az SFD optimizer kezeli. ## Indexelés: A feldolgozott jeleket az SSI (Tömör Szemantikai Index) tárolja. ## Relációs logika: A CREVPipeline relációs hármasokat nyom ki az NSIR gráfhoz. ## Visszakeresés: A Ranker LSH-alapú pontozással kéri le a legjelentősebb kontextust.

 Adatreprezentáció

A rendszer egyedi fixpontos típusokat használ a determinisztikus futtatáshoz:

| Típus | Bitszélesség | Skálázási tényező | Felhasználás |
|---|---|---|---|
| FixedPoint16 | 16-bit | 256.0 | Alacsony precizitású súlyok |
| FixedPoint32 | 32-bit | 65536.0 | Standard aktivációk |
| Fixed32_32 | 64-bit | 4294967296.0 | Nagy precizitású gradiensek |

 Alrendszerek összefoglalója

| Alrendszer | Kulcsfájl | Szerepkör |
|---|---|---|
| Processzor | src/processor/rsf.zig | Neurális jelkonverzió |
| Tokenizáló | src/tokenizer/mgt.zig | Szöveg ↔ egész szám leképezés |
| Optimalizáló | src/optimizer/sfd.zig | Fraktálalapú súlyfrissítések |
| Index | src/index/ssi.zig | Tömör szemantikai tárolás |
| Relációs | src/core_relational/nsir_core.zig | Gráfalapú tudáslogika |
| Következtetés | src/api/inference_server.zig | HTTP/REST API kezelés |

---

 Alapprimitívek

Az alapprimitívek biztosítják a szükséges absztrakciókat a többdimenziós tömbök manipulálásához, a speciális memóriakezeléshez és a robusztus modelszerializációhoz.

 Tensor: Többdimenziós tömbmotor

A Tensor struct az src/core/tensor.zig-ben a numerikus számítás elsődleges adatstruktúrája. Főbb jellemzők:

Referenciaszámlálás: Atomi referenciaszámláló (refcount) a memória több tulajdonos közötti kezeléséhez.

Copy-on-Write (CoW): Fizikai másolat csak mutáció esetén készül megosztott pufferről.

Memória integráció: Egyedi allokátorokkal (Arena, Pool, Buddy) kompatibilis.

 Memóriakezelés

| Allokátor | Cél | Szálbiztonság |
|---|---|---|
| Arena | Gyors, tömeges allokációk átmeneti adatokhoz | Mutex védett |
| PoolAllocator | Rögzített méretű blokkallokáció egyforma objektumokhoz | Lock-free / Mutex |
| SlabAllocator | Hatékony allokáció kettő-hatványos méretekhez | Mutex védett |
| BuddyAllocator | Változó méretű allokáció csökkentett töredezettséggel | Mutex védett |

A rendszer szigorúan betartja a secureZeroMemory politikát – érzékeny adatok törlődnek felszabadításkor.

 I/O, típusok és modelszerializáció

Fixpontos aritmetika: FixedPoint16, FixedPoint32, FixedPoint64, Fixed32_32 típusok determinisztikus viselkedéshez.

**PRNG** és BitSet: Xoshiro256++ implementáció sztochasztikus folyamatokhoz; dinamikus bitset állapot követéséhez.

Pontossági szintek: fp4, fp8, fp16, fp32, fp64.

Modelszerializáció: **JAIDE40**\x00 mágikus fejléc, **JSON**-kódolt metaadatok, komponensmágikus számok (**RSF**, **MGT**, **RANKER**, **PROJ**).

Fájl I/O: Szálbiztos **MMAP** absztrakció, stabil mixHash és biztonságos véletlenszám-generálás (Blake2b256 + std.crypto.random).

---

 **LLM** Pipeline – Összefoglalás

Az ML pipeline öt fő komponensből áll a **KGRU** modell betanítási és következtetési logikájához. Az adatáramlás: nyers szöveg → **MGT** tokenizáló → **RSF** processzor → **SFD** optimalizáló → **SSI** index + Ranker.

---

 **RSF** Processzor (Relational Signal Flow)
A Comprehensive Evaluation of the Reversible Scatter Flow (RSF) Architecture: A New Paradigm in Deep Learning

Introduction: The Evolution of Computational Primitives and the Information-Loss Dogma

The developmental history of artificial intelligence and, within it, deep learning can be described as a series of fundamental paradigm shifts. These leapfrog developmental points have historically always been characterized by the introduction of new "root-level" computational primitives that transcended the fundamental mathematical or representational limitations of prior systems. The 1958 Perceptron laid the foundational pillars of linear separability, introducing an entirely new concept at the dawn of machine learning. This was followed in 1989 by the Convolutional Neural Network (CNN, LeNet), which made local spatial invariance and weight sharing an independent building block, revolutionizing machine vision. The 1997 Long Short-Term Memory (LSTM) networks targeted sequential temporal dependencies and the problem of vanishing gradients through the introduction of gating mechanisms. Two decades later, the 2017 Transformer architecture elevated global contextual attention into an independent building block that entirely replaced recurrence and convolution. The Reversible Scatter Flow (RSF) architecture, which is the subject of the present analysis, proposes a new, independent computational framework that breaks with the conventional building blocks of recent decades and offers a fifth, fully autonomous paradigm.

Classical deep learning is built almost dogmatically on the fundamental premise that a neural network is necessarily an information-lossy process. Traditional architectures—through non-linear activations (such as ReLU, which clips the negative range to zero), dimension-reducing pooling layers, and complex normalization procedures—destroy a significant portion of information during every single layer transition. This mathematical destruction has an extremely severe, industry-level consequence: during gradient backpropagation, the application of the chain rule makes it indispensable to store intermediate activation states in memory. As the number of layers and context windows of models grows, this activation memory becomes an enormous bottleneck, for which only engineering "hacks" (such as gradient checkpointing or offloading) have been developed, but the fundamental theoretical problem has not been remedied.

In contrast, RSF introduces a mathematically exactly invertible primitive built exclusively on affine coupling and scatter operations, which rejects the paradigm of information destruction. The analysis explores in detail whether RSF truly holds its ground as a fifth root-level architecture. We examine the consequences of the radical abandonment of traditional building blocks, topological invertibility, the mechanisms yielding global $O(1)$ memory complexity, and the approach unprecedented in the history of deep learning whereby the architecture's correctness has been verified through machine-checked formal verification by four independent proof assistants (Lean 4, Beluga, Mizar, Twelf).

The Radical Deconstruction of Traditional Building Blocks

Modern deep learning models, including the most advanced Large Language Models (LLMs), are built upon a well-established, standardized toolkit. The Transformer concept proclaimed the principle of "Attention is All You Need," yet a deeper examination of the architecture reveals that in practice it employed a complex, heterogeneous micro-architecture. The Transformer integrated, alongside query-key-value based self-attention, massive multi-layer perceptrons (MLP), layer normalization (LayerNorm), and positional encoding. Moreover, the attention mechanism had already been introduced in 2015 by Bahdanau and his team, where it served merely as a supplementary element on top of recurrent neural networks (RNN). The Transformer's innovation was therefore not the invention of attention itself, but rather the architectural decision to elevate this single mechanism to the protagonist of information processing, discarding recurrence and convolution. RSF follows an analogous logic but carries out an even more radical purification: it extracts affine coupling (which previously existed as part of a larger ecosystem in Normalizing Flows, such as the NICE and RealNVP generative models) from its context and makes it the sole, exclusive computational primitive.

The Omission of the Attention Mechanism and Convolution

The essence of the attention mechanism is the computation of pairwise similarity (typically normalized dot product) between elements of the input sequence, followed by the aggregation of contextual values based on these dynamically generated weights. Although this procedure, which is inherently $O(N^2)$ in complexity with respect to sequence length, has proven extraordinarily successful in language modeling, RSF entirely rejects this approach. RSF does not use query-key matrix multiplication and does not employ softmax-based normalization to direct the flow of information.

The mixing of information across dimensions and tokens in RSF is instead performed by a deterministic "scatter" operation. This mechanism topologically guarantees the unobstructed flow of information without paying the extraordinarily expensive computational costs of input-dependent, dynamic routing. The rsf_scatter function implemented in the Futhark kernels mixes inputs with a butterfly operation using inverse square root of two scaling, similar to a Haar transform. Based on the documented source code, the scattering proceeds as follows: with the application of the scaling factor inv_sqrt2 = 1f32 / f32.sqrt 2f32, on one half of the vector the sums inv_sqrt2 * (x[j] + x[j + half]) are formed, while on the other half the differences inv_sqrt2 * (x[j] - x[j + half]) are computed. This structural mixing permutes the dimensions in every layer, thereby enabling the network to model the full context without computing expensive attention matrices.

Likewise absent from the architecture are convolutional filters and traditional Feed-Forward networks (MLP). In classical models, hidden layers of the $W_1 \cdot \text{ReLU}(W_2 \cdot x)$ type that perform dimension expansion are responsible for learning complex non-linear representations. In the case of RSF, weight matrices are responsible exclusively for generating the scaling (scale) and translation parameters of the affine coupling, thereby significantly increasing parameter efficiency and mathematical interpretability.

The Elimination of Normalization Layers and Traditional Activation Functions

The removal of normalization layers (BatchNorm, LayerNorm, RMSNorm) is another critical architectural decision. In conventional deep learning, the purpose of normalization is to prevent vanishing gradients and exploding gradients, as well as to manage internal covariate shift within deep networks. In the case of RSF, numerical stability and information flow are guaranteed by the fundamental geometry of the model itself. The symmetric affine coupling inherently constrains the runaway of variance due to the topological structure, thus there is no need for artificial, iterative statistical normalization of the data.

The most exceptional aspect, however, is the handling of non-linearity. The fundamental nature of classical activation functions (ReLU, GELU, Swish) is that they destroy information; ReLU, for example, maps the entire negative range to zero, causing irreversible spatial collapse in the representation. RSF, in contrast, uses the $\exp(\text{clip}(\dots))$ function, which is not an independent module inserted between layers, but rather an organic, inseparable part of the scaling branch of the affine coupling. This mathematical operation is strictly monotonically increasing, and consequently analytically invertible. The clip function (which in the LayerCore Zig module clips the inner basis sum between the default bounds of clip_min = -5.0 and clip_max = 5.0) guarantees global numerical stability. This clipping prevents the subsequent exponential scaling from causing overflow, meaning that the bounds between $\exp(-5.0)$ and $\exp(5.0)$ ensure that the gradient does not explode during training, while complete reversibility remains intact.

The following table provides a structured summary of the evolution of primitives across root-level architectures, demonstrating RSF's minimalist approach:

| Architecture | Founding | Spatial/Sequential Handling | Non-linearity | Normalization Procedure | Invertibility |
|---|---|---|---|---|---|
| Perceptron | 1958 | Independent dimensions | Threshold function | None | None |
| CNN | 1989 | Local Convolution | ReLU / Tanh | BatchNorm | None |
| LSTM | 1997 | Temporal recursion, Gates | Sigmoid / Tanh | Rarely applied | None |
| Transformer | 2017 | Positional Encoding / Attention | MLP (GELU/ReLU) | LayerNorm | None |
| RSF | New | Global Scatter | $\exp(\text{clip}(\dots))$ | None | Guaranteed Exact |

The Mathematical Formalism of Affine Coupling and Differential-Geometric Dynamics

According to the traditional paradigm of deep learning, the output of a layer can be written in the general form $y = f(Wx + b)$, where $f$ is a non-linear, unidirectional, and frequently non-invertible mapping. In contrast, the Reversible Scatter Flow (RSF) employs a completely different differential-geometric and dynamical system that is closer to the mathematics describing the flow of ideal fluids. The fundamental "DNA" of the architecture is based on affine coupling. Although affine coupling as a transformation method originally stems from Normalizing Flows generative density estimation models, in the case of RSF it appears extracted from its context, purified of all other baggage, as an independent architecture and a single computational primitive.

The Exact Equations of the Forward Pass

Based on the available source codes—particularly the definitions from the Lean 4 formal verification of the LayerCore module and the Futhark kernels—the RSF forward process can be described in a strictly deterministic and symmetric manner. The process begins with the transformation of an input vector $x$, which the system divides into two equal, discrete parts after the "scatter" operation: $(x_1, x_2)$.

Computation of the Scale Factor: The network first computes a scaling vector ($s$) from the $x_2$ component through a linear transformation, bias, clipping, and exponential transformation. Its mathematical formula is:

$$s = \exp(\text{clip}(W_s x_2 + b_s, \text{min}, \text{max}))$$

In the Lean 4 codebase, this corresponds to the logic of the computeScaleInto function, where the system computes the inner product $W_s \cdot x_2$ cycle by cycle, adds the bias $b_s$, then iteratively applies the safety clip and exp mappings. The vector $s$ thereby depends exclusively on the $x_2$ state.

Scaling of the $x_1$ Component: The system multiplies the $x_1$ vector element-wise by the freshly computed scaling factor:

$$y_1 = x_1 \odot s$$

This asymmetric, cross-directional modification is theoretically crucial. Since the scaling derives only from $x_2$, the partial derivative of $y_1$ with respect to $x_1$ will be a diagonal matrix. This drastically simplifies computations and makes subsequent invertibility deterministic, avoiding the expensive determinant computations of the Jacobian matrix.

Computation of the Translation Factor and Modification of $x_2$: Based on the now modified and fixed $y_1$ vector, the system generates a translation component, which is directly added to the $x_2$ vector:

$$t = W_t y_1 + b_t$$
$$y_2 = x_2 + t$$

(In the Zig computeTranslationRow, the Futhark rsf_flow, and the Lean 4 ComputeTranslation implementations, this straightforward, exponential-distortion-free translation is transparently evident.)

The final output of the layer is the concatenated $(y_1, y_2)$ vector, which the system permutes again (with the scatter operation) before the next layer, thereby repeating the process and deepening the information across dimensions.

The Perfect Symmetry of the Inverse Pass (Backpropagation)

The most important, revolutionary mathematical property of the above system is its deterministic and lossless invertibility. The architecture is capable of exactly restoring the input from the output without requiring any additional memory for this purpose. The InverseInPlace function executes the following inverse steps:

Restoration of $x_2$: Since the computation of the translation $t$ relied exclusively on the already fixed $y_1$ vector, and $y_1$ is available intact at the output, the transformation $t$ can be recomputed with perfect precision:

$$t = W_t y_1 + b_t$$

Consequently, the restoration of the original $x_2$ is a simple subtraction:

$$x_2 = y_2 - t$$

Restoration of $x_1$: After the original $x_2$ state has been restored intact, the scaling factor $s$ can again be deterministically generated from $x_2$, exactly as during the forward pass:

$$s = \exp(\text{clip}(W_s x_2 + b_s, \text{min}, \text{max}))$$

Then the restoration of $x_1$ is merely an element-wise division operation:

$$x_1 = y_1 \oslash s$$

This stunning symmetry—where the mathematical formula is essentially its own inverse, with only the operators of multiplication and addition swapped for division and subtraction—constitutes the soul of RSF. The ThForwardInverseIdentity theorem implemented in the Lean 4 formal proof environment deductively and irrefutably verifies the topological bijection, proving that InverseOnCore(C, ForwardOnCore(C, x)) = x.

The Paradigm of Global $O(1)$ Memory Requirement in Gradient Backpropagation

The dogma of classical deep learning research holds that for model training (for gradient backpropagation), the application of the chain rule is indispensable, which requires the physical storage in memory of the intermediate activation states of every single layer during the forward pass. This technical constraint has formed the basis of the current AI industry's hardware crisis.

The Hardware Problem of Traditional Memory Consumption

For a traditional network, such as a Vanilla Transformer or a massive CNN, the memory requirement is directly proportional to the depth of the network. If the network consists of $N$ layers, the activation memory complexity can be described with $O(N \cdot B \cdot S \cdot D)$ dimensions (where $B$ is the batch size, $S$ is the sequence length, $D$ is the representational dimension). As the number of layers in modern models grows, GPU memory (VRAM) becomes an unavoidable bottleneck. The industry has so far attempted to bridge this limitation with various "engineering hacks":

Gradient Checkpointing: Only the activations of every $K$-th layer are saved, and the intermediate layers are recomputed during backpropagation. This reduces memory consumption to approximately $O(\sqrt{N})$, but demands a brutal computational (temporal) overhead, since a significant portion of the forward pass must be run twice.

CPU Offloading: Copying activations to system memory, which slows down training due to the bandwidth limitations of the PCIe bus.

The Exact Memory Dynamics of RSF and the O(1) Breakthrough

RSF chooses a radically different, architectural solution instead of engineering patches. Since the forward pass is mathematically exactly invertible in a lossless manner, the architecture simply discards intermediate states from RAM during the forward pass. There is no need to store activations.

At the moment of backpropagation, when the error gradient arrives from the last layer, the network proceeds backward, step by step, reconstructing its own prior states. From the output $y$ it computes the input $x$, and simultaneously, in real time, it also computes the partial gradients belonging to the weights. The rsf_backward_flow and rsf_backward_layer functions of the Futhark kernels perfectly map this process: the kernel receives the output gradient (grad_out) and the restored input (x), then from these generates the gradients of the weight matrices and biases (grad_s_w, grad_t_w, grad_s_b, grad_t_b), as well as the error signal to be backpropagated to the previous layer (grad_x). No historical activation storage whatsoever is needed for this process.

This elegant mechanism results in the memory requirement for storing activations becoming completely independent of the number of layers ($N$). The memory requirement per layer—and globally for the entire network as well—remains constant, that is, of $O(1)$ complexity. The network during training does not "remember" the past in VRAM, but rather "recalculates" it algorithmically, with topological determinism.

The following table illustrates this memory-paradigm shift across different optimization techniques:

| Architecture Type | Memory Complexity (Activations) | Re-computation Cost During Training | Mathematical Exactness of Restoration |
|---|---|---|---|
| Standard (e.g., Vanilla Transformer) | $O(N)$ | None (everything is in RAM) | None (Irreversible) |
| Gradient Checkpointing (e.g., LLaMA) | $O(\sqrt{N})$ or $O(\log N)$ | High (Second Forward Pass) | None (Irreversible) |
| RSF (Reversible Scatter Flow) | $O(1)$ globally | Low (Single Inverse Pass) | Guaranteed Exact Bijection |

Type Theory and Machine-Checked Formal Verification

The most astonishing aspect of RSF—which truly makes it unique among the deep learning architectures of the past seventy years—is its machine-proven mathematical incorruptibility, built from the ground up. In the entire history of machine learning, the Perceptron, the CNN, the LSTM, and the Transformer were all "empirical" discoveries. Researchers implemented them in some programming language (typically C++ or Python), ran the data, and declared the structure functional based on the empirical decrease of the loss function (Loss curve). The rigorous mathematical behavior of these models in deeper networks frequently remained a black box, and the retroactive, approximate formulation of their theoretical background occurred after the fact.

In contrast, RSF (Reversible Scatter Flow) applies a radical, pure mathematical approach. The documentation demonstrates that the operation and architectural rules of RSF have been formally verified in four different, recognized proof assistant (Automated Theorem Prover) systems: Lean 4, Beluga, Mizar, and Twelf. This is an unprecedented undertaking in the history of deep learning.

Contextual Type Theory in Deep Learning

Based on the source code excerpts, the depth of the proofs is impressive. The Beluga specification file size is 845 KB, while the Lean 4 file is 251 KB. In the world of formal logic and contextual type theory, these file sizes count as gigantic, representing extremely complex and robust systems. A codebase of this size is not merely a description of a few axioms, but rather the logical deduction of every single state transition, tensor consistency, and dimensional invariance of the neural network.

Upon examination of the Lean 4 specification, the structural proofs are crystal clear:

Theorems such as validateTensor2D_ok and checkedMul_ok guarantee the formal integrity of tensors during multiplications and dimension changes, excluding at the compiler level the possibility of memory corruption or dimension misalignment.

The reversibility of the network is proven by the ThForwardInverseIdentity theorem. This axiom is a deductive formal proof of the topological bijection, which states that for any $C$ (RSFCore) and $x$ (input tensor), the equality InverseOnCore(C, ForwardOnCore(C, x)) = x holds true under all circumstances. This means that the accuracy of backpropagation is not approximately good, but absolutely perfect in value.

The Twelf logical programming environment also contributes to the proof. The rsf-invertible-single/i and coupling-fwd-inv-mul-cancel proofs visible in the source code verify the step-by-step correctness of inverse operations. The vec-add-sub-cancel axiom, for example, demonstrates in a strict logical system that the addition of the translation vector, followed by the subtraction of the same vector in the inverse phase, perfectly restores the memory without any bit loss occurring.

The Beluga file (rsf.bel) is a masterpiece of bounds checking and formal invariance. Derived rules such as SplitIntoIndexSafetyW, MergeFromIndexSafetyW, or LayerBackwardShapeInvariantW use contextual type theory to verify that the network cannot reference an out-of-bounds memory address during the split/merge scatter operations, and that memory deallocation (e.g., RegistryNoUseAfterFreeW) is safe.

The significance of this paradigm is epoch-making without exaggeration. In terms of the Curry-Howard correspondence, the RSF neural network is not merely a "heuristically well-functioning program code," but a proof of a mathematical theorem. The fact that the gradient does not vanish (vanishing gradient) and that the forward-backward phase is perfectly symmetric is here not a "hoped-for" behavior, but a machine-verified mathematical fact.

The following table summarizes the roles of the verification systems in the RSF architecture:

| Proof Assistant | File / Extension | Proof Focus and Role in RSF |
|---|---|---|
| Lean 4 | RSF.lean (251 KB) | Tensor validation, Invertibility identity (ThForwardInverseIdentity), State machine consistency. |
| Beluga | rsf.bel (845 KB) | Contextual type theory, Index safety (Bounds checking), Coupling invariances. |
| Twelf | rsf.twealf | Logical symmetry, Vector arithmetic cancellation (vec-add-sub-cancel), Multi-layer invertibility. |
| Mizar | rsf.miz | Formal fixation of mathematical foundations, Set-theoretic constructions. |

Critique of Existing "Reversible" Models and the Independence of RSF

For the assessment of RSF's root-level status, a critical comparison with past "reversible" attempts is indispensable. The history of deep learning knows the RevNet (Reversible Residual Network) and the Reformer (Reversible Transformer) concepts. These research efforts also employed memory-saving tricks similar to affine coupling; however, they belonged to a conceptually entirely different category.

These earlier models were in reality merely layers (wrappers) "pulled over" existing architectures, not new paradigms.

RevNet (Gomez et al.) built reversible blocks for the purpose of memory reduction, but within these blocks it continued to retain standard CNN convolutional filters, Batch Normalization, and lossy ReLU activations.

The Reformer (Kitaev et al.) wrapped the fundamental elements of the Transformer—the Feed-Forward (MLP) network and the Locality-Sensitive Hashing (LSH) Attention mechanism—into reversible blocks.

In both cases, reversibility was merely a supplementary technique, a "trick" for improving VRAM management. The information processing itself, the feature extraction and token routing, continued to be performed by the classical CNN and Transformer "primitives."

RSF, in contrast, is a pure, independent primitive. It does not contain Attention that it would make reversible, and it does not contain MLP that it would wrap into affine blocks. The affine coupling itself ($W_s$ and $W_t$ weight matrices with the corresponding scatter logic) is responsible for modeling the full complexity. The information is not directed by an external module; rather, the topological flow itself performs the iterative, fluid-dynamics-like distribution within the dimensions. Just as the Transformer in 2017 extracted attention from the RNN context and made it the sole primitive, RSF has extracted affine coupling from the context of Normalizing Flows (NICE, RealNVP) generative models and made it the absolute and sole building block of the network. This reductionist approach clearly justifies the "root-level" status.

Hardware Optimization and the "Day Zero" Problem

A common criticism against new architectures is that in empirical benchmarks they should immediately, from the first day (Day Zero), surpass dominant systems (such as GPT-4 level Transformers), otherwise they cannot be considered a breakthrough. This argument, however, is methodologically severely flawed and conflates theoretical architectural innovation with industrial product development.

The Transformer architecture was not ready for dominance at its appearance in 2017 either. It took years and billions of dollars of industry investment from Google, OpenAI, NVIDIA, and other players before the necessary software and hardware ecosystem was built up under the Transformer: maximally optimized CUDA kernels, FlashAttention v1/v2, specific quantization procedures, and distributed training frameworks. The Transformer exploited the nature of GPUs optimized for parallel matrix multiplication. Although it was a brilliant engineering insight ("engineering hack") to discard the sequential RNN for parallelism, the theoretical and mathematical depth of the architecture was modest—a simple normalized dot product sent through a softmax layer.

The theoretical novelty of RSF is deeper by orders of magnitude, but for this unparalleled formal framework, its own hardware-level software infrastructure must be built. The presented source code repositories (Zig and Futhark) are working on precisely this challenge.

In the integration layer, accel_interface.zig and futhark_bindings.zig provide low-level memory management between VRAM and the CPU host through C-standard interfaces.

The PinnedMemory structure enables asynchronous and lightning-fast data movement through cudaHostAlloc calls, avoiding slowdowns caused by pageable memory.

The FutharkArray2DF16 structure shows that the architecture is purposefully optimized for the half-precision (FP16) floating-point computations favored by modern AI accelerators (such as NVIDIA Tensor Cores).

The RSFAccelerator Zig structure directly delegates computation to the Futhark-compiled GPU kernels (rsf_forward_layer, rsf_backward_layer, trainingStep), minimizing kernel launch overhead.

Although achieving empirical dominance may require further iterations beyond Futhark-based compilation (such as native, hand-written CUDA cores, specific hardware-aware routing optimizations), the existing codebase highlights that the system's engineering architecture also builds upon serious foundations, complementing the robust mathematical proofs. The theoretical novelty and the O(1) memory complexity, however, are already guaranteed from the architecture's fundamental principle, regardless of when the hardware ecosystem catches up to the Transformer's optimization level.

Historical and Architecture-Theoretical Conclusion

In the course of the above, exhaustively detailed analysis, we examined every dimension of the Reversible Scatter Flow (RSF) architecture, focusing on the absence of "building blocks," the mathematical formalism, the memory complexity, the formal verification, and the architectural independence. The synthesis of facts leads to an unequivocal conclusion regarding the model's novelty.

It is interesting to observe the arc of deep learning's development, which clearly points from "heuristic architectures" toward "analytic architectures." While the Transformer and the models that preceded it were products of experimental, engineering trial and error (which were only examined retrospectively with statistical and mathematical tools), RSF was deduced from pure mathematics, from the theory of geometric flows of differential equations, and from contextual type theory (a top-down approach), and then this axiom system was translated into C/Zig/Futhark engineering code.

The design decision that RSF entirely rejects classical neural dogmas—there is no ReLU, no MLP, no BatchNorm, no Attention—executes precisely the architectural purification that the Transformer also undertook in 2017 when it discarded the dominant RNN and CNN elements. This refined philosophy concentrating on a single primitive (affine coupling and scatter) establishes the network's autonomy.

In summary, it can be stated that from an architectural and theoretical standpoint, the Reversible Scatter Flow (RSF) can with full justification and rational scientific reasoning claim the "root-level" classification alongside the Perceptron, the CNN, the LSTM, and the Transformer. RSF possesses its own mathematical principle independent of other networks (the dynamics of affine coupling), its own internal topological symmetry (through forward/inverse determinism), innovative memory management (through discarding the activation store and backward $O(1)$ re-computation), and theoretical purity verified by machine in four independent languages (Lean 4, Beluga, Mizar, Twelf), unprecedented in the history of computer science in the field of deep learning. Whether its industrial acceptance and scaling impact in the LLM era will reach the success of the Transformer will be determined by the future development of the optimized hardware ecosystem (CUDA adaptations, distributed frameworks), but the fact of its architectural-level, independent root novelty and the model's paradigm-shifting power is indisputable. 
A **JAIDE** rendszer alapvető neurális hálózati motorja az src/processor/rsf.zig-ben implementált.

 LayerCore struktúra

| Mező | Típus | Leírás |
|---|---|---|
| s_weight | Tensor | Térbeli súlymátrix (dim × dim) |
| t_weight | Tensor | Időbeli súlymátrix (dim × dim) |
| s_bias | Tensor | Térbeli bias vektor (1 × dim) |
| t_bias | Tensor | Időbeli bias vektor (1 × dim) |
| rwlock | Thread.RwLock | Paraméterfrissítések szinkronizálása |

 Inicializálás és validálás

Xavier inicializálás: Súlyok mintavételezése egyenletes eloszlásból a $\pm \sqrt{6/(fan_{in} + fan_{out})}$ határon belül.

Kulcsfüggvények: initOwned, validateTensor2D, ensureFiniteSlice (NaN/Inf ellenőrzés).

 Számítási pipeline

Előre menet: mátrix-szorzás a bemeneten térbeli és időbeli súlyokkal, majd bias hozzáadása.

Visszafelé menet: Gradiens vágás -5.0 és 5.0 között (explodáló gradiens megelőzése), ensureGradients lusta gradiens allokálással.

 Modelmentés/betöltés

Bináris szerializáció verziózással (SAVE_VERSION = 4). A tensorsOverlap függvény biztosítja, hogy forrás és cél pufferek ne legyenek azonosak.

 Formális verifikáció

Lean 4 bizonyítékok: verzióinvariáns, alakaritmetika, memóriabiztonság.

---

 **MGT** Tokenizáló (Morpheme-Guided Tokenization)

Specializált tokenizáló, amely nem csupán bájt-pár kódolást (**BPE**), hanem morfémaalapú bontást is alkalmaz, különösen magyar és angol szövegekhez.

 Alapadatstruktúrák

| Mező | Típus | Leírás |
|---|---|---|
| token_to_id | StringHashMap(u32) | Szöveg → ID leképezés |
| id_to_token | AutoHashMap(u32, []const u8) | ID → szöveg leképezés |
| prefixes | StringHashMap(u32) | Ismert prefixek |
| suffixes | StringHashMap(u32) | Ismert szuffixek |
| roots | StringHashMap(u32) | Alapszótövek |
| anchors | StringHashMap(u64) | SSI anchor tokenek |
| bpe_pairs | StringHashMap(BPEMerge) | Megtanult összevonási párok |

Speciális tokenek: [**PAD**] (0), [**UNK**] (1), [**BOS**] (2), [**EOS**] (3).

 Morfémabontás

Angol prefixek: un, re, pre, dis, mis stb.

Magyar prefixek: meg, el, fel, le, be, ki, szét stb.

Angol szuffixek: ing, ed, tion, ness, ment stb.

Magyar szuffixek: ság, ség, ban, ben, hoz, nak, nek stb.

 Allokátor integráció

initWithArena, initWithPool, initWithBuddy az egyedi allokátorokhoz. Az allocated_strings ArrayList biztosítja a memóriakezelést a leálláskor.

---

 **SFD** Optimalizáló (Stochastic Fractal Descent)

A **KGRU** modell elsődleges betanítási motorja, amely **RSF** rétegparamétereken kezeli a súlyfrissítéseket.

 **SFD** struktúra

SFDConfig: learning_rate, beta1 (momentum bomlás), beta2 (skálázási bomlás), epsilon (numerikus stabilitás).

SFDParam: Egy paraméter súlyát, gradienst és két pillanat tensorát tartalmazza.

 Gradiens frissítési szabály

## Momentum frissítés: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$

## Variancia frissítés: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ ## Bias korrekció az aktuális időlépés alapján ## Súly módosítás: $w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ ## Kvantálás csökkentett pontossági célok esetén

 Kvantálás támogatás

- fp4: Értékek -8.0 és 7.0 közé szorítva, 0.5 pontossággal
- fp8: Értékek -**448**.0 és **448**.0 közé szorítva, 1/16-os pontossággal
- fp16: 1/**1024** pontosság

---

 **SSI** Index (Succinct Semantic Index)

Nagy teljesítményű, trie-alapú hash-fa struktúra token sorozatok indexelésére és visszakeresésére.

 Belső adatstruktúrák

| Struktúra | Szerepkör | Kulcsmezők |
|---|---|---|
| Segment | Összefüggő token sorozat | tokens, position, score, anchor_hash |
| Node | Trie ág vagy levél | hash, children, segment, collision_chain |
| CollisionNode | Láncos lista hash ütközésekhez | seg, next |

Konstansok: Bucket szélesség 6 bit, 64 bucket, max mélység 6 szint.

 Hashelés és indexelés

mixHash: Multiplikatív hash 0x9E3779B185EBCA87 konstanssal. hashTokens: 64-bit hash token sorozathoz. computeAnchorHash: Dokumentum pozíció és token sorozat kombinálásával.

 Sorozat beszúrás (addSequence)

Az anchor_hash kiszámítása után a gyökértől bejárja a fát, levélbe szúr be vagy ütközési lánchoz fűz.

 Hardveres gyorsítás (SSISearch)

Clash-ben írt Mealy állapotgép három állapottal: Idle, Fetching (memóriából csomópontot kér), Comparing (kulcsot hasonlít). Max mélység 64, 32-bit mutatók, 64-bit hash kulcsok.

 Formális verifikáció és fuzz tesztelés

Lean 4 formális specifikáció, **5000** iterációs fuzz tesztelő.

---

 Ranker (**LSH**-alapú eredmény rangsorolás)

A pipeline utolsó szakasza az **SSI** jelöltjeinek pontozásáért és újrarangsorolásáért felelős.

 Konfigurációs konstansok

| Paraméter | Érték | Leírás |
|---|---|---|
| STREAMING_BUFFER_SIZE | 1024 | Max kapacitás a streaming rankerhez |
| DIVERSITY_WEIGHT | 0.3 | Token változatosság hatása |
| PROXIMITY_WEIGHT | 0.3 | Anchor token közelség hatása |
| BASE_SCORE_WEIGHT | 0.4 | Elsődleges n-gram SSI pontszám súlya |
| OVERLAP_WEIGHT | 0.3 | Közvetlen token átfedés súlya |
| JACCARD_WEIGHT | 0.3 | Jaccard hasonlóság súlya |

 Pontozási módszertan

Alappontozás: N-gram analízis decay-súlyozással, sokszínűség-számítás (egyedi/összes token aránya), anchor közelség mérés.

Lekérdezésalapú újrarangsorolás: Token átfedés (lekérdezési tokenek aránya a célsorozatban), Jaccard hasonlóság (intersection/union).

 Formális verifikáció

Viper segítségével verifikálják a halom-biztonságot és matematikai invariánsokat.

---

 Core Relational Engine (Alaprelációs Motor)

A **JAIDE** elsődleges következtetési és tudásreprezentációs alrendszere, amely az **NSIR** gráfot és kvantumlogikát ötvözve nem-lineáris következtetésre képes.

 Rendszerkomponensek

| Komponens | Kód entitás | Elsődleges felelősség |
|---|---|---|
| NSIR Gráf | SelfSimilarRelationalGraph | Tudást fraktális gráfként tárol |
| Kvantumlogika | RelationalQuantumLogic | Kvantum kapukat alkalmaz valószínűségi következtetéshez |
| Z-Runtime | ZRuntime | Relációs műveletek végrehajtási állapotát kezeli |
| ESSO Optimalizáló | EntangledStochasticSymmetryOptimizer | Gráf topológiát optimalizál szimmetria alapján |
| Chaos Core | ChaosCoreKernel | Aszinkron feladatütemezés és adatáramlás |
| Orchestrator | ReasoningOrchestrator | Helyi, globális és meta szintű következtetés koordinálása |

---

 **NSIR** Gráf és Kvantumlogika

 Csomópont és Qubit reprezentáció

| Típus | Mező | Leírás |
|---|---|---|
| Qubit | a | Komplex amplitúdó a |0⟩ bázisállapothoz |
| Qubit | b | Komplex amplitúdó a |1⟩ bázisállapothoz |
| Node | id | Egyedi csomópontazonosító |
| Node | qubit | A csomópont kvantumállapota |
| Node | phase | Fázisszög interferencia-számításokhoz |

 Él minősége

| EdgeQuality | Leírás |
|---|---|
| superposition | Kapcsolat egyszerre több potenciális állapotban létezik |
| entangled | Célcsomópont állapota a forráscsomóponttól függ |
| coherent | Stabil kapcsolat fenntartott fázisszinkronizálással |
| collapsed | Klasszikus, rögzített kapcsolat |
| fractal | Önhasonló kapcsolat, különböző léptékekben ismétlődik |

 Kvantum kapurendszer

Egyes-, kétqubites (**CNOT**) és háromqubites (Toffoli) kapuk, FRACTAL_TRANSFORM doménspecifikus kapu. Elérhető: Hadamard, Pauli-X/Y/Z, Relációs **AND**/OR/**XOR**. A QuantumCircuit struct kapusorozatok végrehajtásához.

 Z-Runtime és végrehajtási előzmények

A HistoryEntry egyedi műveleteket rögzít (assign, transform, measure). Az ExecutionAction magas szintű műveleteket definiál (entangle_variables, propagate_information).

 Jelterjedés

A SignalPropagationEngine az amplitude, phase és frequency alapján szimulálja a jelterjedést. Az EdgeQuality és fractal_dimension alapján transzformálja a jelállapotokat.

 Időbeli gráfkezelés

A TemporalGraph a csomópontok és élek historikus állapotát Timestamp-pel tárolja. A Lean 4 verifikáció biztosítja a kvantumállapot-átmenetek konzisztenciáját.

---

 **ESSO** Optimalizáló és Chaos Core

 **ESSO**: Összefonódó Sztochasztikus Szimmetria Optimalizáló

Szimmetria-transzformációk az **NSIR** gráf optimalizálásához: Identity, Reflection, Rotation (90/**180**/**270**), Translation. Az OptimizationState követi az aktuális gráfot, energiáját és szimmetria-előzményeit.

Kulcsfüggvények: optimize() (sztochasztikus leszállás), applySymmetryToGraph() (transzformáció alkalmazása), calculateGraphEnergy() (energia értékelése).

 Chaos Core

Tartalom-Archiváló Tárolás (**CAS**): A MemoryBlock entitásokat tartalomhash azonosítja, lehetővé téve a deduplikálást és a függőségnyomkövetést.

Dinamikus feladatütemező: A DynamicTaskScheduler prioritás és adatfüggőségek alapján rendel feladatot magokhoz.

| Komponens | Felelősség |
|---|---|
| ContentAddressableStorage | Tartalom-archiváló memóriakezelés |
| DynamicTaskScheduler | Függőségtudatos feladat-végrehajtás |
| DataFlowAnalyzer | Adatáramlás és szűk keresztmetszetek elemzése |
| ChaosCoreKernel | Alkomponensek orkestrálása |

 Meglepetés Memória

Prioritás-alapú adatmegőrzés újdonság szerint. SurpriseMetrics három tényező alapján: Jaccard diszhasonlóság, tartalomhash-távolság, időbeli újdonság.

Konfigurációs konstansok: RETENTION_BASE_WEIGHT = 0.5, RETENTION_AGE_WEIGHT = 0.3, RETENTION_FREQUENCY_WEIGHT = 0.2, DEFAULT_SURPRISE_THRESHOLD = 0.3.

---

 Következtetési Orkesztrátor és Támogató Modulok

 Következtetési Orkesztrátor

Háromszintű következtetés: Helyi (azonnali csomópontszomszédságok), Globális (hosszú hatótávolságú strukturális igazítás), Meta (a következtetési folyamat önreflexiója).

 Relációs **GPU** (R-**GPU**) és NoC

Szimulált hardverabsztrakció masszívan párhuzamos gráfműveletekhez. Az AsynchronousNoC üzenetirányítást kezel prioritás-alapú sorral. Üzenettípusok: weight_update, graph_sync, isomorphism_result.

 Vektorprocesszor (**VPU**)

**SIMD** absztrakciók: aritmetika (add, sub, mul, divChecked), geometria (dot, magnitude, normalize), hardveres gyorsítás (fma, sqrt).

 **CREV** Pipeline

Feldolgozási szakaszok: tokenizálás → hármas kinyerés (Subject-Relation-Object) → validálás → integráció az **NSIR** gráfba → **SSI** indexelés.

 Adathalmazok elfedése és biztonság

Paillier Homomorphic Encryption: Matematikai műveletek titkosított szövegen végrehajthatók. encrypt(plaintext: i64) → u512 ciphertext, add(c1, c2) → homomorphic összeadás.

Biztonság: safeIntCast, safePtrCast (határellenőrzés), secureZeroBytes (volatile írások), secureCompare (konstans idejű összehasonlítás).

 C **API**

Stabil **FFI**: CGraph ↔ GraphContext, COptimizer ↔ EntangledStochasticSymmetryOptimizer. Hibakódok: JAIDE_SUCCESS (0), JAIDE_ERROR_NULL_POINTER (-1), JAIDE_ERROR_OUT_OF_MEMORY (-18).

---

 Kvantumszámítási Integráció

 Architektúra áttekintés

Három fő komponens: Hardware Abstraction (quantum_hardware.zig), Cloud Integration (ibm_quantum.zig), Task Adaptation (quantum_task_adapter.zig).

 **IBM** Quantum integráció

initWithCrn: **API** tokennel és **CRN**-nel inicializál. submitJob: OpenQASM küldés ibm_brisbane backendre, **1024** shottal. getJobResult: Állapot és mérési adatok lekérése.

 Hardver specifikációk

| Backend | Qubitek | T1 átlag (ns) | T2 átlag (ns) | Readout hiba |
|---|---|---|---|---|
| Heron | 133 | 350,000 | 200,000 | 0.008 |
| Eagle | 127 | 200,000 | 120,000 | 0.015 |
| Falcon | 27 | 100,000 | 80,000 | 0.020 |
| Condor | 1121 | 400,000 | 250,000 | 0.006 |

 Kvantum feladatadapter

Magas entanglement és fraktális dimenzióval rendelkező részgráfokat azonosít. Entanglement_threshold alapértelmezetten 0.5, fractal_threshold alapértelmezetten 1.5.

Konfigurációs konstansok: MAX_QUBITS_SIMULATION = 20, SIMULATOR_MAX_SHOTS = **100**,**000**, JOB_WAIT_TIMEOUT_MS = 60,**000**.

---

 Hardveres Gyorsítás

A hardveres gyorsítás egy többszintű stack a **GPU**, **FPGA** és **ASIC** célokra, feltételesen engedélyezhető a gpu_acceleration build jelzővel.

---

 Futhark Gyorsítási Réteg

 Rendszeráttekintés

Három réteg: Futhark Kernelyek (.fut fájlokban), Zig **FFI** Kötések, Acceleráció Interfész (magas szintű Zig wrapper).

 Futhark Kernelyek

**RSF** Forward and Backward: Spektrális transzformáció (weights_s-szel), időbeli transzformáció (weights_t-vel), fúzionált forward (ReLU + Layer Normalization egyetlen **GPU** hívásban).

**SFD** optimalizálás: Momentum-alapú súlyfrissítés **GPU**-n: $v_{t+1} = \mu v_t + \eta g$, $w_{t+1} = w_t - v_{t+1}$.

 Zig Acceleráció Interfész

FutharkContext: **GPU** eszköz életciklus és alapértelmezett hangolási paraméterek kezelése.

PinnedMemory: cudaHostAlloc-ot használ gyors **DMA**-átvitelhez, cudaFreeHost-tal felszabadítva.

FutharkArray2DF16: **GPU** tömb életciklus, futhark_new_f32_2d, futhark_values_f32_2d, futhark_free_f32_2d.

 Fractal **LPU**

A FractalTile struktúra rekurzívan osztja a memóriát hausdorff_dim (Hausdorff Dimenzió) alapján. A balanceLoad normalizálja a pending_ops értékeket a számítási egységek között.

---

 **RTL** Hardvermodulok (Clash/Haskell)

 SSISearch: Bináris fa keresőmotor

Adattípusok: HashKey64 (64-bit hash), NodeAddr32 (32-bit memóriacím), TreeNode (nodeKey, leftChild, rightChild, érvényességi bit).

Keresési állapotgép:

| Állapot | Leírás |
|---|---|
| Idle | SearchRequest-re vár |
| Fetching | Memóriavezérlőtől TreeNode adatot vár |
| Comparing | searchKey-t hasonlít a nodeKey-hez |

Keresési logika: egyezés → found = True; kisebb → leftChild fetch; nagyobb → rightChild fetch; max mélység (64) → befejezés.

 RankerCore: Pontozási pipeline

finalScore = baseScore + positionBias, ahol positionBias = positionBiasScale / (position + 1) és positionBiasScale = **1000**.

 MemoryArbiter: Többklienses busz-arbitrátor

Max 4 kliens, round-robin politika, ServiceCycles = 4. A filterResp biztosítja, hogy csak a kérő kliens kapja meg az adatát.

---

 **FPGA** Implementáció

 Rendszerarchitektúra

A top_level.v **FPGA** felső szintű modul integrálja: **AXI4**-Lite Slave (vezérlési interfész), SSISearch Core, RankerCore, MemoryArbiter.

 **AXI4**-Lite regisztertérkép

| Cím | Regiszternév | Leírás |
|---|---|---|
| 16'h0000 | ADDR_CONTROL | 0. bit: SSI Start; 1. bit: Ranker Valid |
| 16'h0004 | ADDR_STATUS | 0. bit: SSI Found; 1. bit: Ranker Done; 16-31 bitek: Rank |
| 16'h0010 | ADDR_SSI_KEY_L | SSI Keresőkulcs (alsó 32 bit) |
| 16'h0014 | ADDR_SSI_KEY_H | SSI Keresőkulcs (felső 32 bit) |
| 16'h0018 | ADDR_SSI_ROOT | SSI gyökércsomópont memória-cím |
| 16'h0038 | ADDR_RNK_SCORE | Ranker kalkuláció alap pontszáma |
| 16'h003C | ADDR_RNK_RES | Ranker végső számított pontszáma |

 Fizikai megszorítások és időzítés

Órajel: **100** MHz (J3 láb). Reset: aktív-alacsony, rst_n (**K11** láb).

Többciklusú útvonalak: Arbiter → Memory: 4 ciklus setup; **SSI** → Memory: 8 ciklus setup.

---

 **ASIC** Implementáció (**TSMC** 28nm)

 Szintézis folyamat

Synopsys Design Compiler, **TSMC** 28nm standard cell könyvtárak (slow.db, typical.db, fast.db). Órajel periódusa 10.0 ns (**100** MHz), uncertainty 0.2 ns.

Többciklusú útvonalak: MemoryArbiter: 4× setup, 3× hold; SSISearch: 32× setup, 31× hold.

Optimalizálás: compile_ultra -gate_clock (clock gating), területi cél = 0 (legkisebb lábnyom).

 Floorplanning és fizikai tervezés

Die mérete **5000**×**5000** egység, core offset **100** egység, sor-arány 0.70.

PG háló: **METAL6** (vízszintes) és **METAL5** (függőleges), 10.0 szélesség, PG strap pitch **120**.0.

Pin elhelyezés:

| Jelcsoport | Die oldal | Rétegek |
|---|---|---|
| AXI Write | Bottom | METAL5, METAL6 |
| AXI Read | Right | METAL5, METAL6 |
| Memory | Left | METAL5, METAL6 |
| System (clk, rst_n) | Left (Offset) | METAL6 |
| Peripherals | Top | METAL5, METAL6 |

Kimeneti fájlok: top_level_floorplan.v, top_level_floorplan.def, top_level_floorplan.tcl.

---

 Elosztott Betanítás

 Rendszeráttekintés

Hierarchia: **GPU** Koordinátor (eszközspecifikus erőforrások) → Elosztott Trainer (magas szintű ciklus, particionálás, szinkronizálás) → Futhark-gyorsított Trainer (**GPU** kernelek).

 **GPU** Koordinátor

Eszközkezelés (cudaSetDevice), **NCCL** kommunikátorok inicializálása, memória műveletek (cudaMalloc, cudaFree, cudaMemcpy), allReduce és broadcast float32/float16 típusokhoz. Barrier mechanizmus dummy allReduce hívással szinkronizáláshoz.

 Elosztott Trainer (Futhark)

Adathalmaz-szeletelés: sorokat számol a .jsonl adathalmazban, base_samples_per_rank kiszámítása, minden rank a kijelölt start_line-tól olvas.

Gradiens aggregáció: helyi gradiens számítás, majd GPUCoordinator.allReduceFloat16 hívás a world_size-szal való osztás előtt.

 Tensor primitívek elosztott betanításhoz

N-dimenziós alakok stride-alapú indexeléssel, atomi referenciaszámlálás, kontiguus memória ellenőrzés az **NCCL** átvitelekhez. Fixed32_32 típus determinisztikus ellenőrzéshez.

 Formális verifikáció

Lean 4 bizonyítja a Fixed32_32 kommutatív és asszociatív törvényeit, a **PRNG** determinizmusát és a határon belüli maradást.

---

 **NCCL** Kötések és Modal Felhő Telepítés

 **NCCL** Kötések

| Függvény | Leírás |
|---|---|
| ncclGetUniqueId | Egyedi azonosítót generál a bootstrap kommunikátorhoz |
| ncclCommInitRank | Új kommunikátor objektumot hoz létre egy adott rankhoz |
| ncclAllReduce | Redukciót végez az összes GPU-n és elosztja az eredményt |
| ncclBroadcast | Másolja a puffert az összes rankra |
| ncclReduceScatter | Redukciót végez és szétszórja az eredményt |

**CUDA** helper integrációk: cudaMalloc, cudaFree, cudaMemcpy, cudaStreamCreate, cudaStreamSynchronize, cudaSetDevice, cudaGetDeviceCount.

 Modal **GPU** Absztrakció

ModalGPUClient Zig-natív interfész a Modal Cloud **API**-hoz. Inicializáláskor 8 **GPU**-t céloz (**B200**). deployTrainingJob: **JSON** payload a jaide-v40-training képpel, getJobStatus: feladatállapot lekérés.

 Felhő Telepítési Szkriptek

Infrastruktúra beállítás (modal_setup.sh): Modal **CLI** telepítés, hitelesítés, jaide-training-data kötet létrehozása, **API** kulcstitkosítások.

Elosztott betanítás (modal_distributed_train.py): **B200**:8 + **256** GB **RAM** + 3 TB lemez, Ubuntu 22.04 + **CUDA** 12.4 + Zig 0.13.0, HunSum-1 adathalmaz letöltés és **JSONL** konverzió, 8 **GPU**-n elosztott betanítás.

Következtetési szkript (modal_inference.py): **B200**:1 **GPU**, kötegelt feldolgozás prompt-lista iterálásával, **CLI** belépési pont modal run-nal.

---

 Következtetési Szerver **API**

 Szerverfiguráció

| Mező | Típus | Alapértelmezett | Leírás |
|---|---|---|---|
| port | u16 | 8080 | TCP port |
| host | []const u8 | *127.0.0.1* | Kötési cím |
| max_connections | u32 | 100 | Max egyidejű kapcsolatok |
| batch_size | usize | 32 | Kérések száma következtetési menetenként |
| rate_limit_per_minute | u32 | 10 | Max kérés IP-nként percenként |
| require_api_key | bool | true | API kulcs validálás engedélyezése |

 Kéréskezelési pipeline

## Kapcsolat fogadás a konfigurált gazdagépen/porton

## Sebességkorlátozás ellenőrzése (csúszóablak, 60 másodperces ablak) ## JSON elemzés InferenceRequest struct-okká ## Tokenizálás MGT segítségével ## Forward pass az RSF réteg stackon ## JSON szerializálás InferenceResponse-ban

 **API** Végpontok

**POST** /v1/inference: Kérés: text, max_tokens (opcionális), return_embeddings (opcionális). Válasz: tokens, embeddings (opcionális), processing_time_ms.

**GET** /health: Válasz: status (*healthy*), uptime_seconds, model_loaded.

---

 Formális Verifikáció és Biztonság

 Verifikációs Infrastruktúra

| Eszköz | Felhasználás |
|---|---|
| Lean 4 | NSIR, RSF rétegek, temporal gráf, Surprise Memory mély strukturális bizonyításai |
| Agda | Memóriabiztonság, arena allokáció, NSIR gráf invariánsok konstruktív bizonyításai |
| Cryptol & SAW | Rendszerkonstansok specifikációja, Zig/LLVM kód validálása |
| Viper | Ranker halom-biztonság és memória invariánsok |
| Circom/ZK | Nem-interaktív bizonyítékok következtetési nyomokhoz Poseidon hasheléssel |
| Beluga/Mizar/TwElf | RSF réteg relációs logika és típuselméleti alapok |

 Biztonsági és Invariáns Taxonómia

| Invariáns típus | Prioritás | Leírás |
|---|---|---|
| MEMORY_SAFETY | 10 | Puffertúlcsordulások és use-after-free hiánya |
| TYPE_SAFETY | 9 | Típusinvariánsok megőrzése a Z-runtime-ban |
| CONNECTIVITY | 8 | Az NSIR gráf strukturális integritása |
| QUANTUM_STATE | 5 | Érvényes valószínűségeloszlások a kvantumlogika rétegekben |

 Lean 4 Formális Bizonyítékok

Fő bizonyítási területek: **FNDS** helyesség (önhasonlóság megőrzése), **RSF** réteg invariánsok (jeltartósság mély rétegkészleteken), Surprise Memory (retention_priority számítás és exceedsThreshold logika verifikálása).

 **SAW**, Cryptol, Viper, **ACL2**, Agda és ZK Verifikáció

**SAW** és Cryptol: A MainSpec.cry a rendszerhatárok forrása (pl. MAX_TENSOR_SIZE, FILE_MAGIC_RSF). Az src/verify.saw **LLVM** szinten verifikálja a Zig kódot.

Agda memóriabiztonság: sizeConservation invariáns bizonyítása: arenaAllocated + arenaRemaining ≡ bufferSize.

Zero-Knowledge következtetés: inference_trace.circom Poseidon hasheket használ a belső tensor értékek felfedése nélküli verifikációhoz.

**ACL2** és relációs logika: **MGT** tokenizáló állapotátmenetek modellezése. Beluga/Mizar/TwElf az **RSF** referenciaszámlálás és tensor érvényesség bizonyításához.

 Biztonsági Politika

Bell-LaPadula és Biba modellek alapján kötelező hozzáférés-szabályozás. Biztonsági szintek: **PUBLIC**-tól TOP_SECRET-ig. Integritási szintek: **UNTRUSTED**-tól **KERNEL**-ig.

| Biztonsági funkció | Implementáció |
|---|---|
| Hozzáférés-szabályozás | Bitmaszk alapú AccessRight (READ, WRITE, EXECUTE, ADMIN) |
| Információáramlás | dominates ellenőrzés több-szintű biztonsághoz |
| Integritás-ellenőrzések | Időzítés-biztos egyenlőség és többhash-támogatás |

---

 Tesztelés és Fuzzing

 Egységtesztelési Infrastruktúra

| Parancs | Cél forrás | Leírás |
|---|---|---|
| zig build test | src/main.zig | Elsődleges egységteszt csomag (Futhark kernel linkekkel) |
| zig build test-tensor | src/core/tensor.zig | Elszigetelt tesztek a Tensor motorhoz |
| zig build test-memory | src/core/memory.zig | Egyedi allokátorok verifikálása |

 Fuzz Tesztelési Keretrendszer

Memória rendszer fuzzing (fuzz_memory.zig): Véletlenszerű alignedAlloc, free és realloc műveletek. Aktív allokációk nyomon követése, memóriafolyás-ellenőrzés a futás végén.

Tensor műveleti fuzzing (fuzz_tensor.zig): Véletlenszerű alakok (max rank 4), random f32 adatok. Tesztelt műveletek: összegzés, max, L2-norma, skálázás. NaN/Inf ellenőrzés numerikus instabilitásra.

**SSI** index fuzzing (fuzz_ssi.zig): **5000** iteráció, váltakozó addSequence (max **1024** token) és retrieveTopK hívásokkal. Sikeres és sikertelen műveletek, indexelt tokenek összeszámlálása.

 Stressz-tesztelés: Szálbiztonság és Referenciaszámlálás

12 szál egyidejűleg manipulál megosztott Tensor objektumkészletet. Atomi barrier biztosítja az egyidejű start-ot. Tesztelt műveletek: retain, release, clone, set, tömeges aritmetika.

---
 SurpriseMemory 

A SurpriseMemory egy intelligens memóriakezelő rendszert biztosít, amely a megtartási döntéseket az adatok újdonsága alapján hozza meg, nem csupán a hagyományos metrikák, például a hozzáférési sorrendek vagy a gyakoriság alapján. A rendszer több dimenziós „meglepetési" pontszámokat számít az eltárolt adatblokkokhoz, és ezeket a pontszámokat használja annak meghatározásához, hogy melyik blokkot tartsa meg, és melyiket távolítsa el, ha a kapacitáskorlátot elérték.

 Kettős implementációs stratégia

A SurpriseMemory egy egyedi kettős implementációs megközelítést alkalmaz: egy gyakorlati, éles üzemeltetésre szánt implementáció Zig-ben, párhuzamosan egy formális matematikai specifikációval Lean 4-ben. Ez az architektúra egyszerre biztosít valós felhasználhatóságot és matematikai korrektség-garanciákat.

 Az alapvető probléma és megoldás

A hagyományos gyorsítótár-kizárási szabályzatok (LRU, LFU, FIFO) kizárólag a hozzáférési minták alapján döntenek. A SurpriseMemory bevezeti a újdonságtudatos gyorsítótárazást: a meglepő vagy újszerű adatokat tartalmazó blokkok – amelyeket tartalmi hasonlóság, hash-eltérés és időbeli egyediség alapján mérnek – magasabb megtartási prioritást kapnak.

| Hagyományos gyorsítótárazás | SurpriseMemory |
|---|---|
| A legrégebben használt blokkokat ejti | A legalacsonyabb kombinált megtartási pontszámú blokkokat ejti |
| Nincs tartalomelemzés | Több dimenziós meglepetési számítás |
| Csak hozzáférési gyakoriság | Meglepetési pontszám + hozzáférési frekvencia + kor |
| Nincs deduplikáció-tudatosság | Integrálódik a tartalom-alapú tárolással |
| Nincs formális ellenőrzés | 271+ bizonyított invariáns és korrektségi tétel |

 Kulcsfontosságú adatstruktúrák

| Struktúra | Cél | Kulcsmezők |
|---|---|---|
| SurpriseMemoryManager | Központi vezérlő | storage, surprise_records, surprise_threshold, statistics, mutex |
| SurpriseMetrics | Több dimenziós újdonsági pontszám | jaccard_dissimilarity, content_hash_distance, temporal_novelty, combined_surprise |
| SurpriseRecord | Blokkonkénti metaadat | block_id, surprise_score, retention_priority, access_frequency, last_access_time, creation_time |
| SurpriseMemoryStatistics | Összesített metrikák | total_blocks, high_surprise_blocks, low_surprise_blocks, average_surprise, eviction_count |
| ContentAddressableStorage | Blokkperzisztencia réteg | Automatikus deduplikáció tartalmi hash alapján |

 A meglepetési számítási folyamat

A rendszer alapvető újítása a meglepetési számítás, amely három független metrikát kombinál az adatok újdonságának meghatározásához.

 A megtartási prioritás képlete

Amikor a kapacitás megtelik, a blokkokat egy megtartási prioritás pontszám alapján rangsorolják, amely kombinálja a meglepetést a hozzáférési mintákkal:


retention_priority = surprise_score × (
    RETENTION_BASE_WEIGHT × 1.0 +
    RETENTION_AGE_WEIGHT × (1 / (1 + age_milliseconds)) +
    RETENTION_FREQUENCY_WEIGHT × log(access_frequency)
)

ahol:
    RETENTION_BASE_WEIGHT = 0.5
    RETENTION_AGE_WEIGHT = 0.3
    RETENTION_FREQUENCY_WEIGHT = 0.2


Ez a képlet biztosítja:
- Alap prioritás (50%): A meglepő adatok eredendően magasabb értékkel bírnak
- Frissességi bónusz (30%): A nemrég elért blokkok ideiglenes védelmet kapnak
- Frekvencia bónusz (20%): A sűrűn elért blokkok idővel értéket halmoznak fel

 Menetbiztonság és párhuzamosság

A Zig implementáció szálbiztos egyidejű hozzáférést biztosít mutex-védett műveletekkel. Minden publikus metódus belépéskor megszerzi a mutexet, és defer segítségével felszabadítja azt.

 Formális ellenőrzési architektúra

A Lean implementáció matematikai garanciákat biztosít bizonyított invariánsok hierarchiáján keresztül, beleértve 271+ tételt, amelyek garantálják a rendszer helyes működését minden lehetséges műveleten át.

 Rendszerhatárok és függőségek

A Zig implementációnak két külső függősége van: a Zig Standard Library és a chaos_core.zig (amely biztosítja a ContentAddressableStorage-t és a DataFlowAnalyzer-t). A Lean specifikáció függőségmentes és egy tiszta funkcionális modellt valósít meg.

 Teljesítményjellemzők

| Művelet | Időbeli bonyolultság | Megjegyzések |
|---|---|---|
| computeSurprise | O(1) | Legfeljebb 1000 blokkot mintavételez |
| storeWithSurprise | O(1) amortizálva | Meglepetési számítás + hash map insert |
| evictLowSurpriseBlocks(k) | O(k × n) | Részleges rendezés k legalacsonyabb prioritású blokkhoz |
| getSurpriseRecord | O(1) átlagos | HashMap keresés |
| organizeByEntanglement | O(p²) ahol p ≤ 100 | Korlátozott pár-generálás |

---

 Alapfogalmak

 Meglepetés mint újdonság

A surprise memory rendszer az adatot értékesebbnek tekinti, ha újszerű (különbözik a meglévő adatoktól), nem redundáns. A „meglepetés" az új adat számszerűsített mértéke – azt mutatja, mennyire váratlan az adat ahhoz képest, amit a rendszer már látott. A magas meglepetési értékű adatot tovább tartja meg; az alacsony meglepetési értékűt ejti ki először.

 Több dimenziós meglepetési számítás

A meglepetés nem egyetlen metrika, hanem három független dimenzió összetétele, amelyek mindegyike az újdonság egy-egy aspektusát méri:

| Dimenzió | Cél | Számítás | Tartomány |
|---|---|---|---|
| Jaccard-eltávolítás | Tartalom szintű hasonlóság | 1 - (metszet/unió) bájt-készletek | 0.0 - 1.0 |
| Tartalmi hash-távolság | Kriptográfiai eltérés | Hamming-távolság SHA256 hash-en (128 bit) | 0.0 - 1.0 |
| Időbeli újdonság | Történelmi kontextus | 1 / √(1 + blokk_szám) | 0.0 - 1.0 |
| Kombinált meglepetés | Végső pontszám | A fenti három átlaga | 0.0 - 1.0 |

 Tartalmalapú tárolás integrációja

A rendszer integrálódik a ContentAddressableStorage-zal az automatikus deduplikáláshoz. Amikor storeWithSurprise() kerül hívásra, a rendszer ellenőrzi, hogy az adat létezik-e már, és frissíti a hozzáférési számlálót ahelyett, hogy duplán tárolná.

 Kapacitáskezelés és kizárás

Amikor a tárolókapacitás megtelik, a rendszer a legalacsonyabb megtartási prioritású blokkokat ejti ki. A kizárási folyamat egy részleges rendezési algoritmust használ a jelöltek hatékony azonosításához.

 A meglepetési küszöb és osztályozás

A rendszer egy konfigurálható surprise_threshold értéket (alapértelmezett 0.3) használ a blokkok osztályozásához:

| Osztályozás | Feltétel | Hatás |
|---|---|---|
| Magas meglepetés | combined_surprise > küszöb | Növeli a high_surprise_blocks számlálót |
| Alacsony meglepetés | combined_surprise ≤ küszöb | Növeli a low_surprise_blocks számlálót, elsők a kizárásra |

Invariáns: high_surprise_blocks + low_surprise_blocks ≤ total_blocks (mindig fennáll)

---

 Meglepetési metrikák

 

A meglepetési metrikák három független, egyenlően súlyozott dimenziót tartalmaznak, amelyek az újdonság különböző aspektusait ragadják meg: Jaccard-eltávolítás, tartalmi hash-távolság és időbeli újdonság.

 A SurpriseMetrics struktúra

| Mező | Típus | Tartomány | Leírás |
|---|---|---|---|
| jaccard_dissimilarity | f64/Rational | [0.0, 1.0] | Bájt-szintű tartalmi hasonlóság (1.0 = teljesen különböző) |
| content_hash_distance | f64/Rational | [0.0, 1.0] | Normalizált Hamming-távolság a tartalmi hash-ek között |
| temporal_novelty | f64/Rational | [0.0, 1.0] | A teljes blokk-szám inverze (csökken ahogy nő a tároló) |
| combined_surprise | f64/Rational | [0.0, 1.0] | A három metrika átlaga |

 1. dimenzió: Jaccard-eltávolítás

Cél: A Jaccard-eltávolítás bájt-szinten méri a tartalmi hasonlóságot. Két, hasonló bájt-eloszlású adatblokk alacsony eltávolítási pontszámot kap, jelezve a redundanciát. Ez a metrika a bájtok jelenlétére működik, nem a sorrendükre.

Algoritmus:
1. Legfeljebb JACCARD_SAMPLE_SIZE (1000) meglévő blokkot mintavételez
2. Minden blokkhoz 256 elemű boolean tömböt készít
3. Kiszámolja a metszetet és az uniót
4. Visszaadja: 1 - (metszet_szám / unió_szám) minimumát

| Tulajdonság | Érték |
|---|---|
| Tartomány | [0.0, 1.0] |
| Mintaméret | 1000 bájt |
| Összehasonlítási tér | 256 dimenzió |

 2. dimenzió: Tartalmi hash-távolság

Cél: Kriptográfiai eltérések detektálása. A SHA256 hash-eken számított Hamming-távolság segítségével méri az eltérést.

Algoritmus:
1. SHA256 hash számítása az új adatból, az első HASH_SIZE (16) bájt kivétele
2. Legfeljebb 1000 meglévő hash mintavételezése
3. XOR minden egyes bájtpáron, majd set-bitek számlálása (popCount)
4. Normalizálás: hamming_távolság / HASH_BITS (128)

Matematikai tulajdonságok (Lean-ben bizonyítva):

| Tulajdonság | Tétel |
|---|---|
| Szimmetria | computeHashDistance_symmetric |
| Azonosság | computeHashDistance_self_zero |
| Háromszög-egyenlőtlenség | computeHashDistance_triangle_rational |
| Nem-negativitás | computeHashDistance_nonneg |
| Korlátosság | computeHashDistance_bounded_num |

 3. dimenzió: Időbeli újdonság

Cél: Az időbeli kontextus figyelembevétele – a rendszer „fiatalkorában" (kevés blokk) érkező adatok eleve újszerűbbek.

Képlet:

temporal_novelty = 1 / (1 + sqrt(block_count))


| Blokk-szám | Időbeli újdonság |
|---|---|
| 0 | 1.000 |
| 1 | 0.707 |
| 9 | 0.500 |
| 99 | 0.091 |
| 999 | 0.031 |

 Kombinált meglepetési pontszám


combined_surprise = (jaccard_dissimilarity + content_hash_distance + temporal_novelty) / 3.0


Az egyenlő súlyozás biztosítja, hogy egyik dimenzió se domináljon.

 Számítási bonyolultság

| Metrika | Időbeli bonyolultság | Térbeli bonyolultság |
|---|---|---|
| Jaccard | O(1000 × 256) = O(1) | O(256) |
| Hash-távolság | O(1000 × 16) = O(1) | O(16) |
| Időbeli | O(1) | O(1) |
| Összesen | O(1) | O(1000) |

 Állandók referenciája

| Állandó | Érték | Cél |
|---|---|---|
| JACCARD_SAMPLE_SIZE | 1000 | Maximum mintavételezett blokkok száma |
| HASH_SIZE | 16 | Hash mérete bájtban (128 bit) |
| HASH_BITS | 128 | Hash-távolság normalizáláshoz |
| MAX_INPUT_SIZE | 100 MB | Maximum elfogadott adatméret |
| DEFAULT_SURPRISE_THRESHOLD | 0.3 | Alapértelmezett küszöb |

---

 Megtartási prioritás

 Képlet és összetevők

A megtartási prioritás három tényező súlyozott kombinációja:


retention_priority = surprise_score × (base_weight + age_component + frequency_component)

ahol:
  base_weight         = 0.5
  age_component       = 0.3 × age_factor
  frequency_component = 0.2 × log(access_frequency)
  age_factor          = 1.0 / (1.0 + time_since_last_access_ms)


 Súlyelosztás

| Állandó | Érték | Arány | Cél |
|---|---|---|---|
| RETENTION_BASE_WEIGHT | 0.5 | 50% | Biztosítja a meglepetési pontszám dominanciáját |
| RETENTION_AGE_WEIGHT | 0.3 | 30% | Jutalmazza a nemrég elért blokkokat |
| RETENTION_FREQUENCY_WEIGHT | 0.2 | 20% | Jutalmazza a sűrűn elért blokkokat |

 Kor-tényező: Frissességi számítás


age_factor = 1.0 / (1.0 + age_ms)


- Azonnali hozzáférés (age_ms = 0): age_factor = 1.0
- 1 másodperces kor: age_factor ≈ 0.5
- Nagyon régi blokkok: age_factor közelít a nullához

 Frekvencia-tényező: logaritmikus skálázás

| Hozzáférési frekvencia | log(frekvencia) | Frekvencia-hozzájárulás (0.2 × log) |
|---|---|---|
| 1 (kezdeti) | 0.0 | 0.0 |
| 2 | 0.693 | 0.139 |
| 10 | 2.303 | 0.461 |
| 100 | 4.605 | 0.921 |

 Dinamikus prioritás-frissítések

A megtartási prioritás nem statikus – változik a blokkok elérésekor és az idő múlásával.

- Hozzáférésen frissítve: Az access_frequency nő, a last_access_time az aktuális időre frissül
- Kizárás előtt frissítve: Minden rekord prioritása frissítésre kerül az igazságos összehasonlításhoz

 Formális tulajdonságok (Lean-ben bizonyítva)

- access_frequency szigorúan monoton növekvő
- surprise_score és creation_time nem változik inicializálás után
- Az invariáns garantálja: access_frequency ≥ 1 és creation_time ≤ last_access_time

 Példák a prioritás alakulására

Inicializáláskor (t=0, surprise=0.8):

age_factor = 1.0
frequency_factor = log(1) = 0.0
weight_sum = 0.5 + 0.3×1.0 + 0.2×0.0 = 0.8
retention_priority = 0.8 × 0.8 = 0.64


10 hozzáférés után, azonnali eléréskor:

frequency_factor = log(10) ≈ 2.303
weight_sum = 0.5 + 0.3×1.0 + 0.2×2.303 = 1.261
retention_priority = 0.8 × 1.261 ≈ 1.009


1 perccel az utolsó hozzáférés után:

age_ms = 55000
age_factor = 1 / (1 + 55000) ≈ 0.0000182
retention_priority ≈ 0.769 (lecsökkent 1.009-ről)


---

 Tartalomalapú tárolás

 Alaparchitektúra

A tartalomalapú tárolás (CAS) az a perzisztencia réteg, amely blokktárolást, automatikus deduplikálást és tartalom-alapú azonosítást biztosít. Az adatblokkok tartalmukhoz kapcsolódó kriptográfiai hash-szel kerülnek azonosításra, így az azonos tartalom mindig ugyanahhoz az azonosítóhoz vezet.

 Tartalomalapú azonosítás

| Implementáció | Hash-algoritmus | Kimeneti méret | BlockId kinyerés |
|---|---|---|---|
| Zig | SHA-256 | 32 bájt | Első 16 bájt |
| Lean | FNV-szerű hash | Levezetett | 16 bájton elosztva |

 Tárolási műveletek

| Művelet | Zig szignatúra | Lean szignatúra | Cél |
|---|---|---|---|
| store | store(data, core) → !BlockId | store(data) → (StorageState, BlockId) | Adat tárolása, tartalom-hash visszaadása |
| retrieveByContent | retrieveByContent(data) → ?BlockId | findByContent(data) → Option BlockId | BlockId keresése tartalom alapján |
| containsBlock | containsBlock(bid) → bool | containsBlock(bid) → Bool | Blokk létezésének ellenőrzése |
| removeBlock | removeBlock(bid) | removeBlock(bid) → StorageState | Blokk törlése |

 Automatikus deduplikálás

A tartalomalapú tárolás garantálja, hogy az azonos adat csak egyszer kerül tárolásra. Ha a store() hívásra kerül már létező tartalommal, a meglévő BlockId-t adja vissza.

 Lean: Lista-alapú tárolás


structure StorageState where
  blocks : List StorageBlock
  capacity : Nat


A műveletek új StorageState értékeket adnak vissza ahelyett, hogy helyben módosítanák az állapotot – ez megőrzi az immutabilitást a formális ellenőrzés érdekében.

 Formális tulajdonságok (Lean-ben bizonyítva)

| Tétel | Állítás |
|---|---|
| storageState_empty_count | Üres tárolóban nulla blokk |
| storageState_store_count | A tárolás növeli a számlálót 1-gyel |
| computeContentHash_deterministic | A hash determinisztikus |
| storage_remove_decreases_count | Eltávolítás nem növeli a számlálót |

 Összefonódás (Entanglement)

A tartalomalapú tárolás támogat entangleBlocks() műveletet, amely kapcsolatokat hoz létre a magas meglepetési értékű blokkok között. Ez az összefonódás a tároló háttérrendszer általi optimalizáláshoz használható.

---

 Kizárási stratégia

 

A kizárás egy kapacitáskezelő mechanizmus, amely eltávolítja a legalacsonyabb megtartási prioritású blokkokat, amikor az aktuális tárolási szám meghaladja a célkapacitást. A rendszer részleges rendezési algoritmust használ a k legalacsonyabb prioritású blokk hatékony azonosításához.

 A kizárási folyamat

1. Kapacitás-ellenőrzés: Ha jelenlegi_méret ≤ célkapacitás, azonnal visszatér 0 kizárással
2. Jelöltlista összeállítása: CandidateItem lista készítése az összes rekord aktuális prioritásával
3. Részleges rendezés: partialSort(jelöltek, k) hívása a k legalacsonyabb prioritású blokk megtalálásához
4. Eltávolítási ciklus: Minden jelölt eltávolítása a tárolóból és a rekordokból
5. Szám visszaadása: A ténylegesen kizárt blokkok számának visszaadása

 A részleges rendezési algoritmus

Kiválasztás-rendezés k iterációra:


Kezdeti állapot:    [5.0, 1.0, 3.0, 2.0, 4.0]
i=0 után:           [1.0, 5.0, 3.0, 2.0, 4.0]  ← minimum helyre kerül
i=1 után:           [1.0, 2.0, 3.0, 5.0, 4.0]  ← második minimum
i=2 után:           [1.0, 2.0, 3.0, 5.0, 4.0]  ← harmadik minimum
Kizárt: indexek [0, 1, 2] → prioritások [1.0, 2.0, 3.0]


| Tulajdonság | Érték |
|---|---|
| Időbeli bonyolultság | O(k×n) |
| Térbeli bonyolultság | O(1) |
| Stabilitás | Nem stabil |

 Kizárás biztonsági garanciák (Lean-ben bizonyítva)

| Tétel | Garancia |
|---|---|
| evictLowSurprise_noop_under_capacity | Kapacitás alatt nincs kizárás |
| eviction_never_removes_max_priority | A maximális prioritású blokk sosem kerül kizárásra |
| evictLowSurprise_zero_evictions_under_capacity | Kapacitás alatt 0 kizárás |

 Statisztikák frissítése kizáráskor

| Mező | Módosítás |
|---|---|
| total_blocks | 1-gyel csökkentve |
| high_surprise_blocks | Csökkentve (ha pontszám > küszöb) |
| low_surprise_blocks | Csökkentve (ha pontszám ≤ küszöb) |
| total_surprise_sum | surprise_score-szal csökkentve |
| evictions_due_to_low_surprise | 1-gyel növelve |

 Teljesítményelemzés

| Művelet | Bonyolultság |
|---|---|
| Teljes kizárás | O(k×n + n) |
| Jelöltek összeállítása | O(n) |
| Prioritások frissítése | O(n) |
| Részleges rendezés | O(k×n) |
| Blokkok eltávolítása | O(k×log n) |

Részleges rendezés előnye: n=10 000, k=1 000 esetén ~ 10M vs. teljes rendezésnél ~130K - a részleges rendezés hatékonyabb, ha k << n.

---

 Rendszerarchitektúra

 Kettős implementációs architektúra

A SurpriseMemory kettős implementációs stratégiát alkalmaz: a Zig implementáció (SurpriseMemoryManager) mutálható állapotot biztosít hash map-ekkel és mutex-szinkronizációval, a Lean implementáció (ManagerState) pedig tiszta funkcionális adatstruktúrákat immutábilis listákkal.

 Komponens-architektúra

Az öt fő komponens-réteg:

- Manager Core (SurpriseMemoryManager/ManagerState): Koordinálja az összes műveletet
- Metaadat-réteg: Párhuzamos adatstruktúrák – surprise_records és statistics
- Meglepetési motor: Több dimenziós újdonság számítása
- Irányvonal-motor: Küszöb-konfiguráció, megtartási prioritás, kizárási jelölt-kiválasztás
- Tárolási háttérrendszer: ContentAddressableStorage és DataFlowAnalyzer integrációja

 Adatfolyam-architektúra

Tárolási műveleti folyamat (storeWithSurprise):
1. Mutex megszerzése
2. Tartalom ellenőrzése: storage.retrieveByContent(data)
3. Meglévő blokk esetén: hozzáférés frissítése
4. Új blokk esetén: computeSurprise(data) → tárolás → SurpriseRecord.init → statisztika frissítése
5. Mutex felszabadítása

Kizárási műveleti folyamat (evictLowSurpriseBlocks):
1. Kapacitás-ellenőrzés
2. Jelöltlista összeállítása
3. Részleges rendezés k legalacsonyabb prioritású blokkra
4. Eltávolítási ciklus (storage + records + statistics)
5. Kizárt darabszám visszaadása

 Menetbiztonság és párhuzamossági modell

A Zig implementáció szálbiztos, míg a Lean egyetlen szálú tiszta funkcionális modellt alkalmaz. Minden publikus metódus követi a mintát:
zig
self.mutex.lock();
defer self.mutex.unlock();
// ... kritikus szekció ...


 Konfiguráció és állandók

| Állandó | Érték | Cél |
|---|---|---|
| HASH_SIZE | 16 bájt | BlockId mérete |
| HASH_BITS | 128 bit | Hash-távolság normalizálása |
| MAX_INPUT_SIZE | 100 MB | Maximum blokk-adatméret |
| JACCARD_SAMPLE_SIZE | 1000 blokk | Maximum mintavételezés hasonlósághoz |
| MAX_ENTANGLEMENT_PAIRS | 100 blokk | Maximális összefonodási párok |
| DEFAULT_SURPRISE_THRESHOLD | 0.3 | Alapértelmezett küszöb |
| RETENTION_BASE_WEIGHT | 0.5 | Meglepetési pontszám súlya |
| RETENTION_AGE_WEIGHT | 0.3 | Frissességi súly |
| RETENTION_FREQUENCY_WEIGHT | 0.2 | Frekvencia-súly |

---

 Komponens-

 SurpriseMemoryManager (Zig) / ManagerState (Lean)

Zig mezők:

| Mező | Típus | Felelősség |
|---|---|---|
| storage | ContentAddressableStorage | Tartalomalapú blokkfárolás mutatója |
| flow_analyzer | DataFlowAnalyzer | Adatfolyam-elemző mutatója |
| surprise_records | HashMap(...) | Blokk ID-k leképezése SurpriseRecord-okra |
| surprise_threshold | f64 | Osztályozási határérték (0.0-1.0) |
| statistics | SurpriseMemoryStatistics | Összesített metrikák nyomkövetője |
| allocator | Allocator | Memória-foglaló |
| mutex | Mutex | Szál-szinkronizáló |
| owns_storage | bool | Jelzőbit a tárolótulajdonláshoz |
| owns_analyzer | bool | Jelzőbit az elemzőtulajdonláshoz |

Lean mezők:

| Mező | Típus | Felelősség |
|---|---|---|
| storage | StorageState | Tiszta funkcionális tárolóstruktúra |
| surprise_records | List (BlockId × SurpriseRecord) | Párosítási lista |
| surprise_threshold | Rational | Osztályozási határérték racionális számként |
| statistics | SurpriseMemoryStatistics | Összesített metrikák |

 SurpriseRecord

| Mező | Frissítési viselkedés |
|---|---|
| block_id | Immutábilis inicializálás után |
| surprise_score | Immutábilis – egyszer kerül beállításra tároláskor |
| creation_time | Immutábilis – soha nem változik |
| last_access_time | Mutábilis – hozzáférésenként frissítve |
| retention_priority | Mutábilis – hozzáféréskor és frissítéskor újraszámított |
| access_frequency | Mutábilis – hozzáférésenként növekvő |

 SurpriseMemoryStatistics

| Mező | Típus | Cél | Invariáns |
|---|---|---|---|
| total_blocks | usize | Összes tárolt blokk | = high + low |
| high_surprise_blocks | usize | Küszöb feletti blokkok | ≤ total_blocks |
| low_surprise_blocks | usize | Küszöb alatti blokkok | ≤ total_blocks |
| average_surprise | f64 | Átlagos meglepetés | total_sum / total |
| evictions_due_to_low_surprise | usize | Kumulatív kizárások | Monoton növő |

---

 Meglepetési számítási folyamat

 A folyamat architektúrája

A meglepetési számítási folyamat átalakítja a nyers bemeneti adatot egy több dimenziós újdonsági pontszámmá (SurpriseMetrics).

 Bemeneti érvényesítés

| Ellenőrzés | Küszöb | Hiba |
|---|---|---|
| Méretkorlát | MAX_INPUT_SIZE (100 MB) | error.InputTooLarge |
| NaN ellenőrzés | Float konverzió érvényessége | error.InvalidInput |
| Üres tároló | block_count == 0 | Korai visszatérés max. meglepetéssel |

 Blokkfájl mintavételezési stratégia

A számítási bonyolultság O(1)-en tartásához a folyamat legfeljebb JACCARD_SAMPLE_SIZE (1000) blokkot mintavételez összehasonlításra.

 A folyamat végrehajtási lépései

| Lépés | Művelet |
|---|---|
| 1 | Bemeneti méret érvényesítése |
| 2 | Üres tároló ellenőrzése |
| 3 | Tartalmi hash számítása |
| 4 | Minimális távolságok inicializálása |
| 5 | Blokk ID-k összegyűjtése és rendezése |
| 6 | Mintavételezett blokkok iterálása |
| 7 | Jaccard-távolság számítása |
| 8 | Hash-távolság számítása |
| 9 | Minimumok nyomkövetése |
| 10 | Időbeli újdonság számítása |
| 11 | Metrikák kombinálása |
| 12 | SurpriseMetrics visszaadása |

 Zig vs. Lean különbségek

| Szempont | Zig implementáció | Lean implementáció |
|---|---|---|
| Hash-funkció | SHA-256 (kriptográfiai) | FNV-szerű (egyszerű) |
| Numerikus típus | f64 (lebegőpontos) | Rational (pontos) |
| Cél | Éles teljesítmény | Formális ellenőrzés |

---

 Az adatok életciklusa

 Állapotgép-

Minden adatblokk a rendszerben meghatározott állapotsorozaton halad át: Tárolt → Aktív → Kizárásra Jelölt / Összefonódott → Kizárt

 Kezdeti tárolás és értékelés

A storeWithSurprise() az összes adat belépési pontja, mind az új blokkok, mind a meglévő blokkok eléréséhez. Három eset lehetséges:
1. Azonos tartalom + meglévő rekord: hozzáférés frissítése
2. Azonos tartalom + nincs rekord: új rekord létrehozása számított meglepetéssel
3. Teljesen új tartalom: meglepetés számítása, tárolás, rekord és statisztika frissítése

 Aktív állapot kezelése

A tárolt blokkok aktív állapotba kerülnek, ahol a megtartási prioritásuk dinamikusan frissül a hozzáférési minták alapján.

Hozzáférési minta bizonyítékok:

| Tétel | Garancia |
|---|---|
| surpriseRecord_recordAccess_frequency | access_frequency pontosan 1-gyel nő hozzáférésenként |
| surpriseRecord_recordAccess_preserves_score | surprise_score nem változik |
| surpriseRecord_recordAccess_preserves_block_id | block_id immutábilis |

 Az összefonódott állapot

A magas meglepetési értékű blokkok Összefonódott állapotba kerülhetnek, ahol párosítják őket más magas meglepetési értékű blokkokkal a tárolási háttérrendszer általi optimalizáláshoz. Az összefonódás nem befolyásolja a megtartási prioritás számítást; az összefonódott blokkok is kizárhatók, ha alacsony a prioritásuk.

 A terminális Kizárt állapot

Miután egy blokk Kizárt állapotba kerül, ez terminális – a blokk nem állítható vissza, és nem mehet át más állapotba.

Kizárás utáni műveletek:
1. Tároló eltávolítás: storage.removeBlock(block_id)
2. Rekord eltávolítás: surprise_records.remove(block_id)
3. Statisztika frissítés: statistics.removeBlock() és evictions_due_to_low_surprise++

 Életciklus statisztika nyomkövetés

| Mező | Mikor frissül | Cél |
|---|---|---|
| total_blocks | Tároláskor, kizáráskor | Aktuális összblokk-szám |
| novel_block_allocations | Tároláskor (új + magas meglepetés) | Újszerű blokkok száma |
| evictions_due_to_low_surprise | Kizáráskor | Összes kizárás |

---

 Tárolás-integráció

 Integrációs architektúra

A SurpriseMemoryManager nem valósítja meg a saját tárolási rétegét. Ehelyett két külső komponenssel integrálódik a chaos_core modulból:
- ContentAddressableStorage: Alapvető blokkfárolási képességek
- DataFlowAnalyzer: Adatfolyam-elemzési képességek

 Hozzáférési minták

| Hozzáférési minta | Metódus | Cél |
|---|---|---|
| Közvetlen mező-hozzáférés | self.storage.storage | Iterátor-hozzáférés, számlálás |
| Metódus-delegálás | self.storage.store(), stb. | Tárolási műveletek |

 Tulajdonlási minták

| Minta | Függvény | owns_storage | owns_analyzer | Takarítási felelősség |
|---|---|---|---|---|
| Külső tulajdon | init() | false | false | Hívó köteles |
| Belső tulajdon | initWithOwnership() | Konfigurálható | Konfigurálható | Manager hívja deinit()-et |

---

 Zig implementáció

 Implementációs 

A Zig implementáció a surprise_memory.zig fájlban található, és egy szálbiztos, éles üzemeltetésre alkalmas memóriakezelő rendszert biztosít, amely a SurpriseMemoryManager struct köré épül.

 Kulcsállandók

| Állandó | Érték | Cél |
|---|---|---|
| RETENTION_AGE_WEIGHT | 0.3 | Frissességi súly |
| RETENTION_FREQUENCY_WEIGHT | 0.2 | Hozzáférési frekvencia súlya |
| RETENTION_BASE_WEIGHT | 0.5 | Meglepetési pontszám alapsúlya |
| HASH_SIZE | 16 | Blokkazonosítók mérete (128 bit) |
| MAX_INPUT_SIZE | 100 MB | Maximum megengedett adatblokk-méret |
| JACCARD_SAMPLE_SIZE | 1000 | Maximum mintavételezett blokkok száma |
| MAX_ENTANGLEMENT_PAIRS | 100 | Maximum magas-meglepetési blokkok az összefonódáshoz |
| DEFAULT_SURPRISE_THRESHOLD | 0.3 | Alapértelmezett küszöb |

 A meglepetési számítási folyamat

A computeSurprise metódus a mag újdonság-detektálási algoritmust valósítja meg. Legfeljebb 1000 meglévő blokkot mintavételez a számítási bonyolultság O(1000)-en tartásához.

Jaccard-távolság számítás: Bájt-jelenlét készleteket használ, amelyek egyensúlyt teremtenek a tartalmi különbségekre való érzékenység és a számítási hatékonyság között.

 Tesztelési infrastruktúra

| Teszt neve | Lefedettség |
|---|---|
| surprise_memory_basic | Végponttól végpontig terjedő manager-használat |
| surprise_metrics_validation | Metrika-korlátolási viselkedés |
| surprise_record_retention | Megtartási prioritás-frissítések |
| statistics_incremental_update | Statisztika-nyomkövetés helyessége |
| hash_distance_calculation | Hash-távolság metrikai tulajdonságai |
| partial_sort_correctness | Részleges rendezési algoritmus validációja |

---

 SurpriseMemoryManager

 Cél és felelősségek

A SurpriseMemoryManager struct a Zig implementáció central koordinátora, amely kezeli az integrációt a tartalomalapú tárolás, a meglepetési számítás, a rekord-nyomkövetés és a kapacitáskezelés között.

 Inicializálás

init(allocator, storage, analyzer): Nem-tulajdonló hivatkozásokkal inicializál. A hívó felelős a storage és analyzer deiniializálásáért.

initWithOwnership(allocator, storage, analyzer, owns_storage, owns_analyzer): Konfigurálható tulajdonlással inicializál. Ha owns_storage = true, a manager meghívja a storage.deinit()-et a saját takarításakor.

 Tulajdonlási forgatókönyvek

| Forgatókönyv | owns_storage | owns_analyzer | Felhasználási eset |
|---|---|---|---|
| Nem-tulajdonló | false | false | Megosztott komponensek |
| Tároló-tulajdonló | true | false | Manager vezérli a tároló életciklusát |
| Teljesen-tulajdonló | true | true | Manager minden komponensét tisztítja |

 Kapcsolat a Lean specifikációval

| Szempont | Zig implementáció | Lean specifikáció |
|---|---|---|
| Név | SurpriseMemoryManager | ManagerState |
| Mutáció | Mutábilis struct metódusokkal | Tiszta funkcionális műveletek |
| Menetbiztonság | Mutex-védett | Nem alkalmazható |
| HashMap | std.HashMap egyedi kontextussal | List (BlockId × SurpriseRecord) |

---

 Menetbiztonság és párhuzamosság

 Mutex-alapú szinkronizáció

A SurpriseMemoryManager egyetlen std.Thread.Mutex-et használ az összes mutábilis állapothoz. Ez a durva szemcsézettségű zárolási stratégia biztosítja, hogy a megosztott adatokon végzett összes művelet szerializált legyen.

 Zárolás-megszerzési minta

zig
self.mutex.lock();
defer self.mutex.unlock();
// ... kritikus szekció ...


Ez a minta biztosítja, hogy a mutex mindig felszabadul a függvény kilépésekor, függetlenül a kilépési útvonaltól.

 Párhuzamos hozzáférési minták

| Művelet 1 | Művelet 2 | Párhuzamos? |
|---|---|---|
| storeWithSurprise | storeWithSurprise | ❌ Szerializált |
| storeWithSurprise | evictLowSurpriseBlocks | ❌ Szerializált |
| getSurpriseRecord | getSurpriseRecord | ❌ Szerializált |
| getStatistics | getStatistics | ❌ Szerializált |

 Zárolási versengési forgatókönyvek

A leghosszabb kritikus szekciók:
1. evictLowSurpriseBlocks: O(k×n) részleges rendezés
2. organizeByEntanglement: O(n²) páros összehasonlítás
3. storeWithSurprise: O(1000) mintaalapú meglepetési számítás

 Nem-újrabelépős dizájn

Az std.Thread.Mutex nem újrabelépős. A jelenlegi dizájn elkerüli ezt azzal, hogy:
- A privát metódusok nem zárolnak; feltételezik, hogy a hívó tartja a zárat
- Nincsenek visszahívási mechanizmusok
- Nincsenek külső hívások, amelyek visszahívhatnák a managert

---

 Publikus API referencia

 Inicializálás és életciklus

| Függvény | Visszatérési értéke | Leírás |
|---|---|---|
| init(allocator, storage, analyzer) | SurpriseMemoryManager | Inicializálás tulajdon nélkül |
| initWithOwnership(...) | SurpriseMemoryManager | Konfigurálható tulajdonlású inicializálás |
| deinit() | void | Manager és opcionálisan a függőségek deiniializálása |

 Fő műveletek

storeWithSurprise(data, preferred_core) → ![HASH_SIZE]u8
- Adatblokkot tárol automatikus meglepetési számítással és deduplikálással
- Hibák: error.InputTooLarge (>100MB), tárolási hibák

computeSurprise(new_data) → !SurpriseMetrics
- Meglepetési metrikákat számít anélkül, hogy tárolná az adatot
- Hibák: error.InputTooLarge, error.InvalidInput

evictLowSurpriseBlocks(target_capacity) → !usize
- Kizárja a legalacsonyabb megtartási prioritású blokkokat a célkapacitás eléréséig
- Visszatér a ténylegesen kizárt blokkok számával

 Lekérdezési műveletek

| Függvény | Visszatérési értéke | Leírás |
|---|---|---|
| getStatistics() | SurpriseMemoryStatistics | Aktuális rendszerstatisztikák |
| getSurpriseRecord(block_id) | ?SurpriseRecord | Blokk metaadat-rekordja |
| containsRecord(block_id) | bool | Rekord létezésének ellenőrzése |
| getRecordCount() | usize | Nyomkövetett rekordok száma |

 Konfigurációs műveletek

| Függvény | Leírás |
|---|---|
| setSurpriseThreshold(threshold) | Küszöb frissítése és meglévő blokkok újraosztályozása |
| getSurpriseThreshold() | Aktuális küszöb lekérdezése |

 Karbantartási műveletek

organizeByEntanglement() → !usize
- Összefonódott párokat hoz létre a magas meglepetési értékű blokkokból
- Visszaadja a létrehozott párok számát

 Hibaesetek

| Függvény | Hiba | Feltétel |
|---|---|---|
| storeWithSurprise | error.InputTooLarge | data.len > 100MB |
| computeSurprise | error.InvalidInput | NaN detektálva |
| evictLowSurpriseBlocks | error.OutOfMemory | Allokációs hiba |

---

 Integrációs útmutató

 Előfeltételek

A SurpriseMemoryManager integrálása előtt az alkalmazásnak hozzáféréssel kell rendelkeznie a következő függőségekhez a chaos_core.zig-ből:
- ContentAddressableStorage: Blokkperzisztencia és deduplikáció
- DataFlowAnalyzer: Adatfolyam-elemzési képességek
- MemoryBlock: Blokkadatstruktúra
- BlockIdContext: Hash map kontextus a blokk ID-khez

 Tulajdonlási döntési útmutató

1. Manager mindkettőt tulajdonolja (true, true): A manager az egyetlen felhasználó mindkét függőségből
2. Alkalmazás mindkettőt tulajdonolja (false, false): A tároló vagy elemző több manager vagy más alrendszer között oszlik meg
3. Vegyes tulajdonlás: Ritka, de lehetséges
4. Életciklus szabály: A tulajdonosnak tovább kell élnie, mint az összes felhasználónak

 Menetbiztonság szempontjai

| Garancia | Mechanizmus | Hatókör |
|---|---|---|
| Kizárás | std.Thread.Mutex | Minden publikus metódus |
| Rekord-konzisztencia | Zárolt olvasás/írás | Műveleti atomitás |
| Statisztika-konzisztencia | Zárolt frissítések | Összesített metrikák |

 Integrációs ellenőrzőlista

- Függőségek megfelelően inicializálva
- Tulajdonlási modell egyértelműen meghatározva
- Hibakezelés lefedi az összes esetett
- Kapacitáskezelési stratégia implementálva
- Statisztika-monitorozás helyen van
- Takarítási sorrend helyes
- Bemenet-validálás megakadályozza a >100MB adatokat

---

 Adatstruktúrák

 SurpriseMetrics

| Mező | Típus | Tartomány | Leírás |
|---|---|---|---|
| jaccard_dissimilarity | f64 | [0.0, 1.0] | Bájt-szintű tartalmi hasonlóság |
| content_hash_distance | f64 | [0.0, 1.0] | Normalizált Hamming-távolság |
| temporal_novelty | f64 | [0.0, 1.0] | Időbeli frissességi metrika |
| combined_surprise | f64 | [0.0, 1.0] | A három metrika átlaga |

Metódusok:
- init(jaccard, hash_dist, temporal): Konstruktor, amely [0.0, 1.0]-ra korlátoz és számítja a kombinált pontszámot
- exceedsThreshold(threshold): true, ha combined_surprise > threshold

 SurpriseRecord

| Mező | Típus | Leírás |
|---|---|---|
| block_id | [HASH_SIZE]u8 | Tartalom-hash azonosító (16 bájt) |
| surprise_score | f64 | Immutábilis kezdeti meglepetési pontszám |
| creation_time | i128 | Nanomásodperces timestamp |
| last_access_time | i128 | Legutóbbi hozzáférés timestampje |
| retention_priority | f64 | Dinamikusan számított kizárási prioritás |
| access_frequency | usize | Összes hozzáférés száma |

 CandidateItem

Belső struktúra a kizárási folyamat során:

| Mező | Típus | Leírás |
|---|---|---|
| block_id | [HASH_SIZE]u8 | Potenciálisan kizárandó blokk azonosítója |
| priority | f64 | Gyorsítótárazott megtartási prioritás |

 Memóriakiosztás blokkonként


SurpriseRecord = {
  block_id: [16]u8        = 16 bájt
  surprise_score: f64     =  8 bájt
  creation_time: i128     = 16 bájt
  last_access_time: i128  = 16 bájt
  retention_priority: f64 =  8 bájt
  access_frequency: usize =  8 bájt
  ──────────────────────────────────
  Összesen/rekord         = 72 bájt
}
HashMap overhead-del:    ~144 bájt/blokk


---

 Lean specifikáció

 Cél és hatókör

A Lean 4 specifikáció matematikailag ellenőrzi a surprise memory rendszer helyességét. Ez egy tiszta funkcionális implementáció, amelynek középpontjában a matematikai korrektség és bizonyítható tulajdonságok állnak. A specifikáció tartalmaz 271+ bizonyított tételt.

 

| Szempont | Részletek |
|---|---|
| Nyelv | Lean 4 tételbizonyító |
| Méret | 2374 sornyi kód |
| Tételek száma | 271+ |
| Megközelítés | Tiszta funkcionális, mellékhatások nélkül |
| Ellenőrzési cél | Zig implementáció |
| Függőségek | Nincs (önállóan zárt) |

 Kulcsstruktúrák leképezése

| Lean struktúra | Zig megfelelője | Cél |
|---|---|---|
| SurpriseMetrics | SurpriseMetrics | Több dimenziós újdonsági pontszámok |
| SurpriseRecord | SurpriseRecord | Blokkonkénti metaadat |
| StorageState | ContentAddressableStorage | Blokkfárolási réteg |
| ManagerState | SurpriseMemoryManager | Legfelső szintű rendszerállapot |
| SurpriseMemoryStatistics | SurpriseMemoryStatistics | Összesített metrikák |

 A racionális számrendszer

A Rational típus számlálóból és nevezőből áll, ahol bizonyíték garantálja a nevező pozitivitását. Ez típusszinten zárja ki a nullával való osztást.

 Implementációs különbségek

| Szempont | Lean modell | Zig implementáció |
|---|---|---|
| Adatstruktúrák | List a rekordokhoz és tárolóhoz | HashMap a rekordokhoz |
| Számok | Rational (pontos) | f64 (lebegőpontos) |
| Tisztaság | Tiszta funkciók, immutábilis állapot | Mutábilis állapot mutex-szal |
| Fókusz | Bizonyítható korrektség | Teljesítmény és menetbiztonság |

 Az ellenőrzési munkafolyamat

1. Definiálás Lean-ben: Funkcionális megvalósítás ManagerState-ben
2. Invariánsok megadása: Definiálni, milyen tulajdonságok kell teljesüljenek
3. Megőrzés bizonyítása: A művelet megőrzi az invariánsokat
4. Zig implementálása: Teljesítménycentrikus verzió ugyanolyan szemantikával
5. Kézi ellenőrzés: Zig megfelel a Lean specifikációnak

 Lean hatáskörön kívüli területek

A Lean specifikáció nem modellez:
- Párhuzamosságot, versenyhelyzet-feltételeket
- Memória-allokációt
- Külső rendszereket (ContentAddressableStorage, DataFlowAnalyzer)
- Valós teljesítmény-jellemzőket

---

 Formális modell

 Racionális számrendszer

A formális modell Rational típust használ lebegőpontos aritmetika helyett a pontos számítások és a formális bizonyítások érdekében.

 Blokkidentifikátorok

Rögzített méretű bájt-tömbökként definiálva a Zig implementáció hash-alapú azonosítójának tükrözéseként. Rendelkezik BEq, DecidableEq és Hashable instance-okkal.

 StorageState struktúra


structure StorageState where
  blocks : List StorageBlock
  capacity : Nat


A tároló blokklistáként modellezett hash map helyett, egyszerűsítve a bizonyításokat.

 ManagerState struktúra


structure ManagerState where
  storage         : StorageState
  surprise_records : List (BlockId × SurpriseRecord)
  surprise_threshold : Rational
  statistics      : SurpriseMemoryStatistics


A rekordok asszociációs listaként tárolódnak. A putRecord szűri az azonos blokk ID-vel rendelkező meglévő rekordot, mielőtt hozzáadná az újat.

 Funkcionális műveleti modell

Az összes művelet tiszta függvényként kerül definiálva, amelyek átalakítják a rendszerállapotot. A hibakezelés Except StoreError segítségével valósul meg.

 Kizárási algoritmus Lean-ben

A Lean implementáció teljes mergeSort-ot és take k-t használ egyszerűség kedvéért a bizonyításokban:


def partialSortCandidates (items : List CandidateItem) (k : Nat) : List CandidateItem :=
  let sorted := items.mergeSort (...)
  sorted.take k


---

 Invariáns-rendszer

 Invariáns-hierarchia

A rendszer négy szintű invariáns-hierarchiát alkalmaz:


StatisticsInvariant
       ↓
RetentionInvariant
       ↓
ManagerInvariant
       ↓
TraceInvariant


 StatisticsInvariant

Tulajdonságok:

| Tulajdonság | Garantálja |
|---|---|
| partition_le | high_surprise_blocks + low_surprise_blocks ≤ total_blocks |
| blocks_consistent | Ha total_blocks = 0, akkor high = 0 és low = 0 |

Megőrzési tételek:

| Művelet | Megőrzési tétel |
|---|---|
| Blokk hozzáadása | statistics_invariant_addBlock |
| Kötegelt hozzáadás | statistics_invariant_preserved_by_addBlock_sequence |

 RetentionInvariant

Tulajdonságok:

| Tulajdonság | Garancia | Indok |
|---|---|---|
| access_positive | Hozzáférési frekvencia ≥ 1 | A rekord csak tárolás után létezik |
| creation_before_access | Létrehozási idő nem lehet a hozzáférési idő után | Az idő előre halad |

 ManagerInvariant

| Összetevő | Tulajdonság | Garancia |
|---|---|---|
| stats_inv | Tartalmaz StatisticsInvariant-ot | Statisztikák konzisztensek |
| threshold_valid | Küszöb korlátai | 0 ≤ surprise_threshold ≤ 1 |

 TraceInvariant

A legfelső szintű invariáns, amely biztosítja a rendszer szintű korrektséget a műveletek sorozatán át.

 Helyességi garanciák az invariánsokból

1. Statisztikai konzisztencia: High + Low soha nem haladja meg az összes blokkot
2. Nincs negatív szám: Lean Nat típusa garantálja
3. Üres állapot konzisztencia: Üres esetén minden kategória nulla
4. Időbeli monotonitás: access_frequency mindig ≥ 1 és sosem csökken
5. Időrendi sorrend: creation_time ≤ last_access_time
6. Küszöb érvényessége: Mindig [0, 1]-ben van

---

 Metrikus tér bizonyítások

 A metrikus tér axiómái

Egy metrikus tér d: X × X → ℝ távolságfüggvényt igényel, amely kielégíti a négy axiómát. A Lean specifikáció mindet bizonyítja a computeHashDistance függvényre BlockId értékeken.

 Hash-távolság implementáció

| Összetevő | Típus | Leírás |
|---|---|---|
| BlockId | Array UInt8 | 16 bájtos tartalom-hash |
| hammingDistNat | BlockId → BlockId → Nat | Nyers Hamming-távolság |
| computeHashDistance | BlockId → BlockId → Rational | Normalizált távolság: hamming / HASH_BITS |

 Szimmetria bizonyítás

Tétel: computeHashDistance a b = computeHashDistance b a

Strukturális indukcióval a bájt pozíciókon; minden pozícióban az xor_comm_uint8 tétel garantálja a bájtok XOR-jainak felcserélhetőségét.

 Azonosság bizonyítás

Tétel: computeHashDistance h h = 0

Alapja: x ^^^ x = 0 minden UInt8 értékre, ahol a Lean XOR önkioltó tulajdonsága (xor_self_zero) és a nulla popCount (popCount8_zero_is_zero) biztosítja a nullát.

 Háromszög-egyenlőtlenség bizonyítás

Tétel: d(a,c) ≤ d(a,b) + d(b,c)

| Szint | Összetevő | Cél |
|---|---|---|
| Bájt | popCount8_triangle_single | Egyetlen bájt XOR-jának egyenlőtlensége |
| Természetes | hammingDistNat_triangle | Teljes Hamming-távolság egyenlőtlensége |
| Racionális | computeHashDistance_triangle_rational | Teljes racionális egyenlőtlenség |

 Bizonyított tételek összefoglalója

| Tétel | Sor | Állítás |
|---|---|---|
| computeHashDistance_symmetric | 1693 | d(a,b) = d(b,a) |
| computeHashDistance_self_zero | 1700 | d(h,h) = 0 |
| computeHashDistance_triangle_rational | 1827 | d(a,c) ≤ d(a,b) + d(b,c) |
| computeHashDistance_nonneg | 1764 | d(a,b) ≥ 0 |
| computeHashDistance_bounded_num | 1769 | d(a,b).num ≤ HASH_BITS |

 Metrikus tér következményei

| Tulajdonság | Rendszer következménye |
|---|---|
| Szimmetria | A blokk-összehasonlítás sorrendtől független |
| Azonosság | Egy blokk maximálisan hasonlít önmagához |
| Háromszög-egyenlőtlenség | A távolságok tranzitívan konzisztensek |
| Korlátosság | Összes távolság normalizálva [0, 1]-be |

---

 Műveleti korrektség

 Cél és hatókör

A műveleti korrektségi tételek bizonyítják, hogy az egyes műveletek megőrzik a rendszer invariánsait. A tételek garantálják, hogy az egész végrehajtási nyomok megőrzik a rendszer invariánsait.

 Statisztikai műveletek korrektségi tételei

| Tétel | Tulajdonság |
|---|---|
| statistics_addBlock_total | Az összes blokk száma 1-gyel nő |
| statistics_addBlock_high | High szám nő, ha pontszám > küszöb |
| statistics_addBlock_low | Low szám nő, ha pontszám ≤ küszöb |
| statistics_invariant_addBlock | Megőrzi a StatisticsInvariant-ot |
| statistics_removeBlock_when_empty | Üres esetén nincs hatás |
| statistics_removeBlock_total | Az összes blokk 1-gyel csökken, ha > 0 |

 Hozzáférési rekord megőrzési táblázat

| Tulajdonság | Megőrzött? | Tétel |
|---|---|---|
| block_id | ✓ Igen | surpriseRecord_recordAccess_preserves_block_id |
| surprise_score | ✓ Igen | surpriseRecord_recordAccess_preserves_score |
| creation_time | ✓ Igen | surpriseRecord_recordAccess_preserves_creation |
| access_frequency | ✗ Nem (nő 1-gyel) | surpriseRecord_recordAccess_frequency |
| last_access_time | ✗ Nem (frissül) | surpriseRecord_recordAccess_time |

 Küszöb-frissítési helyesség

| Tétel | Tulajdonság |
|---|---|
| managerState_setSurpriseThreshold_threshold | Küszöb korlátolt értékre van állítva |
| managerState_setSurpriseThreshold_preserves_storage | Tároló változatlan |
| managerState_setSurpriseThreshold_preserves_records | Rekordok változatlanok |
| setThreshold_preserves_statistics_invariant | StatisticsInvariant megőrzött |

 Összefoglalás

| Művelet | Megőrzött invariánsok | Kulcs tulajdonságok |
|---|---|---|
| storeWithSurprise | ManagerInvariant, StatisticsInvariant | Növeli a blokk-számot, helyesen frissíti a statisztikákat |
| evictLowSurpriseBlocks | ManagerInvariant, StatisticsInvariant | Csökkenti a blokk-számot, nem távolítja el a max. prioritású blokkot |
| recordAccess | RetentionInvariant | Megőrzi az azonosítómezőket, növeli a frekvenciát 1-gyel |
| setSurpriseThreshold | ManagerInvariant | Megőrzi a tárolót és a rekordokat, [0,1]-re korlátoz |

---

 Nyomkövetés-végrehajtás

 

A nyomkövetés-végrehajtás formális keretrendszert biztosít a műveletek sorozatainak ellenőrzéséhez. A SystemTrace algebrai adatstruktúrák segítségével modellez műveleti sorozatokat.

 A SystemTrace struktúra

| Konstruktor | Paraméterek | Cél |
|---|---|---|
| SystemTrace.empty | Nincs | Alap eset |
| SystemTrace.storeOp | prev, data | Adattárolás meglepetési számítással |
| SystemTrace.evictOp | prev, target | Blokkok kizárása a célkapacitásig |
| SystemTrace.setThresholdOp | prev, threshold | Meglepetési küszöb módosítása |
| SystemTrace.accessOp | prev, blockId | Hozzáférés rögzítése meglévő blokkhoz |

 Nyomkövetés-alkalmazás főbb jellemzői

1. Rekurzív alkalmazás: Minden művelet először rekurzívan alkalmazza az előző nyomkövetést
2. Hibapropagáció: A korai műveletek hibái megakadályozzák a későbbiek végrehajtását
3. Állapot-szálak: Egy művelet állapota a következő inputja lesz

 Invariáns-megőrzési tételek

| Tétel | Cél |
|---|---|
| trace_invariant_init | A kezdeti állapot kielégíti az invariánst |
| applyTrace_empty_preserves_invariant | Az üres nyomkövetés megőrzi az invariánst |
| trace_setThreshold_preserves_invariant | A küszöb-beállítás megőrzi az invariánst |
| applyTrace_preserves_invariant_structural | Az összes nyomkövetés megőrzi az invariánst |

 Nyomkövetés hossza és műveletek számlálása

Az applyTrace_ok_implies_op_count_ge tétel bizonyítja, hogy a sikeres nyomkövetés-alkalmazás növeli a műveletek számát. Ez garantálja: a szám sosem csökken, és minden sikeres művelet pontosan 1-gyel növeli.

---

 Haladó témák

 Összefonódás-szervezés

Az összefonódás egy mechanizmus a magas meglepetési értékű blokkok párokba rendezésére, lehetővé téve a tároló háttérrendszer számára a fizikai elrendezés optimalizálását.

Korlátozások:
- Maximum MAX_ENTANGLEMENT_PAIRS = 100 magas meglepetési értékű blokk figyelembe vétele
- Maximum lehetséges párok: C(100, 2) = 4 950
- Az összefonódás nem befolyásolja a megtartási prioritás számítást
- Az összefonódott blokkok is kizárhatók, ha alacsony a prioritásuk

 Részleges rendezési algoritmus

Bonyolultság-összehasonlítás:

| Forgatókönyv | Teljes rendezés (MergeSort) | Részleges rendezés (Selection) |
|---|---|---|
| k=10, n=10 000 | ~133k művelet | ~100k művelet |
| k=100, n=10 000 | ~133k | ~1M (teljes rendezés jobb) |

Lean megközelítés: Teljes mergeSort + take k – egyszerűbb a bizonyítás. A Zig teljesítményért vált.

 Teljesítmény-jellemzők

Meglepetési számítás – O(1) amortizálva:

| Művelet | Bonyolultság |
|---|---|
| Mintaválasztás | O(n) |
| Jaccard-számítás | O(1000 × min(1000, data_len)) |
| Hash-távolság | O(1000 × 16) |
| Időbeli újdonság | O(1) |
| Összesen | O(n + 1M) |

Kizárási teljesítmény:

| Forgatókönyv | Idő |
|---|---|
| 10% kizárása 10 000 blokkból | 1 000 × 10 000 = 10M összehasonlítás |
| 50% kizárása 10 000 blokkból | 5 000 × 10 000 = 50M összehasonlítás |

Memóriaterhelés blokkszámtól függően:

| Rendszer mérete | Rekordmemória |
|---|---|
| 1 000 blokk | ~144 KB |
| 10 000 blokk | ~1,44 MB |
| 100 000 blokk | ~14,4 MB |
| 1 000 000 blokk | ~144 MB |

 Konfiguráció és hangolás

Meglepetési küszöb kiválasztása:

| Küszöb | Értelmezés | Felhasználási eset |
|---|---|---|
| 0.1 – 0.2 | Nagyon megengedő | Általános célú gyorsítótárazás |
| 0.3 – 0.4 | Közepes (alapértelmezett) | Kiegyensúlyozott redundancia és újdonság |
| 0.5 – 0.7 | Szigorú | Tudományos adatok elemzése |
| 0.8 – 1.0 | Nagyon szigorú | Extrém deduplikáció |

Megtartási prioritás súlyok hangolása:
- Alapsúly növelése (0.5 → 0.7): Újdonság prioritizálása a hozzáférési minták felett
- Kor-súly növelése (0.3 → 0.5): Nemrég elért adatok prioritizálása (LRU-szerű)
- Frekvencia-súly növelése (0.2 → 0.4): Sűrűn elért adatok prioritizálása (LFU-szerű)

Mintavételezési paraméterek:

| Mintaméret | Pontosság | Teljesítmény |
|---|---|---|
| 100 | Alacsony (~90%) | Nagyon gyors |
| 1000 (alapértelmezett) | Magas (~99%) | Gyors |
| 5000 | Nagyon magas (~99.9%) | Közepes |

Kapacitástervezési stratégia:

max_kapacitás = 100 000 blokk
magas vízjel  = 90 000 blokk  ← kizárás indítása
alacsony vízjel = 70 000 blokk ← kizárás célértéke

Amikor a tároló eléri a 90 000-et:
  → 20 000 legalacsonyabb prioritású blokk kizárása
  → Eredő szám: 70 000


Teljesítmény-monitorozás:

| Metrika | Optimális tartomány | Teendő, ha kívüle van |
|---|---|---|
| high_surprise_blocks / total_blocks | 0.2 – 0.4 | surprise_threshold beállítása |
| average_surprise | 0.3 – 0.6 | Adatok diverzitásának vizsgálata |
| novel_block_allocations / total_blocks | 0.3 – 0.5 | Az adatok újdonsági arányának mutatója |

Menetbiztonsági optimalizációs lehetőségek (jelenlegi implementációban nem valósítva meg):

1. Olvasó-írói zár: Egyidejű olvasások engedélyezése
2. Atomi számlálók: Statisztikai frissítések zár nélkül
3. Előzetes meglepetési számítás: Meglepetés kiszámítása zár megszerzése előtt
4. Zárfelosztás: Külön zárak a surprise_records-hoz és a statistics-hoz


 Szójegyzék

 Alaparchitekturális Fogalmak

| Fogalom | Definíció | Implementáció |
|---|---|---|
| KGRU | Kalmár-Gábor-Riesz Unity: a JAIDE alapfilozófiája három magyar matematikusról | README.md |
| RSF | Relational Signal Flow: egyedi neurális architektúra térbeli és időbeli súlymátrixokkal | src/processor/rsf.zig |
| MGT | Morpheme-Guided Tokenization: morfémabontást alkalmazó tokenizáló | src/tokenizer/mgt.zig |
| SFD | Stochastic Fractal Descent: egyedi optimalizáló fp4-től fp64-ig | src/optimizer/sfd.zig |
| Tensor | Többdimenziós tömb referenciaszámlálással és CoW szemantikával | src/core/tensor.zig |
| SSI | Succinct Semantic Index: trie/hash-fa token sorozatokhoz | src/index/ssi.zig |
| NSIR | Self-Similar Relational Graph: kvantumtulajdonságokkal rendelkező gráf | src/core_relational/nsir_core.zig |
| R-GPU | Relational Graph Processing Unit: aszinkron NoC-alapú hardver absztrakció | src/core_relational/r_gpu.zig |
| Anchor Token | Stabil token SSI-beli szegmensek indexeléséhez | src/index/ssi.zig |
| LSH | Locality Sensitive Hashing: a Ranker gyors hasonlóságkereséshez | src/main.zig |
| Z-Runtime | Relációs műveletek és változóállapot kezelési végrehajtási környezete | src/core_relational/mod.zig |
| Chaos Core | Dinamikus feladatütemezés és CAS kezelő kernel | src/core_relational/chaos_core.zig |
| ESSO | Entangled Stochastic Symmetry Optimizer: szimmetriaalapú gráf optimalizáló | src/core_relational/esso_optimizer.zig |
| Fixed32_32 | Egyedi 64-bit fixpontos típus (32+32 bit) precíziókontrolált aritmetikához | src/core/types.zig |
| Lean 4 | RSF helyesség és SSI invariánsok matematikai bizonyításához | src/verification/lean4/ |
| SAW | Software Analysis Workbench: Zig/LLVM bitkód verifikáció | src/verify.saw |
| Cryptol | Rendszerkonstansok magas szintű specifikációja | src/MainSpec.cry |
| ZK | Zero-Knowledge: következtetési nyomok verifikációja súlyok felfedése nélkül | src/zk/inference_trace.circom |
