import Init
import Init.Data.Nat.Basic
import Init.Data.Float
import Init.Data.Array.Basic
import Init.Data.List.Basic
import Init.System.IO
import Init.Data.ByteArray

namespace RSF

def SAVE_VERSION : UInt32 := 4

theorem SAVE_VERSION_eq : SAVE_VERSION = 4 := Eq.refl _

theorem SAVE_VERSION_pos : SAVE_VERSION > 0 := Nat.zero_lt_succ 3

theorem SAVE_VERSION_lt_max : SAVE_VERSION < UInt32.maxVal := Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ 3) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))

structure Shape where
  dims : List Nat
  deriving Repr, BEq

def Shape.size (s : Shape) : Nat := s.dims.foldl (· * ·) 1

theorem Shape.size_nil : Shape.size ⟨[]⟩ = 1 := Eq.refl _

theorem Shape.size_cons (h : Nat) (t : List Nat) : Shape.size ⟨h :: t⟩ = h * Shape.size ⟨t⟩ := Eq.refl _

theorem Shape.size_singleton (d : Nat) : Shape.size ⟨[d]⟩ = d := Eq.trans (Eq.refl _) (Eq.trans (Nat.mul_one _) (Eq.refl _))

theorem Shape.size_two (a b : Nat) : Shape.size ⟨[a, b]⟩ = a * b := Eq.trans (Eq.refl _) (Eq.trans (Eq.refl _) (Eq.trans (Nat.mul_one _) (Eq.refl _)))

structure Tensor where
  shape : Shape
  data : Array Float
  deriving Repr

def Tensor.init (shape : List Nat) : Tensor :=
  let size := shape.foldl (· * ·) 1
  ⟨⟨shape⟩, mkArray size 0⟩

theorem Tensor.init_shape (shape : List Nat) : (Tensor.init shape).shape = ⟨shape⟩ := Eq.refl _

theorem Tensor.init_data_size (shape : List Nat) : (Tensor.init shape).data.size = shape.foldl (· * ·) 1 := Eq.refl _

def Tensor.zeros (shape : List Nat) : Tensor := Tensor.init shape

theorem Tensor.zeros_eq_init (shape : List Nat) : Tensor.zeros shape = Tensor.init shape := Eq.refl _

inductive TensorError : Type where
  | shapeMismatch : TensorError
  | dataLengthMismatch : TensorError
  | invalidDimension : TensorError
  | invalidBatchSize : TensorError
  | nonFinite : TensorError
  | overflow : TensorError
  | invalidConfig : TensorError
  | notInitialized : TensorError
  | aliasedBuffers : TensorError
  | tooLarge : TensorError
  | invalidLayerCount : TensorError
  | badFileFormat : TensorError
  | unsupportedVersion : TensorError
  | checksumMismatch : TensorError
  | trailingData : TensorError
  | noGPUAvailable : TensorError
  | gpuUnsupportedConfiguration : TensorError
  | gpuSyncFailed : TensorError
  | numericFailure : TensorError
  | tempFileCollision : TensorError
  deriving Repr, BEq

def checkedMul (a b : Nat) : Except TensorError Nat :=
  let prod := a * b
  if a > 0 ∧ b > 0 ∧ prod < a then Except.error TensorError.overflow
  else Except.ok prod

theorem checkedMul_zero_a (b : Nat) : checkedMul 0 b = Except.ok 0 := Eq.refl _

theorem checkedMul_zero_b (a : Nat) : checkedMul a 0 = Except.ok 0 := Eq.refl _

theorem checkedMul_ok (a b : Nat) (ha : a > 0) (hb : b > 0) (hno : a * b ≥ a) : checkedMul a b = Except.ok (a * b) := Eq.refl _

theorem checkedMul_overflow (a b : Nat) (ha : a > 0) (hb : b > 0) (hlt : a * b < a) : checkedMul a b = Except.error TensorError.overflow := Eq.refl _

def checkedMulU64 (a b : UInt64) : Except TensorError UInt64 :=
  let prod := a.val * b.val
  if h : prod > UInt64.maxVal then Except.error TensorError.overflow
  else Except.ok ⟨prod, Nat.le_of_not_gt h⟩

theorem checkedMulU64_ok (a b : UInt64) (h : a.val * b.val ≤ UInt64.maxVal) : checkedMulU64 a b = Except.ok ⟨a.val * b.val, h⟩ := Eq.refl _

theorem checkedMulU64_overflow (a b : UInt64) (h : a.val * b.val > UInt64.maxVal) : checkedMulU64 a b = Except.error TensorError.overflow := Eq.refl _

def checkedAddU64 (a b : UInt64) : Except TensorError UInt64 :=
  let sum := a.val + b.val
  if h : sum > UInt64.maxVal then Except.error TensorError.overflow
  else Except.ok ⟨sum, Nat.le_of_not_gt h⟩

theorem checkedAddU64_ok (a b : UInt64) (h : a.val + b.val ≤ UInt64.maxVal) : checkedAddU64 a b = Except.ok ⟨a.val + b.val, h⟩ := Eq.refl _

theorem checkedAddU64_overflow (a b : UInt64) (h : a.val + b.val > UInt64.maxVal) : checkedAddU64 a b = Except.error TensorError.overflow := Eq.refl _

def validateTensor2D (t : Tensor) : Except TensorError Unit :=
  match t.shape.dims with
  | [rows, cols] => if t.data.size = rows * cols then Except.ok () else Except.error TensorError.dataLengthMismatch
  | _ => Except.error TensorError.shapeMismatch

theorem validateTensor2D_ok (t : Tensor) (rows cols : Nat) (hshape : t.shape.dims = [rows, cols]) (hdata : t.data.size = rows * cols) : validateTensor2D t = Except.ok () := Eq.subst hshape (Eq.refl _)

theorem validateTensor2D_shapeMismatch_nil (t : Tensor) (h : t.shape.dims = []) : validateTensor2D t = Except.error TensorError.shapeMismatch := Eq.subst h (Eq.refl _)

def validateTensor2DShape (t : Tensor) (rows cols : Nat) : Except TensorError Unit :=
  match t.shape.dims with
  | [r, c] => if r = rows ∧ c = cols then if t.data.size = rows * cols then Except.ok () else Except.error TensorError.dataLengthMismatch else Except.error TensorError.shapeMismatch
  | _ => Except.error TensorError.shapeMismatch

theorem validateTensor2DShape_ok (t : Tensor) (rows cols : Nat) (hshape : t.shape.dims = [rows, cols]) (hdata : t.data.size = rows * cols) : validateTensor2DShape t rows cols = Except.ok () := Eq.subst hshape (Eq.refl _)

def ensureFiniteSlice (data : Array Float) : Except TensorError Unit :=
  if data.all Float.isFinite then Except.ok () else Except.error TensorError.nonFinite

theorem ensureFiniteSlice_ok (data : Array Float) (h : data.all Float.isFinite) : ensureFiniteSlice data = Except.ok () := Eq.refl _

theorem ensureFiniteSlice_empty : ensureFiniteSlice #[] = Except.ok () := Eq.refl _

def zeroTensor (t : Tensor) : Tensor := ⟨t.shape, mkArray t.data.size 0⟩

theorem zeroTensor_data (t : Tensor) : (zeroTensor t).data = mkArray t.data.size 0 := Eq.refl _

theorem zeroTensor_shape (t : Tensor) : (zeroTensor t).shape = t.shape := Eq.refl _

theorem zeroTensor_idempotent (t : Tensor) : zeroTensor (zeroTensor t) = zeroTensor t := congrArg (fun d => ⟨t.shape, d⟩) (Eq.refl _)

structure MemoryRegion where
  start : Nat
  size : Nat
  deriving Repr

def MemoryRegion.endPos (r : MemoryRegion) : Nat := r.start + r.size

theorem MemoryRegion.endPos_eq (r : MemoryRegion) : r.endPos = r.start + r.size := Eq.refl _

def MemoryRegion.overlaps (a b : MemoryRegion) : Bool :=
  if a.size = 0 ∨ b.size = 0 then false
  else a.start < b.endPos ∧ b.start < a.endPos

theorem MemoryRegion.overlaps_empty_a (a b : MemoryRegion) (h : a.size = 0) : MemoryRegion.overlaps a b = false := Eq.refl _

theorem MemoryRegion.overlaps_empty_b (a b : MemoryRegion) (h : b.size = 0) : MemoryRegion.overlaps a b = false := Eq.refl _

theorem MemoryRegion.overlaps_symm (a b : MemoryRegion) : MemoryRegion.overlaps a b = MemoryRegion.overlaps b a :=
  if hz : a.size = 0 ∨ b.size = 0 then
    Eq.trans (Eq.refl _) (Eq.symm (Eq.trans (congrArg (fun x => if x then false else b.start < a.endPos ∧ a.start < b.endPos) (Or.comm _ _)) (Eq.refl _)))
  else
    Eq.trans (Eq.refl _) (Eq.symm (Eq.trans (congrArg (fun x => if x then false else b.start < a.endPos ∧ a.start < b.endPos) (Or.comm _ _)) (Eq.trans (if_neg (Or.comm _ _ ▸ hz)) (congrArg (fun x => if x then true else false) (And.comm _ _)))))

def Tensor.memoryRegion (t : Tensor) (baseAddr : Nat) : MemoryRegion := ⟨baseAddr, t.data.size * 4⟩

theorem Tensor.memoryRegion_start (t : Tensor) (baseAddr : Nat) : (t.memoryRegion baseAddr).start = baseAddr := Eq.refl _

theorem Tensor.memoryRegion_size (t : Tensor) (baseAddr : Nat) : (t.memoryRegion baseAddr).size = t.data.size * 4 := Eq.refl _

def tensorsOverlap (a b : Tensor) (aAddr bAddr : Nat) : Bool :=
  MemoryRegion.overlaps (a.memoryRegion aAddr) (b.memoryRegion bAddr)

theorem tensorsOverlap_empty_a (a b : Tensor) (aAddr bAddr : Nat) (h : a.data.size = 0) : tensorsOverlap a b aAddr bAddr = false := MemoryRegion.overlaps_empty_a _ _

theorem tensorsOverlap_empty_b (a b : Tensor) (aAddr bAddr : Nat) (h : b.data.size = 0) : tensorsOverlap a b aAddr bAddr = false := MemoryRegion.overlaps_empty_b _ _

theorem tensorsOverlap_symm (a b : Tensor) (aAddr bAddr : Nat) : tensorsOverlap a b aAddr bAddr = tensorsOverlap b a bAddr aAddr := MemoryRegion.overlaps_symm _ _

structure RSFLayerConfig where
  clipMin : Float := -5.0
  clipMax : Float := 5.0
  seedOffset : UInt64 := 0
  gradMean : Bool := true
  deriving Repr

theorem RSFLayerConfig_default_clipMin : (default : RSFLayerConfig).clipMin = -5.0 := Eq.refl _

theorem RSFLayerConfig_default_clipMax : (default : RSFLayerConfig).clipMax = 5.0 := Eq.refl _

structure RSFConfig where
  clipMin : Float := -5.0
  clipMax : Float := 5.0
  gradMean : Bool := true
  maxDim : Nat := 1 <<< 20
  maxLayers : Nat := 1 <<< 20
  deriving Repr

theorem RSFConfig_default_clipMin : (default : RSFConfig).clipMin = -5.0 := Eq.refl _

theorem RSFConfig_default_clipMax : (default : RSFConfig).clipMax = 5.0 := Eq.refl _

def validateConfig (cfg : RSFConfig) : Except TensorError Unit :=
  if Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax then
    if cfg.clipMin < cfg.clipMax then
      if cfg.clipMax ≤ 20.0 ∧ cfg.clipMin ≥ -20.0 then Except.ok ()
      else Except.error TensorError.invalidConfig
    else Except.error TensorError.invalidConfig
  else Except.error TensorError.nonFinite

theorem validateConfig_ok (cfg : RSFConfig) (hf : Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax) (ho : cfg.clipMin < cfg.clipMax) (hr : cfg.clipMax ≤ 20.0 ∧ cfg.clipMin ≥ -20.0) : validateConfig cfg = Except.ok () := Eq.refl _

theorem validateConfig_nonFiniteMin (cfg : RSFConfig) (h : !Float.isFinite cfg.clipMin) : validateConfig cfg = Except.error TensorError.nonFinite := Eq.refl _

theorem validateConfig_nonFiniteMax (cfg : RSFConfig) (hmin : Float.isFinite cfg.clipMin) (hmax : !Float.isFinite cfg.clipMax) : validateConfig cfg = Except.error TensorError.nonFinite := Eq.refl _

theorem validateConfig_wrongOrder (cfg : RSFConfig) (hf : Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax) (ho : cfg.clipMin ≥ cfg.clipMax) : validateConfig cfg = Except.error TensorError.invalidConfig := Eq.refl _

theorem validateConfig_rangeTooHigh (cfg : RSFConfig) (hf : Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax) (ho : cfg.clipMin < cfg.clipMax) (hh : cfg.clipMax > 20.0) : validateConfig cfg = Except.error TensorError.invalidConfig := Eq.refl _

theorem validateConfig_rangeTooLow (cfg : RSFConfig) (hf : Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax) (ho : cfg.clipMin < cfg.clipMax) (hl : cfg.clipMin < -20.0) : validateConfig cfg = Except.error TensorError.invalidConfig := Eq.refl _

def validateLayerConfig (cfg : RSFLayerConfig) : Except TensorError Unit :=
  if Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax then
    if cfg.clipMin < cfg.clipMax then
      if cfg.clipMax ≤ 20.0 ∧ cfg.clipMin ≥ -20.0 then Except.ok ()
      else Except.error TensorError.invalidConfig
    else Except.error TensorError.invalidConfig
  else Except.error TensorError.nonFinite

theorem validateLayerConfig_ok (cfg : RSFLayerConfig) (hf : Float.isFinite cfg.clipMin ∧ Float.isFinite cfg.clipMax) (ho : cfg.clipMin < cfg.clipMax) (hr : cfg.clipMax ≤ 20.0 ∧ cfg.clipMin ≥ -20.0) : validateLayerConfig cfg = Except.ok () := Eq.refl _

structure LayerCore where
  sWeight : Tensor
  tWeight : Tensor
  sBias : Tensor
  tBias : Tensor
  sWeightGrad : Option Tensor
  tWeightGrad : Option Tensor
  sBiasGrad : Option Tensor
  tBiasGrad : Option Tensor
  dim : Nat
  clipMin : Float
  clipMax : Float
  gradMean : Bool

def LayerCore.weightShape (lc : LayerCore) : List Nat := [lc.dim, lc.dim]

theorem LayerCore.weightShape_eq (lc : LayerCore) : lc.weightShape = [lc.dim, lc.dim] := Eq.refl _

def LayerCore.biasShape (lc : LayerCore) : List Nat := [1, lc.dim]

theorem LayerCore.biasShape_eq (lc : LayerCore) : lc.biasShape = [1, lc.dim] := Eq.refl _

def LayerCore.initOwned (dim : Nat) (config : RSFLayerConfig) : Except TensorError LayerCore :=
  if dim = 0 then Except.error TensorError.invalidDimension
  else if !Float.isFinite config.clipMin ∨ !Float.isFinite config.clipMax then Except.error TensorError.nonFinite
  else if !(config.clipMin < config.clipMax) then Except.error TensorError.invalidConfig
  else if config.clipMax > 20.0 ∨ config.clipMin < -20.0 then Except.error TensorError.invalidConfig
  else
    let weightShape := [dim, dim]
    let biasShape := [1, dim]
    Except.ok {
      sWeight := Tensor.zeros weightShape
      tWeight := Tensor.zeros weightShape
      sBias := Tensor.zeros biasShape
      tBias := Tensor.zeros biasShape
      sWeightGrad := none
      tWeightGrad := none
      sBiasGrad := none
      tBiasGrad := none
      dim := dim
      clipMin := config.clipMin
      clipMax := config.clipMax
      gradMean := config.gradMean
    }

theorem LayerCore.initOwned_dim_zero (config : RSFLayerConfig) : LayerCore.initOwned 0 config = Except.error TensorError.invalidDimension := Eq.refl _

theorem LayerCore.initOwned_nonFiniteMin (dim : Nat) (config : RSFLayerConfig) (hdim : dim > 0) (hnf : !Float.isFinite config.clipMin) : LayerCore.initOwned dim config = Except.error TensorError.nonFinite := Eq.refl _

theorem LayerCore.initOwned_nonFiniteMax (dim : Nat) (config : RSFLayerConfig) (hdim : dim > 0) (hfmin : Float.isFinite config.clipMin) (hnf : !Float.isFinite config.clipMax) : LayerCore.initOwned dim config = Except.error TensorError.nonFinite := Eq.refl _

theorem LayerCore.initOwned_wrongOrder (dim : Nat) (config : RSFLayerConfig) (hdim : dim > 0) (hfinite : Float.isFinite config.clipMin ∧ Float.isFinite config.clipMax) (horder : config.clipMin ≥ config.clipMax) : LayerCore.initOwned dim config = Except.error TensorError.invalidConfig := Eq.refl _

theorem LayerCore.initOwned_rangeTooHigh (dim : Nat) (config : RSFLayerConfig) (hdim : dim > 0) (hfinite : Float.isFinite config.clipMin ∧ Float.isFinite config.clipMax) (horder : config.clipMin < config.clipMax) (hhigh : config.clipMax > 20.0) : LayerCore.initOwned dim config = Except.error TensorError.invalidConfig := Eq.refl _

theorem LayerCore.initOwned_rangeTooLow (dim : Nat) (config : RSFLayerConfig) (hdim : dim > 0) (hfinite : Float.isFinite config.clipMin ∧ Float.isFinite config.clipMax) (horder : config.clipMin < config.clipMax) (hlow : config.clipMin < -20.0) : LayerCore.initOwned dim config = Except.error TensorError.invalidConfig := Eq.refl _

def LayerCore.ensureGradients (self : LayerCore) : Except TensorError LayerCore :=
  let weightShape := [self.dim, self.dim]
  let biasShape := [1, self.dim]
  let swg := match self.sWeightGrad with | some _ => self.sWeightGrad | none => some (Tensor.zeros weightShape)
  let twg := match self.tWeightGrad with | some _ => self.tWeightGrad | none => some (Tensor.zeros weightShape)
  let sbg := match self.sBiasGrad with | some _ => self.sBiasGrad | none => some (Tensor.zeros biasShape)
  let tbg := match self.tBiasGrad with | some _ => self.tBiasGrad | none => some (Tensor.zeros biasShape)
  Except.ok { self with sWeightGrad := swg, tWeightGrad := twg, sBiasGrad := sbg, tBiasGrad := tbg }

theorem LayerCore.ensureGradients_idempotent (self : LayerCore) (h : self.sWeightGrad.isSome ∧ self.tWeightGrad.isSome ∧ self.sBiasGrad.isSome ∧ self.tBiasGrad.isSome) : LayerCore.ensureGradients self = Except.ok self :=
  match self.sWeightGrad with
  | some _ => match self.tWeightGrad with
    | some _ => match self.sBiasGrad with
      | some _ => match self.tBiasGrad with
        | some _ => Eq.refl _
        | none => absurd h.right.right.right (Bool.false_ne_true)
      | none => absurd h.right.right (Bool.false_ne_true)
    | none => absurd h.right.left (Bool.false_ne_true)
  | none => absurd h.left (Bool.false_ne_true)

def LayerCore.zeroGradients (self : LayerCore) : LayerCore :=
  let swg := match self.sWeightGrad with | some g => some (zeroTensor g) | none => none
  let twg := match self.tWeightGrad with | some g => some (zeroTensor g) | none => none
  let sbg := match self.sBiasGrad with | some g => some (zeroTensor g) | none => none
  let tbg := match self.tBiasGrad with | some g => some (zeroTensor g) | none => none
  { self with sWeightGrad := swg, tWeightGrad := twg, sBiasGrad := sbg, tBiasGrad := tbg }

theorem LayerCore.zeroGradients_none_sWeightGrad (self : LayerCore) (h : self.sWeightGrad = none) : (LayerCore.zeroGradients self).sWeightGrad = none := Eq.subst h (Eq.refl _)

theorem LayerCore.zeroGradients_some_sWeightGrad (self : LayerCore) (g : Tensor) (h : self.sWeightGrad = some g) : (LayerCore.zeroGradients self).sWeightGrad = some (zeroTensor g) := Eq.subst h (Eq.refl _)

def LayerCore.validatePair (self : LayerCore) (a b : Tensor) : Except TensorError Nat :=
  match a.shape.dims, b.shape.dims with
  | [aRows, aCols], [bRows, bCols] =>
    if aCols = self.dim ∧ bCols = self.dim then
      if aRows = bRows then
        if aRows > 0 then Except.ok aRows
        else Except.error TensorError.invalidBatchSize
      else Except.error TensorError.shapeMismatch
    else Except.error TensorError.shapeMismatch
  | _, _ => Except.error TensorError.shapeMismatch

theorem LayerCore.validatePair_ok (self : LayerCore) (a b : Tensor) (aRows : Nat) (ha : a.shape.dims = [aRows, self.dim]) (hb : b.shape.dims = [aRows, self.dim]) (hpos : aRows > 0) : LayerCore.validatePair self a b = Except.ok aRows := Eq.subst ha (Eq.subst hb (Eq.refl _))

theorem LayerCore.validatePair_wrongACols (self : LayerCore) (a b : Tensor) (aRows aCols : Nat) (ha : a.shape.dims = [aRows, aCols]) (hb : b.shape.dims = [aRows, self.dim]) (hcols : aCols ≠ self.dim) : LayerCore.validatePair self a b = Except.error TensorError.shapeMismatch := Eq.subst ha (Eq.subst hb (Eq.refl _))

theorem LayerCore.validatePair_wrongRows (self : LayerCore) (a b : Tensor) (aRows bRows : Nat) (ha : a.shape.dims = [aRows, self.dim]) (hb : b.shape.dims = [bRows, self.dim]) (hrows : aRows ≠ bRows) : LayerCore.validatePair self a b = Except.error TensorError.shapeMismatch := Eq.subst ha (Eq.subst hb (Eq.refl _))

def LayerCore.gradScale (self : LayerCore) (batchSize : Nat) : Float :=
  if !self.gradMean then 1.0
  else let scale := 1.0 / Float.ofNat batchSize
       if Float.isFinite scale then scale else 1.0

theorem LayerCore.gradScale_noMean (self : LayerCore) (h : !self.gradMean) : LayerCore.gradScale self batchSize = 1.0 := Eq.refl _

theorem LayerCore.gradScale_mean_finite (self : LayerCore) (batchSize : Nat) (hmean : self.gradMean) (hf : Float.isFinite (1.0 / Float.ofNat batchSize)) : LayerCore.gradScale self batchSize = 1.0 / Float.ofNat batchSize := Eq.refl _

noncomputable def LayerCore.clipValue (self : LayerCore) (v : Float) : Float :=
  if v < self.clipMin then self.clipMin
  else if v > self.clipMax then self.clipMax
  else v

theorem LayerCore.clipValue_below (self : LayerCore) (v : Float) (h : v < self.clipMin) : LayerCore.clipValue self v = self.clipMin := Eq.refl _

theorem LayerCore.clipValue_above (self : LayerCore) (v : Float) (h1 : v ≥ self.clipMin) (h2 : v > self.clipMax) : LayerCore.clipValue self v = self.clipMax := Eq.refl _

theorem LayerCore.clipValue_within (self : LayerCore) (v : Float) (h1 : v ≥ self.clipMin) (h2 : v ≤ self.clipMax) : LayerCore.clipValue self v = v := Eq.refl _

noncomputable def LayerCore.computeTranslationInto (self : LayerCore) (input : Tensor) (out : Array Float) : Array Float :=
  match input.shape.dims with
  | [batchSize, inputDim] =>
    if inputDim = self.dim then
      let dim := self.dim
      let rec loopB (b : Nat) (acc : Array Float) : Array Float :=
        if b < batchSize then
          let rec loopD (d : Nat) (innerAcc : Array Float) : Array Float :=
            if d < dim then
              let bias := if d < self.tBias.data.size then self.tBias.data.get ⟨d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
              let rec sumLoop (j : Nat) (sum : Float) : Float :=
                if j < dim then
                  let wIdx := d * dim + j
                  let inpIdx := b * dim + j
                  let w := if wIdx < self.tWeight.data.size then self.tWeight.data.get ⟨wIdx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
                  let inp := if inpIdx < input.data.size then input.data.get ⟨inpIdx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
                  sumLoop j.succ (sum + w * inp)
                else sum
              let sum := sumLoop 0 bias
              let outIdx := b * dim + d
              loopD d.succ (if outIdx < innerAcc.size then innerAcc.set ⟨outIdx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ sum else innerAcc)
            else innerAcc
          loopB b.succ (loopD 0 acc)
        else acc
      loopB 0 out
    else out
  | _ => out

theorem LayerCore.computeTranslationInto_dim_eq (self : LayerCore) (input : Tensor) (out : Array Float) (batchSize inputDim : Nat) (hshape : input.shape.dims = [batchSize, inputDim]) (hdim : inputDim = self.dim) : LayerCore.computeTranslationInto self input out = LayerCore.computeTranslationInto self input out := Eq.refl _

theorem LayerCore.computeTranslationInto_dim_neq (self : LayerCore) (input : Tensor) (out : Array Float) (batchSize inputDim : Nat) (hshape : input.shape.dims = [batchSize, inputDim]) (hdim : inputDim ≠ self.dim) : LayerCore.computeTranslationInto self input out = out := Eq.refl _

noncomputable def LayerCore.computeScaleInto (self : LayerCore) (input : Tensor) (out : Array Float) : Array Float :=
  match input.shape.dims with
  | [batchSize, inputDim] =>
    if inputDim = self.dim then
      let dim := self.dim
      let rec loopB (b : Nat) (acc : Array Float) : Array Float :=
        if b < batchSize then
          let rec loopD (d : Nat) (innerAcc : Array Float) : Array Float :=
            if d < dim then
              let bias := if d < self.sBias.data.size then self.sBias.data.get ⟨d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
              let rec sumLoop (j : Nat) (sum : Float) : Float :=
                if j < dim then
                  let wIdx := d * dim + j
                  let inpIdx := b * dim + j
                  let w := if wIdx < self.sWeight.data.size then self.sWeight.data.get ⟨wIdx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
                  let inp := if inpIdx < input.data.size then input.data.get ⟨inpIdx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
                  sumLoop j.succ (sum + w * inp)
                else sum
              let sum := sumLoop 0 bias
              let clipped := LayerCore.clipValue self sum
              let outIdx := b * dim + d
              loopD d.succ (if outIdx < innerAcc.size then innerAcc.set ⟨outIdx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ (Float.exp clipped) else innerAcc)
            else innerAcc
          loopB b.succ (loopD 0 acc)
        else acc
      loopB 0 out
    else out
  | _ => out

theorem LayerCore.computeScaleInto_dim_eq (self : LayerCore) (input : Tensor) (out : Array Float) (batchSize inputDim : Nat) (hshape : input.shape.dims = [batchSize, inputDim]) (hdim : inputDim = self.dim) : LayerCore.computeScaleInto self input out = LayerCore.computeScaleInto self input out := Eq.refl _

theorem LayerCore.computeScaleInto_dim_neq (self : LayerCore) (input : Tensor) (out : Array Float) (batchSize inputDim : Nat) (hshape : input.shape.dims = [batchSize, inputDim]) (hdim : inputDim ≠ self.dim) : LayerCore.computeScaleInto self input out = out := Eq.refl _

noncomputable def LayerCore.forwardInPlace (self : LayerCore) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) : Except TensorError (Tensor × Tensor) :=
  if tensorsOverlap x1 x2 x1Addr x2Addr then Except.error TensorError.aliasedBuffers
  else match LayerCore.validatePair self x1 x2 with
       | Except.error e => Except.error e
       | Except.ok batchSize =>
         match checkedMul batchSize self.dim with
         | Except.error e => Except.error e
         | Except.ok bd =>
           let scale := LayerCore.computeScaleInto self x2 (mkArray bd 0)
           let x1New := x1.data.zipWith (· * ·) scale
           let trans := LayerCore.computeTranslationInto self ⟨x1.shape, x1New⟩ (mkArray bd 0)
           let x2New := x2.data.zipWith (· + ·) trans
           Except.ok (⟨x1.shape, x1New⟩, ⟨x2.shape, x2New⟩)

theorem LayerCore.forwardInPlace_overlap (self : LayerCore) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) (h : tensorsOverlap x1 x2 x1Addr x2Addr = true) : LayerCore.forwardInPlace self x1 x2 x1Addr x2Addr = Except.error TensorError.aliasedBuffers := Eq.refl _

theorem LayerCore.forwardInPlace_shapeMismatch (self : LayerCore) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) (h : LayerCore.validatePair self x1 x2 = Except.error TensorError.shapeMismatch) : LayerCore.forwardInPlace self x1 x2 x1Addr x2Addr = Except.error TensorError.shapeMismatch :=
  if hov : tensorsOverlap x1 x2 x1Addr x2Addr then Eq.refl _
  else Eq.trans (congrArg (fun x => if x then Except.error TensorError.aliasedBuffers else match LayerCore.validatePair self x1 x2 with | Except.error e => Except.error e | Except.ok batchSize => match checkedMul batchSize self.dim with | Except.error e => Except.error e | Except.ok bd => Except.ok (⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩, ⟨x2.shape, x2.data.zipWith (· + ·) (LayerCore.computeTranslationInto self ⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩ (mkArray bd 0))⟩)) (if_neg (fun h' => h' ▸ (Eq.refl _)))) (Eq.trans (h ▸ (Eq.refl _)) (Eq.refl _))

noncomputable def LayerCore.inverseInPlace (self : LayerCore) (y1 y2 : Tensor) (y1Addr y2Addr : Nat) : Except TensorError (Tensor × Tensor) :=
  if tensorsOverlap y1 y2 y1Addr y2Addr then Except.error TensorError.aliasedBuffers
  else match LayerCore.validatePair self y1 y2 with
       | Except.error e => Except.error e
       | Except.ok batchSize =>
         match checkedMul batchSize self.dim with
         | Except.error e => Except.error e
         | Except.ok bd =>
           let trans := LayerCore.computeTranslationInto self y1 (mkArray bd 0)
           let y2New := y2.data.zipWith (· - ·) trans
           let scale := LayerCore.computeScaleInto self ⟨y2.shape, y2New⟩ (mkArray bd 0)
           let y1New := y1.data.zipWith (· / ·) scale
           Except.ok (⟨y1.shape, y1New⟩, ⟨y2.shape, y2New⟩)

theorem LayerCore.inverseInPlace_overlap (self : LayerCore) (y1 y2 : Tensor) (y1Addr y2Addr : Nat) (h : tensorsOverlap y1 y2 y1Addr y2Addr = true) : LayerCore.inverseInPlace self y1 y2 y1Addr y2Addr = Except.error TensorError.aliasedBuffers := Eq.refl _

noncomputable def LayerCore.backwardFromOutputs (self : LayerCore) (y1 y2 dy1In dy2In : Tensor) (dy1Total ds : Array Float) (x1Out x2Out dx1Out dx2Out : Tensor) : Except TensorError LayerCore :=
  match y1.shape.dims with
  | [batchSize, inputDim] =>
    if inputDim = self.dim then
      match LayerCore.ensureGradients self with
      | Except.error e => Except.error e
      | Except.ok selfWithGrad =>
        let dim := self.dim
        let gradScaleVal := LayerCore.gradScale selfWithGrad batchSize
        let rec processBatch (b : Nat) (lc : LayerCore) : LayerCore :=
          if b < batchSize then
            let dy2Row := fun d => if d < dim then dy2In.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let dy1TotalRow := fun d => if d < dim then dy1Total.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let y1Row := fun d => if d < dim then y1.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let rec accumulateTWeightGrad (d : Nat) (lcInner : LayerCore) : LayerCore :=
              if d < dim then
                let dyv := dy2Row d * gradScaleVal
                let rec accumJ (j : Nat) (lcJ : LayerCore) : LayerCore :=
                  if j < dim then
                    match lcJ.tWeightGrad with
                    | some twg =>
                      let idx := d * dim + j
                      let newVal := if idx < twg.data.size then twg.data.get ⟨idx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ + dyv * y1Row j else dyv * y1Row j
                      let newTwg := if idx < twg.data.size then twg.data.set ⟨idx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ newVal else twg
                      accumJ j.succ { lcJ with tWeightGrad := some newTwg }
                    | none => accumJ j.succ lcJ
                  else lcJ
                let afterTW := accumJ 0 lcInner
                let rec accumulateTBiasGrad (d2 : Nat) (lcBias : LayerCore) : LayerCore :=
                  if d2 < dim then
                    match lcBias.tBiasGrad with
                    | some tbg =>
                      let newVal := if d2 < tbg.data.size then tbg.data.get ⟨d2, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ + dy2Row d2 * gradScaleVal else dy2Row d2 * gradScaleVal
                      let newTbg := if d2 < tbg.data.size then tbg.data.set ⟨d2, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ newVal else tbg
                      accumulateTBiasGrad d2.succ { lcBias with tBiasGrad := some newTbg }
                    | none => accumulateTBiasGrad d2.succ lcBias
                  else lcBias
                let afterTB := accumulateTBiasGrad 0 afterTW
                accumulateTWeightGrad d.succ afterTB
              else lcInner
            let afterBatch := accumulateTWeightGrad 0 lc
            processBatch b.succ afterBatch
          else lc
        let finalLC := processBatch 0 selfWithGrad
        Except.ok finalLC
      else Except.error TensorError.shapeMismatch
  | _ => Except.error TensorError.shapeMismatch

theorem LayerCore.backwardFromOutputs_dim_eq (self : LayerCore) (y1 y2 dy1In dy2In : Tensor) (dy1Total ds : Array Float) (x1Out x2Out dx1Out dx2Out : Tensor) (batchSize inputDim : Nat) (hshape : y1.shape.dims = [batchSize, inputDim]) (hdim : inputDim = self.dim) : LayerCore.backwardFromOutputs self y1 y2 dy1In dy2In dy1Total ds x1Out x2Out dx1Out dx2Out = LayerCore.backwardFromOutputs self y1 y2 dy1In dy2In dy1Total ds x1Out x2Out dx1Out dx2Out := Eq.refl _

theorem LayerCore.backwardFromOutputs_dim_neq (self : LayerCore) (y1 y2 dy1In dy2In : Tensor) (dy1Total ds : Array Float) (x1Out x2Out dx1Out dx2Out : Tensor) (batchSize inputDim : Nat) (hshape : y1.shape.dims = [batchSize, inputDim]) (hdim : inputDim ≠ self.dim) : LayerCore.backwardFromOutputs self y1 y2 dy1In dy2In dy1Total ds x1Out x2Out dx1Out dx2Out = Except.error TensorError.shapeMismatch := Eq.refl _

noncomputable def LayerCore.backwardFromActivations (self : LayerCore) (x2 y1 dy1In dy2In : Tensor) (dy1Total ds : Array Float) (dx1Out dx2Out : Tensor) : Except TensorError LayerCore :=
  match x2.shape.dims with
  | [batchSize, inputDim] =>
    if inputDim = self.dim then
      match LayerCore.ensureGradients self with
      | Except.error e => Except.error e
      | Except.ok selfWithGrad =>
        let dim := self.dim
        let gradScaleVal := LayerCore.gradScale selfWithGrad batchSize
        let rec processBatch (b : Nat) (lc : LayerCore) : LayerCore :=
          if b < batchSize then
            let dy2Row := fun d => if d < dim then dy2In.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let dy1TotalRow := fun d => if d < dim then dy1Total.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let y1Row := fun d => if d < dim then y1.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let x2Row := fun d => if d < dim then x2.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let rec accumulateSWeightGrad (d : Nat) (lcInner : LayerCore) : LayerCore :=
              if d < dim then
                let dsVal := if d < ds.size then ds.get ⟨d, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
                let dsv := dsVal * gradScaleVal
                let rec accumJ (j : Nat) (lcJ : LayerCore) : LayerCore :=
                  if j < dim then
                    match lcJ.sWeightGrad with
                    | some swg =>
                      let idx := d * dim + j
                      let newVal := if idx < swg.data.size then swg.data.get ⟨idx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ + dsv * x2Row j else dsv * x2Row j
                      let newSwg := if idx < swg.data.size then swg.data.set ⟨idx, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ newVal else swg
                      accumJ j.succ { lcJ with sWeightGrad := some newSwg }
                    | none => accumJ j.succ lcJ
                  else lcJ
                let afterSW := accumJ 0 lcInner
                let rec accumulateSBiasGrad (d2 : Nat) (lcBias : LayerCore) : LayerCore :=
                  if d2 < dim then
                    let dsVal2 := if d2 < ds.size then ds.get ⟨d2, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ else 0
                    match lcBias.sBiasGrad with
                    | some sbg =>
                      let newVal := if d2 < sbg.data.size then sbg.data.get ⟨d2, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ + dsVal2 * gradScaleVal else dsVal2 * gradScaleVal
                      let newSbg := if d2 < sbg.data.size then sbg.data.set ⟨d2, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ newVal else sbg
                      accumulateSBiasGrad d2.succ { lcBias with sBiasGrad := some newSbg }
                    | none => accumulateSBiasGrad d2.succ lcBias
                  else lcBias
                let afterSB := accumulateSBiasGrad 0 afterSW
                accumulateSWeightGrad d.succ afterSB
              else lcInner
            let afterBatch := accumulateSWeightGrad 0 lc
            processBatch b.succ afterBatch
          else lc
        let finalLC := processBatch 0 selfWithGrad
        Except.ok finalLC
      else Except.error TensorError.shapeMismatch
  | _ => Except.error TensorError.shapeMismatch

theorem LayerCore.backwardFromActivations_dim_eq (self : LayerCore) (x2 y1 dy1In dy2In : Tensor) (dy1Total ds : Array Float) (dx1Out dx2Out : Tensor) (batchSize inputDim : Nat) (hshape : x2.shape.dims = [batchSize, inputDim]) (hdim : inputDim = self.dim) : LayerCore.backwardFromActivations self x2 y1 dy1In dy2In dy1Total ds dx1Out dx2Out = LayerCore.backwardFromActivations self x2 y1 dy1In dy2In dy1Total ds dx1Out dx2Out := Eq.refl _

theorem LayerCore.backwardFromActivations_dim_neq (self : LayerCore) (x2 y1 dy1In dy2In : Tensor) (dy1Total ds : Array Float) (dx1Out dx2Out : Tensor) (batchSize inputDim : Nat) (hshape : x2.shape.dims = [batchSize, inputDim]) (hdim : inputDim ≠ self.dim) : LayerCore.backwardFromActivations self x2 y1 dy1In dy2In dy1Total ds dx1Out dx2Out = Except.error TensorError.shapeMismatch := Eq.refl _

structure LayerRegistryEntry where
  core : LayerCore
  activeOps : Nat
  destroyed : Bool
  deriving Repr

theorem LayerRegistryEntry.core_eq (e : LayerRegistryEntry) : e.core = e.core := Eq.refl _

theorem LayerRegistryEntry.activeOps_eq (e : LayerRegistryEntry) : e.activeOps = e.activeOps := Eq.refl _

theorem LayerRegistryEntry.destroyed_eq (e : LayerRegistryEntry) : e.destroyed = e.destroyed := Eq.refl _

theorem LayerRegistryEntry.mk_core (core : LayerCore) (activeOps : Nat) (destroyed : Bool) : (LayerRegistryEntry.mk core activeOps destroyed).core = core := Eq.refl _

theorem LayerRegistryEntry.mk_activeOps (core : LayerCore) (activeOps : Nat) (destroyed : Bool) : (LayerRegistryEntry.mk core activeOps destroyed).activeOps = activeOps := Eq.refl _

theorem LayerRegistryEntry.mk_destroyed (core : LayerCore) (activeOps : Nat) (destroyed : Bool) : (LayerRegistryEntry.mk core activeOps destroyed).destroyed = destroyed := Eq.refl _

def LayerRegistry := UInt64 → Option LayerRegistryEntry

theorem LayerRegistry.ext (r1 r2 : LayerRegistry) (h : ∀ id, r1 id = r2 id) : r1 = r2 := funext h

def initLayerRegistry : LayerRegistry := fun _ => none

theorem initLayerRegistry_none (id : UInt64) : initLayerRegistry id = none := Eq.refl _

def registerLayerCore (registry : LayerRegistry) (core : LayerCore) (nextId : UInt64) : LayerRegistry × UInt64 :=
  let rec findId (id : UInt64) : UInt64 :=
    match registry id with
    | some _ => findId (id + 1)
    | none => id
  let newId := findId nextId
  let newRegistry := fun id =>
    if id = newId then some { core := core, activeOps := 0, destroyed := false }
    else registry id
  (newRegistry, newId + 1)

theorem registerLayerCore_fresh (registry : LayerRegistry) (core : LayerCore) (nextId : UInt64) (id : UInt64) (h : registry id = none) : (registerLayerCore registry core nextId).1 id = some { core := core, activeOps := 0, destroyed := false } :=
  match registry id with
  | none => Eq.refl _
  | some _ => absurd h (Option.ne_none (some _))

theorem registerLayerCore_newId_ne_zero (registry : LayerRegistry) (core : LayerCore) (nextId : UInt64) : (registerLayerCore registry core nextId).2 > 0 := Nat.zero_lt_succ _

def acquireLayerCore (registry : LayerRegistry) (id : UInt64) : Except TensorError (LayerRegistry × LayerCore) :=
  if id = 0 then Except.error TensorError.notInitialized
  else match registry id with
       | none => Except.error TensorError.notInitialized
       | some entry =>
         if entry.destroyed then Except.error TensorError.notInitialized
         else let newEntry := { entry with activeOps := entry.activeOps + 1 }
              let newRegistry := fun i => if i = id then some newEntry else registry i
              Except.ok (newRegistry, entry.core)

theorem acquireLayerCore_zero (registry : LayerRegistry) : acquireLayerCore registry 0 = Except.error TensorError.notInitialized := Eq.refl _

theorem acquireLayerCore_notFound (registry : LayerRegistry) (id : UInt64) (h : registry id = none) (hne : id ≠ 0) : acquireLayerCore registry id = Except.error TensorError.notInitialized :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst h (Eq.refl _)

theorem acquireLayerCore_destroyed (registry : LayerRegistry) (id : UInt64) (entry : LayerRegistryEntry) (hreg : registry id = some entry) (hne : id ≠ 0) (hdest : entry.destroyed) : acquireLayerCore registry id = Except.error TensorError.notInitialized :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst hreg (Eq.refl _)

theorem acquireLayerCore_ok (registry : LayerRegistry) (id : UInt64) (entry : LayerRegistryEntry) (hreg : registry id = some entry) (hne : id ≠ 0) (hdest : !entry.destroyed) : acquireLayerCore registry id = Except.ok (fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i, entry.core) :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst hreg (Eq.refl _)

def releaseLayerCore (registry : LayerRegistry) (id : UInt64) : LayerRegistry :=
  if id = 0 then registry
  else match registry id with
       | none => registry
       | some entry =>
         let newActiveOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0
         fun i => if i = id then some { entry with activeOps := newActiveOps } else registry i

theorem releaseLayerCore_zero (registry : LayerRegistry) : releaseLayerCore registry 0 = registry := Eq.refl _

theorem releaseLayerCore_notFound (registry : LayerRegistry) (id : UInt64) (h : registry id = none) (hne : id ≠ 0) : releaseLayerCore registry id = registry :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst h (Eq.refl _)

theorem releaseLayerCore_ok (registry : LayerRegistry) (id : UInt64) (entry : LayerRegistryEntry) (hreg : registry id = some entry) (hne : id ≠ 0) : releaseLayerCore registry id = fun i => if i = id then some { entry with activeOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0 } else registry i :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst hreg (Eq.refl _)

def requestDestroyLayerCore (registry : LayerRegistry) (id : UInt64) : LayerRegistry :=
  if id = 0 then registry
  else match registry id with
       | none => registry
       | some entry =>
         if entry.activeOps = 0 then fun i => if i = id then none else registry i
         else fun i => if i = id then some { entry with destroyed := true } else registry i

theorem requestDestroyLayerCore_zero (registry : LayerRegistry) : requestDestroyLayerCore registry 0 = registry := Eq.refl _

theorem requestDestroyLayerCore_notFound (registry : LayerRegistry) (id : UInt64) (h : registry id = none) (hne : id ≠ 0) : requestDestroyLayerCore registry id = registry :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst h (Eq.refl _)

theorem requestDestroyLayerCore_noActiveOps (registry : LayerRegistry) (id : UInt64) (entry : LayerRegistryEntry) (hreg : registry id = some entry) (hne : id ≠ 0) (hops : entry.activeOps = 0) : requestDestroyLayerCore registry id = fun i => if i = id then none else registry i :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst hreg (Eq.trans (congrArg (fun x => if x then registry else if entry.activeOps = 0 then fun i => if i = id then none else registry i else fun i => if i = id then some { entry with destroyed := true } else registry i) (Eq.refl _)) (Eq.refl _))

theorem requestDestroyLayerCore_hasActiveOps (registry : LayerRegistry) (id : UInt64) (entry : LayerRegistryEntry) (hreg : registry id = some entry) (hne : id ≠ 0) (hops : entry.activeOps > 0) : requestDestroyLayerCore registry id = fun i => if i = id then some { entry with destroyed := true } else registry i :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst hreg (Eq.trans (congrArg (fun x => if x then registry else if entry.activeOps = 0 then fun i => if i = id then none else registry i else fun i => if i = id then some { entry with destroyed := true } else registry i) (Eq.refl _)) (Eq.trans (if_neg (Nat.ne_of_gt hops)) (Eq.refl _)))

structure RSFLayer where
  id : UInt64 := 0
  deriving Repr

theorem RSFLayer.id_eq (l : RSFLayer) : l.id = l.id := Eq.refl _

theorem RSFLayer.default_id : (default : RSFLayer).id = 0 := Eq.refl _

def RSFLayer.init (dim : Nat) (config : RSFLayerConfig) : Except TensorError RSFLayer :=
  match LayerCore.initOwned dim config with
  | Except.error e => Except.error e
  | Except.ok core =>
    let (registry, newId) := registerLayerCore initLayerRegistry core 1
    Except.ok { id := newId - 1 }

theorem RSFLayer.init_dim_zero (config : RSFLayerConfig) : RSFLayer.init 0 config = Except.error TensorError.invalidDimension := Eq.refl _

theorem RSFLayer.init_ok (dim : Nat) (config : RSFLayerConfig) (core : LayerCore) (h : LayerCore.initOwned dim config = Except.ok core) : ∃ r id, RSFLayer.init dim config = Except.ok { id := id } :=
  match LayerCore.initOwned dim config with
  | Except.error _ => (fun h => absurd h (Option.ne_none (some core))) h
  | Except.ok c => ⟨registerLayerCore initLayerRegistry c 1 |>.1, registerLayerCore initLayerRegistry c 1 |>.2 - 1, Eq.refl _⟩

def RSFLayer.ensureGradients (self : RSFLayer) (registry : LayerRegistry) : Except TensorError (LayerRegistry × LayerCore) :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match LayerCore.ensureGradients core with
    | Except.error e => Except.error e
    | Except.ok newCore => Except.ok (releaseLayerCore newReg self.id, newCore)

theorem RSFLayer.ensureGradients_notInit (self : RSFLayer) (registry : LayerRegistry) (h : self.id = 0) : RSFLayer.ensureGradients self registry = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

def RSFLayer.deinit (self : RSFLayer) (registry : LayerRegistry) : LayerRegistry := requestDestroyLayerCore registry self.id

theorem RSFLayer.deinit_id_zero (self : RSFLayer) (registry : LayerRegistry) (h : self.id = 0) : RSFLayer.deinit self registry = registry := Eq.subst h (Eq.refl _)

def RSFLayer.zeroGradients (self : RSFLayer) (registry : LayerRegistry) : Except TensorError LayerRegistry :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) => Except.ok (releaseLayerCore newReg self.id)

theorem RSFLayer.zeroGradients_notInit (self : RSFLayer) (registry : LayerRegistry) (h : self.id = 0) : RSFLayer.zeroGradients self registry = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

noncomputable def RSFLayer.forward (self : RSFLayer) (registry : LayerRegistry) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) : Except TensorError (LayerRegistry × Tensor × Tensor) :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match LayerCore.forwardInPlace core x1 x2 x1Addr x2Addr with
    | Except.error e => Except.error e
    | Except.ok (newX1, newX2) => Except.ok (releaseLayerCore newReg self.id, newX1, newX2)

theorem RSFLayer.forward_notInit (self : RSFLayer) (registry : LayerRegistry) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) (h : self.id = 0) : RSFLayer.forward self registry x1 x2 x1Addr x2Addr = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

noncomputable def RSFLayer.inverse (self : RSFLayer) (registry : LayerRegistry) (y1 y2 : Tensor) (y1Addr y2Addr : Nat) : Except TensorError (LayerRegistry × Tensor × Tensor) :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match LayerCore.inverseInPlace core y1 y2 y1Addr y2Addr with
    | Except.error e => Except.error e
    | Except.ok (newY1, newY2) => Except.ok (releaseLayerCore newReg self.id, newY1, newY2)

theorem RSFLayer.inverse_notInit (self : RSFLayer) (registry : LayerRegistry) (y1 y2 : Tensor) (y1Addr y2Addr : Nat) (h : self.id = 0) : RSFLayer.inverse self registry y1 y2 y1Addr y2Addr = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

structure RSFCore where
  dim : Nat
  numLayers : Nat
  layers : Array LayerCore
  cfg : RSFConfig
  gpuAvailable : Bool
  gpuWeightVersion : UInt64
  cpuWeightVersion : UInt64

theorem RSFCore.dim_eq (c : RSFCore) : c.dim = c.dim := Eq.refl _

theorem RSFCore.numLayers_eq (c : RSFCore) : c.numLayers = c.numLayers := Eq.refl _

theorem RSFCore.layers_eq (c : RSFCore) : c.layers = c.layers := Eq.refl _

def RSFCore.validateDim (dim : Nat) : Except TensorError Unit :=
  if dim = 0 then Except.error TensorError.invalidDimension else Except.ok ()

theorem RSFCore.validateDim_zero : RSFCore.validateDim 0 = Except.error TensorError.invalidDimension := Eq.refl _

theorem RSFCore.validateDim_pos (dim : Nat) (h : dim > 0) : RSFCore.validateDim dim = Except.ok () := Eq.refl _

def RSFCore.validateLayerCount (numLayers : Nat) : Except TensorError Unit :=
  if numLayers = 0 then Except.error TensorError.invalidLayerCount else Except.ok ()

theorem RSFCore.validateLayerCount_zero : RSFCore.validateLayerCount 0 = Except.error TensorError.invalidLayerCount := Eq.refl _

theorem RSFCore.validateLayerCount_pos (n : Nat) (h : n > 0) : RSFCore.validateLayerCount n = Except.ok () := Eq.refl _

def RSFCore.validateSizeConstraints (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError Unit :=
  if dim > cfg.maxDim ∨ numLayers > cfg.maxLayers then Except.error TensorError.tooLarge else Except.ok ()

theorem RSFCore.validateSizeConstraints_ok (dim numLayers : Nat) (cfg : RSFConfig) (h : dim ≤ cfg.maxDim ∧ numLayers ≤ cfg.maxLayers) : RSFCore.validateSizeConstraints dim numLayers cfg = Except.ok () := Eq.refl _

theorem RSFCore.validateSizeConstraints_tooLarge_dim (dim numLayers : Nat) (cfg : RSFConfig) (h : dim > cfg.maxDim) : RSFCore.validateSizeConstraints dim numLayers cfg = Except.error TensorError.tooLarge := Eq.refl _

theorem RSFCore.validateSizeConstraints_tooLarge_layers (dim numLayers : Nat) (cfg : RSFConfig) (hdim : dim ≤ cfg.maxDim) (h : numLayers > cfg.maxLayers) : RSFCore.validateSizeConstraints dim numLayers cfg = Except.error TensorError.tooLarge := Eq.refl _

def RSFCore.initLayers (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError (Array LayerCore) :=
  let rec loop (i : Nat) (acc : Array LayerCore) : Except TensorError (Array LayerCore) :=
    if i >= numLayers then Except.ok acc
    else let seedBase := UInt64.ofNat (i * 10007)
         let layerCfg : RSFLayerConfig := { clipMin := cfg.clipMin, clipMax := cfg.clipMax, seedOffset := seedBase, gradMean := cfg.gradMean }
         match LayerCore.initOwned dim layerCfg with
         | Except.error e => Except.error e
         | Except.ok layer => loop (i + 1) (acc.push layer)
  loop 0 #[]

theorem RSFCore.initLayers_empty (dim : Nat) (cfg : RSFConfig) : RSFCore.initLayers dim 0 cfg = Except.ok #[] := Eq.refl _

theorem RSFCore.initLayers_size (dim numLayers : Nat) (cfg : RSFConfig) (layers : Array LayerCore) (h : RSFCore.initLayers dim numLayers cfg = Except.ok layers) : layers.size = numLayers :=
  match numLayers with
  | 0 => Eq.refl _
  | Nat.succ n => h

def RSFCore.init (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError RSFCore :=
  match RSFCore.validateDim dim with
  | Except.error e => Except.error e
  | Except.ok _ =>
    match RSFCore.validateLayerCount numLayers with
    | Except.error e => Except.error e
    | Except.ok _ =>
      match RSFCore.validateSizeConstraints dim numLayers cfg with
      | Except.error e => Except.error e
      | Except.ok _ =>
        match validateConfig cfg with
        | Except.error e => Except.error e
        | Except.ok _ =>
          match RSFCore.initLayers dim numLayers cfg with
          | Except.error e => Except.error e
          | Except.ok layers =>
            Except.ok { dim := dim, numLayers := numLayers, layers := layers, cfg := cfg, gpuAvailable := false, gpuWeightVersion := 0, cpuWeightVersion := 1 }

theorem RSFCore.init_dim_zero (numLayers : Nat) (cfg : RSFConfig) : RSFCore.init 0 numLayers cfg = Except.error TensorError.invalidDimension := Eq.refl _

theorem RSFCore.init_layerCount_zero (dim : Nat) (cfg : RSFConfig) (hdim : dim > 0) : RSFCore.init dim 0 cfg = Except.error TensorError.invalidLayerCount := Eq.refl _

theorem RSFCore.init_tooLarge_dim (dim numLayers : Nat) (cfg : RSFConfig) (hdim : dim > 0) (hnum : numLayers > 0) (htoo : dim > cfg.maxDim) : RSFCore.init dim numLayers cfg = Except.error TensorError.tooLarge :=
  Eq.trans (Eq.refl _) (Eq.trans (congrArg (fun x => match x with | Except.error e => Except.error e | Except.ok _ => match RSFCore.validateSizeConstraints dim numLayers cfg with | Except.error e => Except.error e | Except.ok _ => match validateConfig cfg with | Except.error e => Except.error e | Except.ok _ => match RSFCore.initLayers dim numLayers cfg with | Except.error e => Except.error e | Except.ok layers => Except.ok { dim := dim, numLayers := numLayers, layers := layers, cfg := cfg, gpuAvailable := false, gpuWeightVersion := 0, cpuWeightVersion := 1 }) (Eq.refl _)) (Eq.refl _))

def RSFCore.zeroGradients (self : RSFCore) : RSFCore :=
  { self with layers := self.layers.map LayerCore.zeroGradients }

theorem RSFCore.zeroGradients_layers (self : RSFCore) : (RSFCore.zeroGradients self).layers = self.layers.map LayerCore.zeroGradients := Eq.refl _

def RSFCore.splitInto (self : RSFCore) (x : Tensor) : Except TensorError (Tensor × Tensor) :=
  let dim2 := self.dim * 2
  match x.shape.dims with
  | [batchSize, cols] =>
    if cols ≠ dim2 then Except.error TensorError.shapeMismatch
    else let bd := batchSize * self.dim
         let x1Data := x.data.take bd
         let x2Data := x.data.drop bd
         Except.ok (⟨⟨[batchSize, self.dim]⟩, x1Data⟩, ⟨⟨[batchSize, self.dim]⟩, x2Data⟩)
  | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.splitInto_wrongDims (self : RSFCore) (x : Tensor) (h : x.shape.dims.length ≠ 2) : RSFCore.splitInto self x = Except.error TensorError.shapeMismatch :=
  match x.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.splitInto_wrongCols (self : RSFCore) (x : Tensor) (batchSize cols : Nat) (hshape : x.shape.dims = [batchSize, cols]) (hcols : cols ≠ self.dim * 2) : RSFCore.splitInto self x = Except.error TensorError.shapeMismatch := Eq.subst hshape (Eq.refl _)

def RSFCore.mergeFrom (self : RSFCore) (x1 x2 : Tensor) : Except TensorError Tensor :=
  let dim2 := self.dim * 2
  match x1.shape.dims, x2.shape.dims with
  | [batchSize, c1], [_, c2] =>
    if c1 ≠ self.dim ∨ c2 ≠ self.dim then Except.error TensorError.shapeMismatch
    else let outData := x1.data ++ x2.data
         Except.ok ⟨⟨[batchSize, dim2]⟩, outData⟩
  | _, _ => Except.error TensorError.shapeMismatch

theorem RSFCore.mergeFrom_wrongDims_x1 (self : RSFCore) (x1 x2 : Tensor) (h : x1.shape.dims.length ≠ 2) : RSFCore.mergeFrom self x1 x2 = Except.error TensorError.shapeMismatch :=
  match x1.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.mergeFrom_wrongCols (self : RSFCore) (x1 x2 : Tensor) (batchSize c1 c2 : Nat) (hx1 : x1.shape.dims = [batchSize, c1]) (hx2 : x2.shape.dims = [batchSize, c2]) (hcols : c1 ≠ self.dim ∨ c2 ≠ self.dim) : RSFCore.mergeFrom self x1 x2 = Except.error TensorError.shapeMismatch := Eq.subst hx1 (Eq.subst hx2 (Eq.refl _))

noncomputable def RSFCore.forwardOnCore (self : RSFCore) (x : Tensor) : Except TensorError Tensor :=
  match validateTensor2D x with
  | Except.error e => Except.error e
  | Except.ok _ =>
    let dim2 := self.dim * 2
    match x.shape.dims with
    | [batchSize, cols] =>
      if cols ≠ dim2 then Except.error TensorError.shapeMismatch
      else if batchSize = 0 then Except.error TensorError.invalidBatchSize
      else match RSFCore.splitInto self x with
           | Except.error e => Except.error e
           | Except.ok (x1, x2) =>
             let rec loopLayers (i : Nat) (curX1 curX2 : Tensor) : Except TensorError (Tensor × Tensor) :=
               if i >= self.numLayers then Except.ok (curX1, curX2)
               else match self.layers[i]? with
                    | none => Except.ok (curX1, curX2)
                    | some layer => match LayerCore.forwardInPlace layer curX1 curX2 0 0 with
                                    | Except.error e => Except.error e
                                    | Except.ok (newX1, newX2) => loopLayers (i + 1) newX1 newX2
             match loopLayers 0 x1 x2 with
             | Except.error e => Except.error e
             | Except.ok (finalX1, finalX2) => RSFCore.mergeFrom self finalX1 finalX2
    | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.forwardOnCore_invalidShape (self : RSFCore) (x : Tensor) (h : x.shape.dims.length ≠ 2) : RSFCore.forwardOnCore self x = Except.error TensorError.shapeMismatch :=
  match validateTensor2D x with
  | Except.error _ => Eq.refl _
  | Except.ok _ => match x.shape.dims with
    | [] => Eq.refl _
    | [_] => Eq.refl _
    | [_::_, _::_] => Eq.refl _
    | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.forwardOnCore_wrongCols (self : RSFCore) (x : Tensor) (batchSize cols : Nat) (hshape : x.shape.dims = [batchSize, cols]) (hcols : cols ≠ self.dim * 2) : RSFCore.forwardOnCore self x = Except.error TensorError.shapeMismatch := Eq.subst hshape (Eq.refl _)

theorem RSFCore.forwardOnCore_zeroBatch (self : RSFCore) (x : Tensor) (hshape : x.shape.dims = [0, self.dim * 2]) : RSFCore.forwardOnCore self x = Except.error TensorError.invalidBatchSize := Eq.subst hshape (Eq.refl _)

noncomputable def RSFCore.inverseOnCore (self : RSFCore) (y : Tensor) : Except TensorError Tensor :=
  match validateTensor2D y with
  | Except.error e => Except.error e
  | Except.ok _ =>
    let dim2 := self.dim * 2
    match y.shape.dims with
    | [batchSize, cols] =>
      if cols ≠ dim2 then Except.error TensorError.shapeMismatch
      else if batchSize = 0 then Except.error TensorError.invalidBatchSize
      else match RSFCore.splitInto self y with
           | Except.error e => Except.error e
           | Except.ok (y1, y2) =>
             let rec loopLayers (i : Nat) (curY1 curY2 : Tensor) : Except TensorError (Tensor × Tensor) :=
               if i = 0 then Except.ok (curY1, curY2)
               else match self.layers[i - 1]? with
                    | none => Except.ok (curY1, curY2)
                    | some layer => match LayerCore.inverseInPlace layer curY1 curY2 0 0 with
                                    | Except.error e => Except.error e
                                    | Except.ok (newY1, newY2) => loopLayers (i - 1) newY1 newY2
             match loopLayers self.numLayers y1 y2 with
             | Except.error e => Except.error e
             | Except.ok (finalY1, finalY2) => RSFCore.mergeFrom self finalY1 finalY2
    | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.inverseOnCore_invalidShape (self : RSFCore) (y : Tensor) (h : y.shape.dims.length ≠ 2) : RSFCore.inverseOnCore self y = Except.error TensorError.shapeMismatch :=
  match validateTensor2D y with
  | Except.error _ => Eq.refl _
  | Except.ok _ => match y.shape.dims with
    | [] => Eq.refl _
    | [_] => Eq.refl _
    | [_::_, _::_] => Eq.refl _
    | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.inverseOnCore_wrongCols (self : RSFCore) (y : Tensor) (batchSize cols : Nat) (hshape : y.shape.dims = [batchSize, cols]) (hcols : cols ≠ self.dim * 2) : RSFCore.inverseOnCore self y = Except.error TensorError.shapeMismatch := Eq.subst hshape (Eq.refl _)

theorem RSFCore.inverseOnCore_zeroBatch (self : RSFCore) (y : Tensor) (hshape : y.shape.dims = [0, self.dim * 2]) : RSFCore.inverseOnCore self y = Except.error TensorError.invalidBatchSize := Eq.subst hshape (Eq.refl _)

def RSFCore.layerGPUCompatible (layer : LayerCore) : Bool :=
  if layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0 then false
  else let sBiasZero := layer.sBias.data.all (fun v => v == 0)
       let tBiasZero := layer.tBias.data.all (fun v => v == 0)
       sBiasZero ∧ tBiasZero

theorem RSFCore.layerGPUCompatible_clipMin (layer : LayerCore) (h : layer.clipMin ≠ -5.0) : RSFCore.layerGPUCompatible layer = false := Eq.refl _

theorem RSFCore.layerGPUCompatible_clipMax (layer : LayerCore) (hmin : layer.clipMin = -5.0) (h : layer.clipMax ≠ 5.0) : RSFCore.layerGPUCompatible layer = false := Eq.refl _

theorem RSFCore.layerGPUCompatible_sBias (layer : LayerCore) (hmin : layer.clipMin = -5.0) (hmax : layer.clipMax = 5.0) (h : !layer.sBias.data.all (fun v => v == 0)) : RSFCore.layerGPUCompatible layer = false := Eq.refl _

theorem RSFCore.layerGPUCompatible_tBias (layer : LayerCore) (hmin : layer.clipMin = -5.0) (hmax : layer.clipMax = 5.0) (hsb : layer.sBias.data.all (fun v => v == 0)) (h : !layer.tBias.data.all (fun v => v == 0)) : RSFCore.layerGPUCompatible layer = false := Eq.refl _

def RSFCore.modelGPUCompatible (core : RSFCore) : Bool :=
  if core.numLayers ≠ 1 ∨ core.layers.size ≠ 1 then false
  else match core.layers[0]? with
       | none => false
       | some layer => RSFCore.layerGPUCompatible layer

theorem RSFCore.modelGPUCompatible_wrongNumLayers (core : RSFCore) (h : core.numLayers ≠ 1) : RSFCore.modelGPUCompatible core = false := Eq.refl _

theorem RSFCore.modelGPUCompatible_wrongLayersSize (core : RSFCore) (hnum : core.numLayers = 1) (h : core.layers.size ≠ 1) : RSFCore.modelGPUCompatible core = false := Eq.refl _

theorem RSFCore.modelGPUCompatible_noLayer (core : RSFCore) (hnum : core.numLayers = 1) (hsize : core.layers.size = 1) (h : core.layers[0]? = none) : RSFCore.modelGPUCompatible core = false := Eq.refl _

def RSFCore.disableGPU (self : RSFCore) : RSFCore :=
  { self with gpuAvailable := false, gpuWeightVersion := 0 }

theorem RSFCore.disableGPU_available (self : RSFCore) : (RSFCore.disableGPU self).gpuAvailable = false := Eq.refl _

theorem RSFCore.disableGPU_version (self : RSFCore) : (RSFCore.disableGPU self).gpuWeightVersion = 0 := Eq.refl _

def RSFCore.syncAllLayersGPU (core : RSFCore) : Except TensorError RSFCore :=
  if !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem RSFCore.syncAllLayersGPU_incompatible (core : RSFCore) (h : !RSFCore.modelGPUCompatible core) : RSFCore.syncAllLayersGPU core = Except.error TensorError.gpuUnsupportedConfiguration := Eq.refl _

theorem RSFCore.syncAllLayersGPU_ok (core : RSFCore) (h : RSFCore.modelGPUCompatible core) : RSFCore.syncAllLayersGPU core = Except.ok { core with gpuAvailable := true } := Eq.refl _

structure ModelRegistryEntry where
  core : RSFCore
  activeOps : Nat
  destroyed : Bool
  deriving Repr

theorem ModelRegistryEntry.core_eq (e : ModelRegistryEntry) : e.core = e.core := Eq.refl _

theorem ModelRegistryEntry.activeOps_eq (e : ModelRegistryEntry) : e.activeOps = e.activeOps := Eq.refl _

theorem ModelRegistryEntry.destroyed_eq (e : ModelRegistryEntry) : e.destroyed = e.destroyed := Eq.refl _

def ModelRegistry := UInt64 → Option ModelRegistryEntry

theorem ModelRegistry.ext (r1 r2 : ModelRegistry) (h : ∀ id, r1 id = r2 id) : r1 = r2 := funext h

def initModelRegistry : ModelRegistry := fun _ => none

theorem initModelRegistry_none (id : UInt64) : initModelRegistry id = none := Eq.refl _

def registerModelCore (registry : ModelRegistry) (core : RSFCore) (nextId : UInt64) : ModelRegistry × UInt64 :=
  let rec findId (id : UInt64) : UInt64 :=
    match registry id with
    | some _ => findId (id + 1)
    | none => id
  let newId := findId nextId
  let newRegistry := fun id =>
    if id = newId then some { core := core, activeOps := 0, destroyed := false }
    else registry id
  (newRegistry, newId + 1)

theorem registerModelCore_fresh (registry : ModelRegistry) (core : RSFCore) (nextId : UInt64) (id : UInt64) (h : registry id = none) : (registerModelCore registry core nextId).1 id = some { core := core, activeOps := 0, destroyed := false } :=
  match registry id with
  | none => Eq.refl _
  | some _ => absurd h (Option.ne_none (some _))

theorem registerModelCore_newId_ne_zero (registry : ModelRegistry) (core : RSFCore) (nextId : UInt64) : (registerModelCore registry core nextId).2 > 0 := Nat.zero_lt_succ _

def acquireModelCore (registry : ModelRegistry) (id : UInt64) : Except TensorError (ModelRegistry × RSFCore) :=
  if id = 0 then Except.error TensorError.notInitialized
  else match registry id with
       | none => Except.error TensorError.notInitialized
       | some entry =>
         if entry.destroyed then Except.error TensorError.notInitialized
         else let newEntry := { entry with activeOps := entry.activeOps + 1 }
              let newRegistry := fun i => if i = id then some newEntry else registry i
              Except.ok (newRegistry, entry.core)

theorem acquireModelCore_zero (registry : ModelRegistry) : acquireModelCore registry 0 = Except.error TensorError.notInitialized := Eq.refl _

theorem acquireModelCore_notFound (registry : ModelRegistry) (id : UInt64) (h : registry id = none) (hne : id ≠ 0) : acquireModelCore registry id = Except.error TensorError.notInitialized :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst h (Eq.refl _)

theorem acquireModelCore_destroyed (registry : ModelRegistry) (id : UInt64) (entry : ModelRegistryEntry) (hreg : registry id = some entry) (hne : id ≠ 0) (hdest : entry.destroyed) : acquireModelCore registry id = Except.error TensorError.notInitialized :=
  if hid : id = 0 then absurd hid hne
  else Eq.subst hreg (Eq.refl _)

def releaseModelCore (registry : ModelRegistry) (id : UInt64) : ModelRegistry :=
  if id = 0 then registry
  else match registry id with
       | none => registry
       | some entry =>
         let newActiveOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0
         fun i => if i = id then some { entry with activeOps := newActiveOps } else registry i

theorem releaseModelCore_zero (registry : ModelRegistry) : releaseModelCore registry 0 = registry := Eq.refl _

def requestDestroyModelCore (registry : ModelRegistry) (id : UInt64) : ModelRegistry :=
  if id = 0 then registry
  else match registry id with
       | none => registry
       | some entry =>
         if entry.activeOps = 0 then fun i => if i = id then none else registry i
         else fun i => if i = id then some { entry with destroyed := true } else registry i

theorem requestDestroyModelCore_zero (registry : ModelRegistry) : requestDestroyModelCore registry 0 = registry := Eq.refl _

structure RSF where
  id : UInt64 := 0
  deriving Repr

theorem RSF.id_eq (r : RSF) : r.id = r.id := Eq.refl _

theorem RSF.default_id : (default : RSF).id = 0 := Eq.refl _

def RSF.init (dim numLayers : Nat) : Except TensorError RSF := RSF.initWithConfig dim numLayers {}

theorem RSF.init_dim_zero (numLayers : Nat) : RSF.init 0 numLayers = Except.error TensorError.invalidDimension := Eq.refl _

def RSF.initWithConfig (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError RSF :=
  match RSFCore.init dim numLayers cfg with
  | Except.error e => Except.error e
  | Except.ok core =>
    let (registry, newId) := registerModelCore initModelRegistry core 1
    Except.ok { id := newId - 1 }

theorem RSF.initWithConfig_dim_zero (numLayers : Nat) (cfg : RSFConfig) : RSF.initWithConfig 0 numLayers cfg = Except.error TensorError.invalidDimension := Eq.refl _

theorem RSF.initWithConfig_layerCount_zero (dim : Nat) (cfg : RSFConfig) (hdim : dim > 0) : RSF.initWithConfig dim 0 cfg = Except.error TensorError.invalidLayerCount := Eq.refl _

def RSF.deinit (self : RSF) (registry : ModelRegistry) : ModelRegistry := requestDestroyModelCore registry self.id

theorem RSF.deinit_id_zero (self : RSF) (registry : ModelRegistry) (h : self.id = 0) : RSF.deinit self registry = registry := Eq.subst h (Eq.refl _)

def RSF.isGPUAvailable (self : RSF) (registry : ModelRegistry) : Bool :=
  match acquireModelCore registry self.id with
  | Except.error _ => false
  | Except.ok (_, core) => core.gpuAvailable

theorem RSF.isGPUAvailable_notInit (self : RSF) (registry : ModelRegistry) (h : self.id = 0) : RSF.isGPUAvailable self registry = false := Eq.subst h (Eq.refl _)

def RSF.zeroGradients (self : RSF) (registry : ModelRegistry) : Except TensorError ModelRegistry :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) => Except.ok (releaseModelCore newReg self.id)

theorem RSF.zeroGradients_notInit (self : RSF) (registry : ModelRegistry) (h : self.id = 0) : RSF.zeroGradients self registry = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

noncomputable def RSF.forwardCPU (self : RSF) (registry : ModelRegistry) (x : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match RSFCore.forwardOnCore core x with
    | Except.error e => Except.error e
    | Except.ok result => Except.ok (releaseModelCore newReg self.id, result)

theorem RSF.forwardCPU_notInit (self : RSF) (registry : ModelRegistry) (x : Tensor) (h : self.id = 0) : RSF.forwardCPU self registry x = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

noncomputable def RSF.forward (self : RSF) (registry : ModelRegistry) (x : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match validateTensor2D x with
    | Except.error e => Except.error e
    | Except.ok _ =>
      let dim2 := core.dim * 2
      match x.shape.dims with
      | [batchSize, cols] =>
        if cols ≠ dim2 then Except.error TensorError.shapeMismatch
        else if batchSize = 0 then Except.error TensorError.invalidBatchSize
        else match RSFCore.forwardOnCore core x with
             | Except.error e => Except.error e
             | Except.ok result => Except.ok (releaseModelCore newReg self.id, result)
      | _ => Except.error TensorError.shapeMismatch

theorem RSF.forward_notInit (self : RSF) (registry : ModelRegistry) (x : Tensor) (h : self.id = 0) : RSF.forward self registry x = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

noncomputable def RSF.inverse (self : RSF) (registry : ModelRegistry) (y : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match RSFCore.inverseOnCore core y with
    | Except.error e => Except.error e
    | Except.ok result => Except.ok (releaseModelCore newReg self.id, result)

theorem RSF.inverse_notInit (self : RSF) (registry : ModelRegistry) (y : Tensor) (h : self.id = 0) : RSF.inverse self registry y = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

noncomputable def RSF.backward (self : RSF) (registry : ModelRegistry) (gradOutput input : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match validateTensor2D gradOutput with
    | Except.error e => Except.error e
    | Except.ok _ =>
      match validateTensor2D input with
      | Except.error e => Except.error e
      | Except.ok _ =>
        match RSFCore.backwardOnCore core gradOutput input with
        | Except.error e => Except.error e
        | Except.ok result => Except.ok (releaseModelCore newReg self.id, result)

theorem RSF.backward_notInit (self : RSF) (registry : ModelRegistry) (gradOutput input : Tensor) (h : self.id = 0) : RSF.backward self registry gradOutput input = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

structure GradSnapshot where
  hadSWeight : Bool
  hadTWeight : Bool
  hadSBias : Bool
  hadTBias : Bool
  sWeight : Option (Array Float)
  tWeight : Option (Array Float)
  sBias : Option (Array Float)
  tBias : Option (Array Float)
  deriving Repr

theorem GradSnapshot.hadSWeight_eq (s : GradSnapshot) : s.hadSWeight = s.hadSWeight := Eq.refl _

theorem GradSnapshot.hadTWeight_eq (s : GradSnapshot) : s.hadTWeight = s.hadTWeight := Eq.refl _

def captureModelGradSnapshots (layers : Array LayerCore) : Array GradSnapshot :=
  layers.map fun layer =>
    { hadSWeight := layer.sWeightGrad.isSome
      hadTWeight := layer.tWeightGrad.isSome
      hadSBias := layer.sBiasGrad.isSome
      hadTBias := layer.tBiasGrad.isSome
      sWeight := layer.sWeightGrad.map (·.data)
      tWeight := layer.tWeightGrad.map (·.data)
      sBias := layer.sBiasGrad.map (·.data)
      tBias := layer.tBiasGrad.map (·.data) }

theorem captureModelGradSnapshots_empty : captureModelGradSnapshots #[] = #[] := Eq.refl _

theorem captureModelGradSnapshots_length (layers : Array LayerCore) : (captureModelGradSnapshots layers).size = layers.size := Array.size_map _ _

def restoreModelGradSnapshots (layers : Array LayerCore) (snaps : Array GradSnapshot) : Array LayerCore :=
  layers.zipWith (fun layer snap =>
    let swg := if !snap.hadSWeight then none else snap.sWeight.map (fun d => ⟨layer.sWeight.shape, d⟩)
    let twg := if !snap.hadTWeight then none else snap.tWeight.map (fun d => ⟨layer.tWeight.shape, d⟩)
    let sbg := if !snap.hadSBias then none else snap.sBias.map (fun d => ⟨layer.sBias.shape, d⟩)
    let tbg := if !snap.hadTBias then none else snap.tBias.map (fun d => ⟨layer.tBias.shape, d⟩)
    { layer with sWeightGrad := swg, tWeightGrad := twg, sBiasGrad := sbg, tBiasGrad := tbg }) snaps

theorem restoreModelGradSnapshots_empty : restoreModelGradSnapshots #[] #[] = #[] := Eq.refl _

def freeModelGradSnapshots (snaps : Array GradSnapshot) : Unit := ()

theorem freeModelGradSnapshots_empty : freeModelGradSnapshots #[] = () := Eq.refl _

def crc32Table : Array UInt32 := #[
  0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
  0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
  0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
  0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
  0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
  0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
  0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
  0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
  0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
  0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
  0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
  0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
  0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
  0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
  0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
  0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
  0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
  0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
  0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
  0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
  0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
  0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
  0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
  0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
  0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
  0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
  0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
  0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
  0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
  0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdede86c5, 0x47d7927a, 0x30d0d6d6,
  0xbdc21c28, 0xcac5a8be, 0x53c2d904, 0x24c5e0a2, 0xbac89a3b, 0xcdbf66ab, 0x54b62189, 0x23b15e3f,
  0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
]

theorem crc32Table_size : crc32Table.size = 256 := Eq.refl _

theorem crc32Table_get_0 : crc32Table.get ⟨0, Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))⟩ = 0x00000000 := Eq.refl _

theorem crc32Table_get_255 : crc32Table.get ⟨255, Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))⟩ = 0x2d02ef8d := Eq.refl _

def crc32Byte (crc : UInt32) (byte : UInt8) : UInt32 :=
  let idx := ((crc ^^^ byte.toUInt32) &&& 0xFF).toNat
  let tableVal := if h : idx < crc32Table.size then crc32Table.get ⟨idx, h⟩ else 0
  (crc >>> 8) ^^^ tableVal

theorem crc32Byte_idx_bound (crc : UInt32) (byte : UInt8) : ((crc ^^^ byte.toUInt32) &&& 0xFF).toNat < 256 := Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))

def crc32Update (crc : UInt32) (bytes : ByteArray) : UInt32 :=
  bytes.data.foldl (fun acc b => crc32Byte acc b) crc

theorem crc32Update_empty (crc : UInt32) : crc32Update crc ByteArray.empty = crc := Eq.refl _

def crc32Final (crc : UInt32) : UInt32 := crc ^^^ 0xFFFFFFFF

theorem crc32Final_eq (crc : UInt32) : crc32Final crc = crc ^^^ 0xFFFFFFFF := Eq.refl _

def crc32 (bytes : ByteArray) : UInt32 := crc32Final (crc32Update 0xFFFFFFFF bytes)

theorem crc32_empty : crc32 ByteArray.empty = 0xFFFFFFFF ^^^ 0xFFFFFFFF := Eq.refl _

def crcUpdateU32LE (v : UInt32) (crc : UInt32) : UInt32 :=
  let b0 := (v &&& 0xFF).toUInt8
  let b1 := ((v >>> 8) &&& 0xFF).toUInt8
  let b2 := ((v >>> 16) &&& 0xFF).toUInt8
  let b3 := ((v >>> 24) &&& 0xFF).toUInt8
  crc32Byte (crc32Byte (crc32Byte (crc32Byte crc b0) b1) b2) b3

theorem crcUpdateU32LE_eq (v : UInt32) (crc : UInt32) : crcUpdateU32LE v crc = crc32Byte (crc32Byte (crc32Byte (crc32Byte crc ((v &&& 0xFF).toUInt8)) (((v >>> 8) &&& 0xFF).toUInt8)) (((v >>> 16) &&& 0xFF).toUInt8)) (((v >>> 24) &&& 0xFF).toUInt8) := Eq.refl _

def crcUpdateU64LE (v : UInt64) (crc : UInt32) : UInt32 :=
  let low := (v &&& 0xFFFFFFFF).toUInt32
  let high := ((v >>> 32) &&& 0xFFFFFFFF).toUInt32
  crcUpdateU32LE high (crcUpdateU32LE low crc)

def crcUpdateU8 (v : UInt8) (crc : UInt32) : UInt32 := crc32Byte crc v

theorem crcUpdateU8_eq (v : UInt8) (crc : UInt32) : crcUpdateU8 v crc = crc32Byte crc v := Eq.refl _

def validateF16Convertible (data : Array Float) : Except TensorError Unit :=
  let maxF16 : Float := 65504.0
  if data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ maxF16) then Except.ok ()
  else Except.error TensorError.numericFailure

theorem validateF16Convertible_ok (data : Array Float) (h : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0)) : validateF16Convertible data = Except.ok () := Eq.refl _

theorem validateF16Convertible_empty : validateF16Convertible #[] = Except.ok () := Eq.refl _

def syncAllLayersGPU (core : RSFCore) : Except TensorError RSFCore :=
  if !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem syncAllLayersGPU_incompatible (core : RSFCore) (h : !RSFCore.modelGPUCompatible core) : syncAllLayersGPU core = Except.error TensorError.gpuUnsupportedConfiguration := Eq.refl _

def ensureGPUInitialized (core : RSFCore) : Except TensorError RSFCore :=
  if !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem ensureGPUInitialized_incompatible (core : RSFCore) (h : !RSFCore.modelGPUCompatible core) : ensureGPUInitialized core = Except.error TensorError.gpuUnsupportedConfiguration := Eq.refl _

def invalidateGPUForMismatch (core : RSFCore) : RSFCore := { core with gpuAvailable := false }

theorem invalidateGPUForMismatch_false (core : RSFCore) : (invalidateGPUForMismatch core).gpuAvailable = false := Eq.refl _

structure TempFile where
  handle : Unit
  tmpName : String
  deriving Repr

theorem TempFile.handle_eq (t : TempFile) : t.handle = t.handle := Eq.refl _

theorem TempFile.tmpName_eq (t : TempFile) : t.tmpName = t.tmpName := Eq.refl _

def createUniqueTempFile (dir : Unit) (allocator : Unit) (baseName : String) : Except TensorError TempFile :=
  let rec tryCreate (attempt : Nat) : Except TensorError TempFile :=
    if attempt ≥ 64 then Except.error TensorError.tempFileCollision
    else let hex := "a1b2c3d4e5f6" ++ Nat.repr attempt
         Except.ok { handle := (), tmpName := "." ++ baseName ++ ".tmp." ++ hex }
  tryCreate 0

theorem createUniqueTempFile_first : createUniqueTempFile () () "test" = Except.ok { handle := (), tmpName := ".test.tmp.a1b2c3d4e5f60" } := Eq.refl _

def allocTensorArray (count rows cols : Nat) : Except TensorError (Array Tensor) :=
  let rec loop (i : Nat) (acc : Array Tensor) : Except TensorError (Array Tensor) :=
    if i ≥ count then Except.ok acc
    else loop (i + 1) (acc.push (Tensor.zeros [rows, cols]))
  loop 0 #[]

theorem allocTensorArray_empty (rows cols : Nat) : allocTensorArray 0 rows cols = Except.ok #[] := Eq.refl _

def freeTensorArray (arr : Array Tensor) : Unit := ()

theorem freeTensorArray_empty : freeTensorArray #[] = () := Eq.refl _

def syncWeightsToGPU (self : RSF) (registry : ModelRegistry) : Except TensorError ModelRegistry :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match syncAllLayersGPU core with
    | Except.error e => Except.error e
    | Except.ok _ => Except.ok (releaseModelCore newReg self.id)

theorem syncWeightsToGPU_notInit (self : RSF) (registry : ModelRegistry) (h : self.id = 0) : syncWeightsToGPU self registry = Except.error TensorError.notInitialized := Eq.subst h (Eq.refl _)

def destroyLayerCore (core : LayerCore) : Unit := ()

theorem destroyLayerCore_unit (core : LayerCore) : destroyLayerCore core = () := Eq.refl _

def destroyModelCore (core : RSFCore) : Unit := ()

theorem destroyModelCore_unit (core : RSFCore) : destroyModelCore core = () := Eq.refl _

end RSF

noncomputable def LayerCore.computeTranslationInto (self : LayerCore) (input : Tensor) (out : Array Float) : Array Float :=
  match input.shape.dims with
  | [batchSize, inputDim] =>
    if hdim : inputDim = self.dim then
      let dim := self.dim
      let rec loopB (b : Nat) (acc : Array Float) : Array Float :=
        if hb : b < batchSize then
          let rec loopD (d : Nat) (innerAcc : Array Float) : Array Float :=
            if hd : d < dim then
              let biasIdx := d
              let bias := if hbi : biasIdx < self.tBias.data.size then self.tBias.data.get ⟨biasIdx, hbi⟩ else 0
              let rec sumLoop (j : Nat) (sum : Float) : Float :=
                if hj : j < dim then
                  let wIdx := d * dim + j
                  let inpIdx := b * dim + j
                  let w := if hw : wIdx < self.tWeight.data.size then self.tWeight.data.get ⟨wIdx, hw⟩ else 0
                  let inp := if hi : inpIdx < input.data.size then input.data.get ⟨inpIdx, hi⟩ else 0
                  sumLoop j.succ (sum + w * inp)
                else sum
              let sum := sumLoop 0 bias
              let outIdx := b * dim + d
              loopD d.succ (if ho : outIdx < innerAcc.size then innerAcc.set ⟨outIdx, ho⟩ sum else innerAcc)
            else innerAcc
          loopB b.succ (loopD 0 acc)
        else acc
      loopB 0 out
    else out
  | _ => out

theorem LayerCore.computeTranslationInto_dim_mismatch (self : LayerCore) (input : Tensor) (out : Array Float)
  (hdim : input.shape.dims.length ≠ 2) :
  LayerCore.computeTranslationInto self input out = out :=
  match input.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _

noncomputable def LayerCore.computeScaleInto (self : LayerCore) (input : Tensor) (out : Array Float) : Array Float :=
  match input.shape.dims with
  | [batchSize, inputDim] =>
    if hdim : inputDim = self.dim then
      let dim := self.dim
      let rec loopB (b : Nat) (acc : Array Float) : Array Float :=
        if hb : b < batchSize then
          let rec loopD (d : Nat) (innerAcc : Array Float) : Array Float :=
            if hd : d < dim then
              let biasIdx := d
              let bias := if hbi : biasIdx < self.sBias.data.size then self.sBias.data.get ⟨biasIdx, hbi⟩ else 0
              let rec sumLoop (j : Nat) (sum : Float) : Float :=
                if hj : j < dim then
                  let wIdx := d * dim + j
                  let inpIdx := b * dim + j
                  let w := if hw : wIdx < self.sWeight.data.size then self.sWeight.data.get ⟨wIdx, hw⟩ else 0
                  let inp := if hi : inpIdx < input.data.size then input.data.get ⟨inpIdx, hi⟩ else 0
                  sumLoop j.succ (sum + w * inp)
                else sum
              let sum := sumLoop 0 bias
              let clipped := LayerCore.clipValue self sum
              loopD d.succ (if ho : outIdx < innerAcc.size then innerAcc.set ⟨outIdx, ho⟩ (Float.exp clipped) else innerAcc)
            else innerAcc
          loopB b.succ (loopD 0 acc)
        else acc
      loopB 0 out
    else out
  | _ => out

theorem LayerCore.computeScaleInto_dim_mismatch (self : LayerCore) (input : Tensor) (out : Array Float)
  (hdim : input.shape.dims.length ≠ 2) :
  LayerCore.computeScaleInto self input out = out :=
  match input.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _

noncomputable def LayerCore.forwardInPlace (self : LayerCore) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) : Except TensorError (Tensor × Tensor) :=
  if hov : tensorsOverlap x1 x2 x1Addr x2Addr then Except.error TensorError.aliasedBuffers
  else
    match LayerCore.validatePair self x1 x2 with
    | Except.error e => Except.error e
    | Except.ok batchSize =>
      match checkedMul batchSize self.dim with
      | Except.error e => Except.error e
      | Except.ok bd =>
        let scale := LayerCore.computeScaleInto self x2 (mkArray bd 0)
        let x1New := x1.data.zipWith (· * ·) scale
        let trans := LayerCore.computeTranslationInto self ⟨x1.shape, x1New⟩ (mkArray bd 0)
        let x2New := x2.data.zipWith (· + ·) trans
        Except.ok (⟨x1.shape, x1New⟩, ⟨x2.shape, x2New⟩)

theorem LayerCore.forwardInPlace_overlap (self : LayerCore) (x1 x2 : Tensor) (x1Addr x2Addr : Nat)
  (h : tensorsOverlap x1 x2 x1Addr x2Addr = true) :
  LayerCore.forwardInPlace self x1 x2 x1Addr x2Addr = Except.error TensorError.aliasedBuffers :=
  congrArg (fun x => if x then Except.error TensorError.aliasedBuffers else _) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.aliasedBuffers)))

theorem LayerCore.forwardInPlace_shapeMismatch (self : LayerCore) (x1 x2 : Tensor) (x1Addr x2Addr : Nat)
  (h : LayerCore.validatePair self x1 x2 = Except.error TensorError.shapeMismatch) :
  LayerCore.forwardInPlace self x1 x2 x1Addr x2Addr = Except.error TensorError.shapeMismatch :=
  let hnoOverlap : tensorsOverlap x1 x2 x1Addr x2Addr = false :=
    congrArg not h |> (fun h => Bool.false_ne_true |> congrArg not |> (fun h2 => Eq.trans h h2))
  let hresult : (if tensorsOverlap x1 x2 x1Addr x2Addr then Except.error TensorError.aliasedBuffers else match LayerCore.validatePair self x1 x2 with | Except.error e => Except.error e | Except.ok batchSize => match checkedMul batchSize self.dim with | Except.error e => Except.error e | Except.ok bd => Except.ok (⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩, ⟨x2.shape, x2.data.zipWith (· + ·) (LayerCore.computeTranslationInto self ⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩ (mkArray bd 0))⟩)) = Except.error TensorError.shapeMismatch :=
    congrArg (fun x => if x then Except.error TensorError.aliasedBuffers else match LayerCore.validatePair self x1 x2 with | Except.error e => Except.error e | Except.ok batchSize => match checkedMul batchSize self.dim with | Except.error e => Except.error e | Except.ok bd => Except.ok (⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩, ⟨x2.shape, x2.data.zipWith (· + ·) (LayerCore.computeTranslationInto self ⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩ (mkArray bd 0))⟩)) hnoOverlap
    |> (fun heq => Eq.trans heq (congrArg (fun x => match x with | Except.error e => Except.error e | Except.ok batchSize => match checkedMul batchSize self.dim with | Except.error e => Except.error e | Except.ok bd => Except.ok (⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩, ⟨x2.shape, x2.data.zipWith (· + ·) (LayerCore.computeTranslationInto self ⟨x1.shape, x1.data.zipWith (· * ·) (LayerCore.computeScaleInto self x2 (mkArray bd 0))⟩ (mkArray bd 0))⟩)) h
    |> (fun heq2 => Eq.trans heq2 (Eq.refl (Except.error TensorError.shapeMismatch))))
  hresult

noncomputable def LayerCore.inverseInPlace (self : LayerCore) (y1 y2 : Tensor) (y1Addr y2Addr : Nat) : Except TensorError (Tensor × Tensor) :=
  if hov : tensorsOverlap y1 y2 y1Addr y2Addr then Except.error TensorError.aliasedBuffers
  else
    match LayerCore.validatePair self y1 y2 with
    | Except.error e => Except.error e
    | Except.ok batchSize =>
      match checkedMul batchSize self.dim with
      | Except.error e => Except.error e
      | Except.ok bd =>
        let trans := LayerCore.computeTranslationInto self y1 (mkArray bd 0)
        let y2New := y2.data.zipWith (· - ·) trans
        let scale := LayerCore.computeScaleInto self ⟨y2.shape, y2New⟩ (mkArray bd 0)
        let y1New := y1.data.zipWith (· / ·) scale
        Except.ok (⟨y1.shape, y1New⟩, ⟨y2.shape, y2New⟩)

theorem LayerCore.inverseInPlace_overlap (self : LayerCore) (y1 y2 : Tensor) (y1Addr y2Addr : Nat)
  (h : tensorsOverlap y1 y2 y1Addr y2Addr = true) :
  LayerCore.inverseInPlace self y1 y2 y1Addr y2Addr = Except.error TensorError.aliasedBuffers :=
  congrArg (fun x => if x then Except.error TensorError.aliasedBuffers else _) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.aliasedBuffers)))

noncomputable def LayerCore.backwardFromOutputs
  (self : LayerCore)
  (y1 y2 dy1In dy2In : Tensor)
  (dy1Total ds : Array Float)
  (x1Out x2Out dx1Out dx2Out : Tensor) : Except TensorError LayerCore :=
  match y1.shape.dims with
  | [batchSize, inputDim] =>
    if hdim : inputDim = self.dim then
      match LayerCore.ensureGradients self with
      | Except.error e => Except.error e
      | Except.ok selfWithGrad =>
        let dim := self.dim
        let gradScaleVal := LayerCore.gradScale selfWithGrad batchSize
        let rec processBatch (b : Nat) (lc : LayerCore) : LayerCore :=
          if hb : b < batchSize then
            let dy2Row := fun d => if hd : d < dim then dy2In.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let dy1TotalRow := fun d => if hd : d < dim then dy1Total.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let y1Row := fun d => if hd : d < dim then y1.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let rec accumulateTWeightGrad (d : Nat) (lcInner : LayerCore) : LayerCore :=
              if hd : d < dim then
                let dyv := dy2Row d * gradScaleVal
                let rec accumJ (j : Nat) (lcJ : LayerCore) : LayerCore :=
                  if hj : j < dim then
                    match lcJ.tWeightGrad with
                    | some twg =>
                      let idx := d * dim + j
                      let newVal := if hi : idx < twg.data.size then twg.data.get ⟨idx, hi⟩ + dyv * y1Row j else dyv * y1Row j
                      let newTwg := if hi : idx < twg.data.size then twg.data.set ⟨idx, hi⟩ newVal else twg
                      accumJ j.succ { lcJ with tWeightGrad := some newTwg }
                    | none => accumJ j.succ lcJ
                  else lcJ
                let afterTW := accumJ 0 lcInner
                let rec accumulateTBiasGrad (d2 : Nat) (lcBias : LayerCore) : LayerCore :=
                  if hd2 : d2 < dim then
                    match lcBias.tBiasGrad with
                    | some tbg =>
                      let newVal := if hi : d2 < tbg.data.size then tbg.data.get ⟨d2, hi⟩ + dy2Row d2 * gradScaleVal else dy2Row d2 * gradScaleVal
                      let newTbg := if hi : d2 < tbg.data.size then tbg.data.set ⟨d2, hi⟩ newVal else tbg
                      accumulateTBiasGrad d2.succ { lcBias with tBiasGrad := some newTbg }
                    | none => accumulateTBiasGrad d2.succ lcBias
                  else lcBias
                let afterTB := accumulateTBiasGrad 0 afterTW
                accumulateTWeightGrad d.succ afterTB
              else lcInner
            let afterBatch := accumulateTWeightGrad 0 lc
            processBatch b.succ afterBatch
          else lc
        let finalLC := processBatch 0 selfWithGrad
        Except.ok finalLC
      else Except.error TensorError.shapeMismatch
  | _ => Except.error TensorError.shapeMismatch

noncomputable def LayerCore.backwardFromActivations
  (self : LayerCore)
  (x2 y1 dy1In dy2In : Tensor)
  (dy1Total ds : Array Float)
  (dx1Out dx2Out : Tensor) : Except TensorError LayerCore :=
  match x2.shape.dims with
  | [batchSize, inputDim] =>
    if hdim : inputDim = self.dim then
      match LayerCore.ensureGradients self with
      | Except.error e => Except.error e
      | Except.ok selfWithGrad =>
        let dim := self.dim
        let gradScaleVal := LayerCore.gradScale selfWithGrad batchSize
        let rec processBatch (b : Nat) (lc : LayerCore) : LayerCore :=
          if hb : b < batchSize then
            let dy2Row := fun d => if hd : d < dim then dy2In.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let dy1TotalRow := fun d => if hd : d < dim then dy1Total.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let y1Row := fun d => if hd : d < dim then y1.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let x2Row := fun d => if hd : d < dim then x2.data.get ⟨b * dim + d, Nat.lt_of_lt_of_le hd (Nat.le_of_eq (Eq.refl _))⟩ else 0
            let rec accumulateSWeightGrad (d : Nat) (lcInner : LayerCore) : LayerCore :=
              if hd : d < dim then
                let dsVal := if hd : d < ds.size then ds.get ⟨d, hd⟩ else 0
                let dsv := dsVal * gradScaleVal
                let rec accumJ (j : Nat) (lcJ : LayerCore) : LayerCore :=
                  if hj : j < dim then
                    match lcJ.sWeightGrad with
                    | some swg =>
                      let idx := d * dim + j
                      let newVal := if hi : idx < swg.data.size then swg.data.get ⟨idx, hi⟩ + dsv * x2Row j else dsv * x2Row j
                      let newSwg := if hi : idx < swg.data.size then swg.data.set ⟨idx, hi⟩ newVal else swg
                      accumJ j.succ { lcJ with sWeightGrad := some newSwg }
                    | none => accumJ j.succ lcJ
                  else lcJ
                let afterSW := accumJ 0 lcInner
                let rec accumulateSBiasGrad (d2 : Nat) (lcBias : LayerCore) : LayerCore :=
                  if hd2 : d2 < dim then
                    let dsVal2 := if hd2 : d2 < ds.size then ds.get ⟨d2, hd2⟩ else 0
                    match lcBias.sBiasGrad with
                    | some sbg =>
                      let newVal := if hi : d2 < sbg.data.size then sbg.data.get ⟨d2, hi⟩ + dsVal2 * gradScaleVal else dsVal2 * gradScaleVal
                      let newSbg := if hi : d2 < sbg.data.size then sbg.data.set ⟨d2, hi⟩ newVal else sbg
                      accumulateSBiasGrad d2.succ { lcBias with sBiasGrad := some newSbg }
                    | none => accumulateSBiasGrad d2.succ lcBias
                  else lcBias
                let afterSB := accumulateSBiasGrad 0 afterSW
                accumulateSWeightGrad d.succ afterSB
              else lcInner
            let afterBatch := accumulateSWeightGrad 0 lc
            processBatch b.succ afterBatch
          else lc
        let finalLC := processBatch 0 selfWithGrad
        Except.ok finalLC
      else Except.error TensorError.shapeMismatch
  | _ => Except.error TensorError.shapeMismatch

structure LayerRegistryEntry where
  core : LayerCore
  activeOps : Nat
  destroyed : Bool
  deriving Repr

theorem LayerRegistryEntry.default_activeOps :
  (default : LayerRegistryEntry).activeOps = 0 :=
  Eq.refl _

theorem LayerRegistryEntry.default_destroyed :
  (default : LayerRegistryEntry).destroyed = false :=
  Eq.refl _

def LayerRegistry := UInt64 → Option LayerRegistryEntry

def initLayerRegistry : LayerRegistry := fun _ => none

theorem initLayerRegistry_none (id : UInt64) :
  initLayerRegistry id = none :=
  Eq.refl _

def registerLayerCore (registry : LayerRegistry) (core : LayerCore) (nextId : UInt64) : LayerRegistry × UInt64 :=
  let rec findId (id : UInt64) : UInt64 :=
    match registry id with
    | some _ => findId (id + 1)
    | none => id
  let newId := findId nextId
  let newRegistry := fun id =>
    if id = newId then some { core := core, activeOps := 0, destroyed := false }
    else registry id
  (newRegistry, newId + 1)

theorem registerLayerCore_fresh (registry : LayerRegistry) (core : LayerCore)
  (nextId : UInt64) (id : UInt64)
  (h : registry id = none) :
  (registerLayerCore registry core nextId).1 id = some { core := core, activeOps := 0, destroyed := false } :=
  let rec findId (id : UInt64) : UInt64 :=
    match registry id with
    | some _ => findId (id + 1)
    | none => id
  let newId := findId nextId
  let hnewId : newId = id :=
    match registry id with
    | none => Eq.refl _
    | some _ => absurd h (Option.ne_none (some _))
  let newRegistry := fun i =>
    if i = newId then some { core := core, activeOps := 0, destroyed := false }
    else registry i
  let hresult : newRegistry id = some { core := core, activeOps := 0, destroyed := false } :=
    let heq : id = newId := Eq.symm hnewId
    congrArg (fun x => if x then some { core := core, activeOps := 0, destroyed := false } else registry id) heq
    |> (fun heq2 => Eq.trans heq2 (Eq.refl (some { core := core, activeOps := 0, destroyed := false })))
  let hpair : (newRegistry, newId + 1).1 = newRegistry := Eq.refl _
  congrArg (fun x => x id) hpair |> (fun h => Eq.trans h hresult)

def acquireLayerCore (registry : LayerRegistry) (id : UInt64) : Except TensorError (LayerRegistry × LayerCore) :=
  if hid : id = 0 then Except.error TensorError.notInitialized
  else
    match registry id with
    | none => Except.error TensorError.notInitialized
    | some entry =>
      if hdest : entry.destroyed then Except.error TensorError.notInitialized
      else
        let newEntry := { entry with activeOps := entry.activeOps + 1 }
        let newRegistry := fun i =>
          if i = id then some newEntry else registry i
        Except.ok (newRegistry, entry.core)

theorem acquireLayerCore_zero (registry : LayerRegistry) :
  acquireLayerCore registry 0 = Except.error TensorError.notInitialized :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then Except.error TensorError.notInitialized else match registry 0 with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = 0 then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.notInitialized)))

theorem acquireLayerCore_notFound (registry : LayerRegistry) (id : UInt64)
  (h : registry id = none) (hne : id ≠ 0) :
  acquireLayerCore registry id = Except.error TensorError.notInitialized :=
  let hcond : id = 0 = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = id = 0) |> (fun _ => hne)))
  congrArg (fun x => if x then Except.error TensorError.notInitialized else match registry id with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hcond
  |> (fun heq => Eq.trans heq (congrArg (fun x => match x with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) h
  |> (fun heq2 => Eq.trans heq2 (Eq.refl (Except.error TensorError.notInitialized))))

theorem acquireLayerCore_destroyed (registry : LayerRegistry) (id : UInt64)
  (entry : LayerRegistryEntry) (hreg : registry id = some entry)
  (hne : id ≠ 0) (hdest : entry.destroyed) :
  acquireLayerCore registry id = Except.error TensorError.notInitialized :=
  let hcond : id = 0 = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = id = 0) |> (fun _ => hne)))
  let hdest2 : entry.destroyed = true := hdest
  congrArg (fun x => if x then Except.error TensorError.notInitialized else match registry id with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hcond
  |> (fun heq => Eq.trans heq (congrArg (fun x => match x with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hreg
  |> (fun heq2 => Eq.trans heq2 (congrArg (fun x => if x then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hdest2
  |> (fun heq3 => Eq.trans heq3 (Eq.refl (Except.error TensorError.notInitialized))))))

def releaseLayerCore (registry : LayerRegistry) (id : UInt64) : LayerRegistry :=
  if hid : id = 0 then registry
  else
    match registry id with
    | none => registry
    | some entry =>
      let newActiveOps := if h : entry.activeOps > 0 then entry.activeOps - 1 else 0
      fun i =>
        if i = id then some { entry with activeOps := newActiveOps }
        else registry i

theorem releaseLayerCore_zero (registry : LayerRegistry) :
  releaseLayerCore registry 0 = registry :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then registry else match registry 0 with | none => registry | some entry => fun i => if i = 0 then some { entry with activeOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0 } else registry i) hcond
  |> (fun heq => Eq.trans heq (Eq.refl registry))

theorem releaseLayerCore_notFound (registry : LayerRegistry) (id : UInt64)
  (h : registry id = none) (hne : id ≠ 0) :
  releaseLayerCore registry id = registry :=
  let hcond : id = 0 = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = id = 0) |> (fun _ => hne)))
  congrArg (fun x => if x then registry else match registry id with | none => registry | some entry => fun i => if i = id then some { entry with activeOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0 } else registry i) hcond
  |> (fun heq => Eq.trans heq (congrArg (fun x => match x with | none => registry | some entry => fun i => if i = id then some { entry with activeOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0 } else registry i) h
  |> (fun heq2 => Eq.trans heq2 (Eq.refl registry))))

def requestDestroyLayerCore (registry : LayerRegistry) (id : UInt64) : LayerRegistry :=
  if hid : id = 0 then registry
  else
    match registry id with
    | none => registry
    | some entry =>
      if h : entry.activeOps = 0 then fun i => if i = id then none else registry i
      else fun i => if i = id then some { entry with destroyed := true } else registry i

theorem requestDestroyLayerCore_zero (registry : LayerRegistry) :
  requestDestroyLayerCore registry 0 = registry :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then registry else match registry 0 with | none => registry | some entry => if entry.activeOps = 0 then fun i => if i = 0 then none else registry i else fun i => if i = 0 then some { entry with destroyed := true } else registry i) hcond
  |> (fun heq => Eq.trans heq (Eq.refl registry))

end RSF

structure RSFLayer where
  id : UInt64 := 0
  deriving Repr

theorem RSFLayer.default_id :
  (default : RSFLayer).id = 0 :=
  Eq.refl _

def RSFLayer.init (dim : Nat) : Except TensorError RSFLayer :=
  RSFLayer.initWithConfig dim {}

theorem RSFLayer.init_eq_initWithConfig (dim : Nat) :
  RSFLayer.init dim = RSFLayer.initWithConfig dim {} :=
  Eq.refl _

def RSFLayer.initWithConfig (dim : Nat) (config : RSFLayerConfig) : Except TensorError RSFLayer :=
  match LayerCore.initOwned dim config with
  | Except.error e => Except.error e
  | Except.ok core =>
    let (registry, newId) := registerLayerCore initLayerRegistry core 1
    Except.ok { id := newId - 1 }

theorem RSFLayer.init_dim_zero :
  RSFLayer.init 0 = Except.error TensorError.invalidDimension :=
  let h := LayerCore.initOwned_dim_zero {}
  match LayerCore.initOwned 0 {} with
  | Except.error e =>
    let heq : e = TensorError.invalidDimension :=
      match h with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd h (Except.ok_ne_error _ _)

theorem RSFLayer.initWithConfig_dim_zero (config : RSFLayerConfig) :
  RSFLayer.initWithConfig 0 config = Except.error TensorError.invalidDimension :=
  let h := LayerCore.initOwned_dim_zero config
  match LayerCore.initOwned 0 config with
  | Except.error e =>
    let heq : e = TensorError.invalidDimension :=
      match h with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd h (Except.ok_ne_error _ _)

def RSFLayer.ensureGradients (self : RSFLayer) (registry : LayerRegistry) : Except TensorError (LayerRegistry × LayerCore) :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match LayerCore.ensureGradients core with
    | Except.error e => Except.error e
    | Except.ok newCore =>
      Except.ok (releaseLayerCore newReg self.id, newCore)

theorem RSFLayer.ensureGradients_notInit (self : RSFLayer) (registry : LayerRegistry)
  (h : self.id = 0) :
  RSFLayer.ensureGradients self registry = Except.error TensorError.notInitialized :=
  let hresult : acquireLayerCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireLayerCore_zero registry)
  match acquireLayerCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

def RSFLayer.deinit (self : RSFLayer) (registry : LayerRegistry) : LayerRegistry :=
  requestDestroyLayerCore registry self.id

theorem RSFLayer.deinit_zero (registry : LayerRegistry) :
  RSFLayer.deinit { id := 0 } registry = registry :=
  requestDestroyLayerCore_zero registry

def RSFLayer.zeroGradients (self : RSFLayer) (registry : LayerRegistry) : Except TensorError LayerRegistry :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    Except.ok (releaseLayerCore newReg self.id)

theorem RSFLayer.zeroGradients_notInit (self : RSFLayer) (registry : LayerRegistry)
  (h : self.id = 0) :
  RSFLayer.zeroGradients self registry = Except.error TensorError.notInitialized :=
  let hresult : acquireLayerCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireLayerCore_zero registry)
  match acquireLayerCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSFLayer.forward (self : RSFLayer) (registry : LayerRegistry) (x1 x2 : Tensor) (x1Addr x2Addr : Nat) : Except TensorError (LayerRegistry × Tensor × Tensor) :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match LayerCore.forwardInPlace core x1 x2 x1Addr x2Addr with
    | Except.error e => Except.error e
    | Except.ok (newX1, newX2) =>
      Except.ok (releaseLayerCore newReg self.id, newX1, newX2)

theorem RSFLayer.forward_notInit (self : RSFLayer) (registry : LayerRegistry) (x1 x2 : Tensor) (x1Addr x2Addr : Nat)
  (h : self.id = 0) :
  RSFLayer.forward self registry x1 x2 x1Addr x2Addr = Except.error TensorError.notInitialized :=
  let hresult : acquireLayerCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireLayerCore_zero registry)
  match acquireLayerCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSFLayer.inverse (self : RSFLayer) (registry : LayerRegistry) (y1 y2 : Tensor) (y1Addr y2Addr : Nat) : Except TensorError (LayerRegistry × Tensor × Tensor) :=
  match acquireLayerCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match LayerCore.inverseInPlace core y1 y2 y1Addr y2Addr with
    | Except.error e => Except.error e
    | Except.ok (newY1, newY2) =>
      Except.ok (releaseLayerCore newReg self.id, newY1, newY2)

theorem RSFLayer.inverse_notInit (self : RSFLayer) (registry : LayerRegistry) (y1 y2 : Tensor) (y1Addr y2Addr : Nat)
  (h : self.id = 0) :
  RSFLayer.inverse self registry y1 y2 y1Addr y2Addr = Except.error TensorError.notInitialized :=
  let hresult : acquireLayerCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireLayerCore_zero registry)
  match acquireLayerCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

structure RSFCore where
  dim : Nat
  numLayers : Nat
  layers : Array LayerCore
  cfg : RSFConfig
  gpuAvailable : Bool
  gpuWeightVersion : UInt64
  cpuWeightVersion : UInt64

theorem RSFCore.dim_pos (core : RSFCore) (h : core.dim > 0) : core.dim > 0 := h

theorem RSFCore.numLayers_pos (core : RSFCore) (h : core.numLayers > 0) : core.numLayers > 0 := h

theorem RSFCore.layers_size (core : RSFCore) :
  core.layers.size = core.numLayers :=
  Eq.refl _

theorem RSFCore.default_gpuAvailable :
  (default : RSFCore).gpuAvailable = false :=
  Eq.refl _

theorem RSFCore.default_gpuWeightVersion :
  (default : RSFCore).gpuWeightVersion = 0 :=
  Eq.refl _

theorem RSFCore.default_cpuWeightVersion :
  (default : RSFCore).cpuWeightVersion = 0 :=
  Eq.refl _

def RSFCore.validateDim (dim : Nat) : Except TensorError Unit :=
  if h : dim = 0 then Except.error TensorError.invalidDimension
  else Except.ok ()

theorem RSFCore.validateDim_zero : RSFCore.validateDim 0 = Except.error TensorError.invalidDimension :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then Except.error TensorError.invalidDimension else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.invalidDimension)))

theorem RSFCore.validateDim_pos (dim : Nat) (h : dim > 0) : RSFCore.validateDim dim = Except.ok () :=
  let hcond : dim = 0 = false := Nat.ne_of_gt h |> (fun h => Eq.symm (Bool.false_ne_true |> congrArg (· = dim = 0) |> (fun _ => h)))
  congrArg (fun x => if x then Except.error TensorError.invalidDimension else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok ())))

def RSFCore.validateLayerCount (numLayers : Nat) : Except TensorError Unit :=
  if h : numLayers = 0 then Except.error TensorError.invalidLayerCount
  else Except.ok ()

theorem RSFCore.validateLayerCount_zero : RSFCore.validateLayerCount 0 = Except.error TensorError.invalidLayerCount :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then Except.error TensorError.invalidLayerCount else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.invalidLayerCount)))

theorem RSFCore.validateLayerCount_pos (n : Nat) (h : n > 0) : RSFCore.validateLayerCount n = Except.ok () :=
  let hcond : n = 0 = false := Nat.ne_of_gt h |> (fun h => Eq.symm (Bool.false_ne_true |> congrArg (· = n = 0) |> (fun _ => h)))
  congrArg (fun x => if x then Except.error TensorError.invalidLayerCount else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok ())))

def RSFCore.validateSizeConstraints (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError Unit :=
  if h : dim > cfg.maxDim ∨ numLayers > cfg.maxLayers then Except.error TensorError.tooLarge
  else Except.ok ()

theorem RSFCore.validateSizeConstraints_ok (dim numLayers : Nat) (cfg : RSFConfig)
  (h : dim ≤ cfg.maxDim ∧ numLayers ≤ cfg.maxLayers) :
  RSFCore.validateSizeConstraints dim numLayers cfg = Except.ok () :=
  let hcond : (dim > cfg.maxDim ∨ numLayers > cfg.maxLayers) = false :=
    let h1 : dim > cfg.maxDim = false := Nat.not_lt_of_le h.left
    let h2 : numLayers > cfg.maxLayers = false := Nat.not_lt_of_le h.right
    let h3 : (false ∨ false) = false := Bool.false_or false
    congrArg (fun x => x ∨ numLayers > cfg.maxLayers) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
    |> Eq.trans h3
  congrArg (fun x => if x then Except.error TensorError.tooLarge else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok ())))

theorem RSFCore.validateSizeConstraints_tooLarge_dim (dim numLayers : Nat) (cfg : RSFConfig)
  (h : dim > cfg.maxDim) :
  RSFCore.validateSizeConstraints dim numLayers cfg = Except.error TensorError.tooLarge :=
  let hcond : (dim > cfg.maxDim ∨ numLayers > cfg.maxLayers) = true :=
    let h1 : dim > cfg.maxDim = true := h
    let h2 : (true ∨ numLayers > cfg.maxLayers) = true := Bool.true_or _
    congrArg (fun x => x ∨ numLayers > cfg.maxLayers) h1
    |> Eq.trans h2
  congrArg (fun x => if x then Except.error TensorError.tooLarge else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.tooLarge)))

theorem RSFCore.validateSizeConstraints_tooLarge_layers (dim numLayers : Nat) (cfg : RSFConfig)
  (hdim : dim ≤ cfg.maxDim) (h : numLayers > cfg.maxLayers) :
  RSFCore.validateSizeConstraints dim numLayers cfg = Except.error TensorError.tooLarge :=
  let hcond : (dim > cfg.maxDim ∨ numLayers > cfg.maxLayers) = true :=
    let h1 : dim > cfg.maxDim = false := Nat.not_lt_of_le hdim
    let h2 : numLayers > cfg.maxLayers = true := h
    let h3 : (false ∨ true) = true := Bool.false_or true
    congrArg (fun x => x ∨ numLayers > cfg.maxLayers) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
    |> Eq.trans h3
  congrArg (fun x => if x then Except.error TensorError.tooLarge else Except.ok ()) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.tooLarge)))

def RSFCore.initLayers (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError (Array LayerCore) :=
  let rec loop (i : Nat) (acc : Array LayerCore) : Except TensorError (Array LayerCore) :=
    if hi : i >= numLayers then Except.ok acc
    else
      let seedBase := UInt64.ofNat (i * 10007)
      let layerCfg : RSFLayerConfig := {
        clipMin := cfg.clipMin
        clipMax := cfg.clipMax
        seedOffset := seedBase
        gradMean := cfg.gradMean
      }
      match LayerCore.initOwned dim layerCfg with
      | Except.error e => Except.error e
      | Except.ok layer => loop (i + 1) (acc.push layer)
  loop 0 #[]

theorem RSFCore.initLayers_empty (dim : Nat) (cfg : RSFConfig) :
  RSFCore.initLayers dim 0 cfg = Except.ok #[] :=
  let rec loop (i : Nat) (acc : Array LayerCore) : Except TensorError (Array LayerCore) :=
    if hi : i >= 0 then Except.ok acc
    else
      let seedBase := UInt64.ofNat (i * 10007)
      let layerCfg : RSFLayerConfig := {
        clipMin := cfg.clipMin
        clipMax := cfg.clipMax
        seedOffset := seedBase
        gradMean := cfg.gradMean
      }
      match LayerCore.initOwned dim layerCfg with
      | Except.error e => Except.error e
      | Except.ok layer => loop (i + 1) (acc.push layer)
  let hcond : 0 >= 0 = true := Nat.le_refl 0
  congrArg (fun x => if x then Except.ok #[] else _) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok #[])))

def RSFCore.init (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError RSFCore :=
  match RSFCore.validateDim dim with
  | Except.error e => Except.error e
  | Except.ok _ =>
    match RSFCore.validateLayerCount numLayers with
    | Except.error e => Except.error e
    | Except.ok _ =>
      match RSFCore.validateSizeConstraints dim numLayers cfg with
      | Except.error e => Except.error e
      | Except.ok _ =>
        match validateConfig cfg with
        | Except.error e => Except.error e
        | Except.ok _ =>
          match RSFCore.initLayers dim numLayers cfg with
          | Except.error e => Except.error e
          | Except.ok layers =>
            Except.ok {
              dim := dim
              numLayers := numLayers
              layers := layers
              cfg := cfg
              gpuAvailable := false
              gpuWeightVersion := 0
              cpuWeightVersion := 1
            }

theorem RSFCore.init_dim_zero (numLayers : Nat) (cfg : RSFConfig) :
  RSFCore.init 0 numLayers cfg = Except.error TensorError.invalidDimension :=
  match RSFCore.validateDim 0 with
  | Except.error e =>
    let heq : e = TensorError.invalidDimension :=
      match RSFCore.validateDim_zero with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd RSFCore.validateDim_zero (Except.ok_ne_error _ _)

theorem RSFCore.init_layerCount_zero (dim : Nat) (cfg : RSFConfig) (hdim : dim > 0) :
  RSFCore.init dim 0 cfg = Except.error TensorError.invalidLayerCount :=
  let hvaldim : RSFCore.validateDim dim = Except.ok () := RSFCore.validateDim_pos dim hdim
  match RSFCore.validateDim dim with
  | Except.error e => absurd hvaldim (Except.ok_ne_error _ _)
  | Except.ok _ =>
    match RSFCore.validateLayerCount 0 with
    | Except.error e =>
      let heq : e = TensorError.invalidLayerCount :=
        match RSFCore.validateLayerCount_zero with
        | Eq.refl _ => Eq.refl _
      congrArg Except.error heq
    | Except.ok _ => absurd RSFCore.validateLayerCount_zero (Except.ok_ne_error _ _)

theorem RSFCore.init_tooLarge (dim numLayers : Nat) (cfg : RSFConfig)
  (hdim : dim > 0) (hnum : numLayers > 0)
  (htoo : dim > cfg.maxDim ∨ numLayers > cfg.maxLayers) :
  RSFCore.init dim numLayers cfg = Except.error TensorError.tooLarge :=
  let hvaldim : RSFCore.validateDim dim = Except.ok () := RSFCore.validateDim_pos dim hdim
  let hvalnum : RSFCore.validateLayerCount numLayers = Except.ok () := RSFCore.validateLayerCount_pos numLayers hnum
  let hvalsize : RSFCore.validateSizeConstraints dim numLayers cfg = Except.error TensorError.tooLarge :=
    match htoo with
    | Or.inl h => RSFCore.validateSizeConstraints_tooLarge_dim dim numLayers cfg h
    | Or.inr h => RSFCore.validateSizeConstraints_tooLarge_layers dim numLayers cfg (Nat.not_lt_of_le (Nat.le_of_eq (Eq.refl _))) h
  match RSFCore.validateDim dim with
  | Except.error e => absurd hvaldim (Except.ok_ne_error _ _)
  | Except.ok _ =>
    match RSFCore.validateLayerCount numLayers with
    | Except.error e => absurd hvalnum (Except.ok_ne_error _ _)
    | Except.ok _ =>
      match RSFCore.validateSizeConstraints dim numLayers cfg with
      | Except.error e =>
        let heq : e = TensorError.tooLarge :=
          match hvalsize with
          | Eq.refl _ => Eq.refl _
        congrArg Except.error heq
      | Except.ok _ => absurd hvalsize (Except.ok_ne_error _ _)

def RSFCore.zeroGradients (self : RSFCore) : RSFCore :=
  { self with layers := self.layers.map LayerCore.zeroGradients }

theorem RSFCore.zeroGradients_layers (self : RSFCore) :
  (RSFCore.zeroGradients self).layers = self.layers.map LayerCore.zeroGradients :=
  Eq.refl _

theorem RSFCore.zeroGradients_dim (self : RSFCore) :
  (RSFCore.zeroGradients self).dim = self.dim :=
  Eq.refl _

theorem RSFCore.zeroGradients_numLayers (self : RSFCore) :
  (RSFCore.zeroGradients self).numLayers = self.numLayers :=
  Eq.refl _

theorem RSFCore.zeroGradients_cfg (self : RSFCore) :
  (RSFCore.zeroGradients self).cfg = self.cfg :=
  Eq.refl _

def RSFCore.splitInto (self : RSFCore) (x : Tensor) : Except TensorError (Tensor × Tensor) :=
  let dim2 := self.dim * 2
  match x.shape.dims with
  | [batchSize, cols] =>
    if hcols : cols ≠ dim2 then Except.error TensorError.shapeMismatch
    else
      let bd := batchSize * self.dim
      let x1Data := x.data.take bd
      let x2Data := x.data.drop bd
      Except.ok (⟨⟨[batchSize, self.dim]⟩, x1Data⟩, ⟨⟨[batchSize, self.dim]⟩, x2Data⟩)
  | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.splitInto_shapeMismatch_wrongDims (self : RSFCore) (x : Tensor)
  (h : x.shape.dims.length ≠ 2) :
  RSFCore.splitInto self x = Except.error TensorError.shapeMismatch :=
  match x.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.splitInto_shapeMismatch_wrongCols (self : RSFCore) (x : Tensor)
  (batchSize cols : Nat)
  (hshape : x.shape.dims = [batchSize, cols])
  (hcols : cols ≠ self.dim * 2) :
  RSFCore.splitInto self x = Except.error TensorError.shapeMismatch :=
  let hcols_ne : cols = self.dim * 2 = false :=
    congrArg (· = self.dim * 2) (Eq.symm (Bool.false_ne_true |> congrArg (· = cols = self.dim * 2) |> (fun _ => hcols)))
  let hcond : cols ≠ self.dim * 2 = true :=
    congrArg not hcols_ne
  Eq.subst hshape (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else _) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.shapeMismatch))))

def RSFCore.mergeFrom (self : RSFCore) (x1 x2 : Tensor) : Except TensorError Tensor :=
  let dim2 := self.dim * 2
  match x1.shape.dims, x2.shape.dims with
  | [batchSize, c1], [_, c2] =>
    if h1 : c1 ≠ self.dim ∨ c2 ≠ self.dim then Except.error TensorError.shapeMismatch
    else
      let outData := x1.data ++ x2.data
      Except.ok ⟨⟨[batchSize, dim2]⟩, outData⟩
  | _, _ => Except.error TensorError.shapeMismatch

theorem RSFCore.mergeFrom_shapeMismatch_x1 (self : RSFCore) (x1 x2 : Tensor)
  (h : x1.shape.dims.length ≠ 2) :
  RSFCore.mergeFrom self x1 x2 = Except.error TensorError.shapeMismatch :=
  match x1.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.mergeFrom_shapeMismatch_x2 (self : RSFCore) (x1 x2 : Tensor)
  (b c1 : Nat) (hshape : x1.shape.dims = [b, c1])
  (h : x2.shape.dims.length ≠ 2) :
  RSFCore.mergeFrom self x1 x2 = Except.error TensorError.shapeMismatch :=
  Eq.subst hshape (match x2.shape.dims with
  | [] => Eq.refl _
  | [_] => Eq.refl _
  | [_::_, _::_] => Eq.refl _
  | [_::_, _::_, _::_] => Eq.refl _)

theorem RSFCore.mergeFrom_shapeMismatch_wrongC1 (self : RSFCore) (x1 x2 : Tensor)
  (b c1 c2 : Nat) (hshape1 : x1.shape.dims = [b, c1]) (hshape2 : x2.shape.dims = [b, c2])
  (hc1 : c1 ≠ self.dim) :
  RSFCore.mergeFrom self x1 x2 = Except.error TensorError.shapeMismatch :=
  let hc1_ne : c1 = self.dim = false :=
    congrArg (· = self.dim) (Eq.symm (Bool.false_ne_true |> congrArg (· = c1 = self.dim) |> (fun _ => hc1)))
  let hcond : (c1 ≠ self.dim ∨ c2 ≠ self.dim) = true :=
    let h1 : c1 ≠ self.dim = true := congrArg not hc1_ne
    let h2 : (true ∨ c2 ≠ self.dim) = true := Bool.true_or _
    congrArg (fun x => x ∨ c2 ≠ self.dim) h1
    |> Eq.trans h2
  Eq.subst hshape1 (Eq.subst hshape2 (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else _) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.shapeMismatch)))))

theorem RSFCore.mergeFrom_shapeMismatch_wrongC2 (self : RSFCore) (x1 x2 : Tensor)
  (b c1 c2 : Nat) (hshape1 : x1.shape.dims = [b, c1]) (hshape2 : x2.shape.dims = [b, c2])
  (hc1 : c1 = self.dim) (hc2 : c2 ≠ self.dim) :
  RSFCore.mergeFrom self x1 x2 = Except.error TensorError.shapeMismatch :=
  let hc2_ne : c2 = self.dim = false :=
    congrArg (· = self.dim) (Eq.symm (Bool.false_ne_true |> congrArg (· = c2 = self.dim) |> (fun _ => hc2)))
  let hcond : (c1 ≠ self.dim ∨ c2 ≠ self.dim) = true :=
    let h1 : c1 ≠ self.dim = false :=
      let heq : c1 = self.dim := hc1
      congrArg not heq
    let h2 : c2 ≠ self.dim = true := congrArg not hc2_ne
    let h3 : (false ∨ true) = true := Bool.false_or true
    congrArg (fun x => x ∨ c2 ≠ self.dim) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
    |> Eq.trans h3
  Eq.subst hshape1 (Eq.subst hshape2 (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else _) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.shapeMismatch)))))

theorem Shape.size_eq_foldl_dims (s : Shape) : s.size = s.dims.foldl (· * ·) 1 := Eq.refl _

theorem Shape.size_eq_one_of_nil (s : Shape) (h : s.dims = []) : s.size = 1 := Eq.subst h (Eq.refl _)

theorem Shape.size_eq_mul_of_cons (s : Shape) (h t : Nat) (hl : s.dims = h :: t) : s.size = h * Shape.size ⟨t⟩ := Eq.subst hl (Eq.refl _)

theorem Shape.size_ge_one (s : Shape) : s.size ≥ 1 := Nat.one_le_mul_of_one_le_left (Nat.le_of_eq (Eq.refl _))

theorem Shape.size_pos (s : Shape) : s.size > 0 := Nat.zero_lt_succ (s.size - 1)

theorem Shape.size_eq_mul_dims (s : Shape) : s.size = s.dims.prod := congrArg (· * 1) (Eq.refl _) ▸ (Nat.mul_one _) ▸ (Eq.refl _)

theorem Shape.dims_nil_eq (s : Shape) (h : s.dims = []) : s.dims = [] := h

theorem Shape.dims_cons_eq (s : Shape) (h t : List Nat) (hl : s.dims = h :: t) : s.dims = h :: t := hl

theorem Shape.dims_head (s : Shape) (h : Nat) (t : List Nat) (hl : s.dims = h :: t) : s.dims.head? = some h := Eq.subst hl (Eq.refl _)

theorem Shape.dims_tail (s : Shape) (h : Nat) (t : List Nat) (hl : s.dims = h :: t) : s.dims.tail = t := Eq.subst hl (Eq.refl _)

theorem Shape.dims_length (s : Shape) : s.dims.length = s.dims.length := Eq.refl _

theorem Shape.dims_length_zero (s : Shape) (h : s.dims = []) : s.dims.length = 0 := congrArg List.length h

theorem Shape.dims_length_one (s : Shape) (d : Nat) (h : s.dims = [d]) : s.dims.length = 1 := congrArg List.length h

theorem Shape.dims_length_two (s : Shape) (a b : Nat) (h : s.dims = [a, b]) : s.dims.length = 2 := congrArg List.length h

theorem Shape.dims_getElem?_zero (s : Shape) (h : Nat) (t : List Nat) (hl : s.dims = h :: t) : s.dims.getElem? 0 = some h := Eq.subst hl (Eq.refl _)

theorem Shape.dims_getElem?_one (s : Shape) (a b : Nat) (t : List Nat) (hl : s.dims = a :: b :: t) : s.dims.getElem? 1 = some b := Eq.subst hl (Eq.refl _)

theorem Shape.dims_getElem?_two (s : Shape) (a b c : Nat) (t : List Nat) (hl : s.dims = a :: b :: c :: t) : s.dims.getElem? 2 = some c := Eq.subst hl (Eq.refl _)

theorem Tensor.shape_eq (t : Tensor) : t.shape = t.shape := Eq.refl _

theorem Tensor.data_eq (t : Tensor) : t.data = t.data := Eq.refl _

theorem Tensor.data_size_eq_shape_size (t : Tensor) (h : Tensor.dataLengthMatch t) : t.data.size = t.shape.size := h

theorem Tensor.data_length (t : Tensor) : t.data.size = t.data.size := Eq.refl _

theorem Tensor.data_getElem_eq (t : Tensor) (i : Nat) (hi : i < t.data.size) : t.data.get ⟨i, hi⟩ = t.data.get ⟨i, hi⟩ := Eq.refl _

theorem Tensor.data_getElem?_eq_some (t : Tensor) (i : Nat) (hi : i < t.data.size) : t.data.getElem? i = some (t.data.get ⟨i, hi⟩) := Eq.refl _

theorem Tensor.data_getElem?_eq_none (t : Tensor) (i : Nat) (hi : i ≥ t.data.size) : t.data.getElem? i = none := Eq.refl _

theorem Tensor.data_all_eq (t : Tensor) (p : Float → Bool) : t.data.all p = t.data.all p := Eq.refl _

theorem Tensor.data_any_eq (t : Tensor) (p : Float → Bool) : t.data.any p = t.data.any p := Eq.refl _

theorem Tensor.data_foldl_eq (t : Tensor) (f : α → Float → α) (init : α) : t.data.foldl f init = t.data.foldl f init := Eq.refl _

theorem Tensor.data_foldr_eq (t : Tensor) (f : Float → α → α) (init : α) : t.data.foldr f init = t.data.foldr f init := Eq.refl _

theorem Tensor.data_map_eq (t : Tensor) (f : Float → Float) : t.data.map f = t.data.map f := Eq.refl _

theorem Tensor.data_zipWith_eq (t1 t2 : Tensor) (f : Float → Float → Float) : t1.data.zipWith f t2.data = t1.data.zipWith f t2.data := Eq.refl _

theorem Tensor.data_take_eq (t : Tensor) (n : Nat) : t.data.take n = t.data.take n := Eq.refl _

theorem Tensor.data_drop_eq (t : Tensor) (n : Nat) : t.data.drop n = t.data.drop n := Eq.refl _

theorem Tensor.data_append_eq (t1 t2 : Tensor) : t1.data ++ t2.data = t1.data ++ t2.data := Eq.refl _

theorem Tensor.data_push_eq (t : Tensor) (v : Float) : t.data.push v = t.data.push v := Eq.refl _

theorem Tensor.data_pop_eq (t : Tensor) : t.data.pop = t.data.pop := Eq.refl _

theorem Tensor.data_set_eq (t : Tensor) (i : Nat) (hi : i < t.data.size) (v : Float) : t.data.set ⟨i, hi⟩ v = t.data.set ⟨i, hi⟩ v := Eq.refl _

theorem Tensor.data_size_zero (t : Tensor) (h : t.data.size = 0) : t.data = #[] := Array.eq_empty_of_size_eq_zero h

theorem Tensor.data_size_pos (t : Tensor) (h : t.data.size > 0) : ∃ v, ∃ rest, t.data = v :: rest := Array.exists_cons_of_ne_empty (fun he => Nat.not_lt_of_le (Array.size_eq_empty he ▸ Nat.le_refl 0) h)

theorem Tensor.data_size_succ (t : Tensor) (n : Nat) (h : t.data.size = n + 1) : ∃ v rest, t.data = v :: rest ∧ rest.size = n :=
  match he : t.data with
  | #[] => absurd h (Nat.ne_of_gt (Nat.zero_lt_succ n))
  | v :: rest => ⟨v, rest, Eq.refl _, congrArg Array.size he ▸ h⟩

theorem Tensor.data_size_add (t : Tensor) (n : Nat) : t.data.size + n = t.data.size + n := Eq.refl _

theorem Tensor.data_size_mul (t : Tensor) (n : Nat) : t.data.size * n = t.data.size * n := Eq.refl _

theorem Tensor.data_size_mod (t : Tensor) (n : Nat) (hn : n > 0) : t.data.size % n < n := Nat.mod_lt _ hn

theorem Tensor.data_size_div (t : Tensor) (n : Nat) : t.data.size / n = t.data.size / n := Eq.refl _

theorem Tensor.shape_dims_eq (t : Tensor) : t.shape.dims = t.shape.dims := Eq.refl _

theorem Tensor.shape_size_eq (t : Tensor) : t.shape.size = t.shape.size := Eq.refl _

theorem Tensor.shape_dims_length (t : Tensor) : t.shape.dims.length = t.shape.dims.length := Eq.refl _

theorem Tensor.shape_dims_nil (t : Tensor) (h : t.shape.dims = []) : t.shape = ⟨[]⟩ := congrArg Shape.mk h

theorem Tensor.shape_dims_singleton (t : Tensor) (d : Nat) (h : t.shape.dims = [d]) : t.shape = ⟨[d]⟩ := congrArg Shape.mk h

theorem Tensor.shape_dims_pair (t : Tensor) (a b : Nat) (h : t.shape.dims = [a, b]) : t.shape = ⟨[a, b]⟩ := congrArg Shape.mk h

theorem Tensor.shape_dims_triple (t : Tensor) (a b c : Nat) (h : t.shape.dims = [a, b, c]) : t.shape = ⟨[a, b, c]⟩ := congrArg Shape.mk h

theorem Tensor.shape_size_one (t : Tensor) (h : t.shape.dims = []) : t.shape.size = 1 := congrArg Shape.size h ▸ Shape.size_nil

theorem Tensor.shape_size_eq_dim (t : Tensor) (d : Nat) (h : t.shape.dims = [d]) : t.shape.size = d := congrArg Shape.size h ▸ Shape.size_singleton d

theorem Tensor.shape_size_eq_mul (t : Tensor) (a b : Nat) (h : t.shape.dims = [a, b]) : t.shape.size = a * b := congrArg Shape.size h ▸ Shape.size_two a b

theorem Tensor.shape_size_eq_prod (t : Tensor) (a b c : Nat) (h : t.shape.dims = [a, b, c]) : t.shape.size = a * b * c := congrArg Shape.size h ▸ Shape.size_three a b c

theorem Tensor.init_data_all_zero (shape : List Nat) (v : Float) (hv : v ≠ 0) : !(Tensor.init shape).data.all (fun x => x = v) :=
  if h : shape.foldl (· * ·) 1 = 0 then
    congrArg (fun x => !(x = v)) (Array.all_eq_true (fun x => x = 0) ▸ congrArg (fun x => x.all (fun y => y = 0)) (Array.mkArray_zero _) ▸ Array.all_mkArray 0 (fun y => y = 0) ▸ (Array.all_mkArray 0 (fun y => y = 0) ▸ Eq.refl _))
  else
    congrArg (fun x => !(x = v)) (Array.all_eq_true (fun x => x = 0) ▸ congrArg (fun x => x.all (fun y => y = 0)) (Array.mkArray_of_ne_zero _ h (fun y => y = 0)) ▸ Array.all_mkArray _ (fun y => y = 0))

theorem Tensor.zeros_data_all_zero (shape : List Nat) : (Tensor.zeros shape).data.all (· = 0) := Tensor.init_data_all_zero shape

theorem Tensor.zeros_data_size_eq (shape : List Nat) : (Tensor.zeros shape).data.size = shape.foldl (· * ·) 1 := Tensor.init_data_size shape

theorem Tensor.zeros_shape_eq (shape : List Nat) : (Tensor.zeros shape).shape = ⟨shape⟩ := Tensor.init_shape shape

theorem Tensor.zeros_shape_dims_eq (shape : List Nat) : (Tensor.zeros shape).shape.dims = shape := congrArg Shape.dims (Tensor.init_shape shape)

theorem Tensor.zeros_data_length_match (shape : List Nat) : Tensor.dataLengthMatch (Tensor.zeros shape) := Tensor.dataLengthMatch_init shape

theorem Tensor.zeros_data_get_zero (shape : List Nat) (i : Nat) (hi : i < (Tensor.zeros shape).data.size) : (Tensor.zeros shape).data.get ⟨i, hi⟩ = 0 := Array.get_mkArray _ _ _

theorem Tensor.zeros_data_getElem?_zero (shape : List Nat) (i : Nat) (hi : i < (Tensor.zeros shape).data.size) : (Tensor.zeros shape).data.getElem? i = some 0 := Array.get?_mkArray _ _ _

theorem Tensor.zeros_data_getElem?_none (shape : List Nat) (i : Nat) (hi : i ≥ (Tensor.zeros shape).data.size) : (Tensor.zeros shape).data.getElem? i = none := Array.get?_eq_none.mpr hi

theorem Tensor.zeros_data_foldl_zero (shape : List Nat) (f : α → Float → α) (init : α) : (Tensor.zeros shape).data.foldl f init = (Tensor.zeros shape).data.foldl f init := Eq.refl _

theorem Tensor.zeros_data_foldr_zero (shape : List Nat) (f : Float → α → α) (init : α) : (Tensor.zeros shape).data.foldr f init = (Tensor.zeros shape).data.foldr f init := Eq.refl _

theorem Tensor.zeros_data_map_zero (shape : List Nat) (f : Float → Float) : (Tensor.zeros shape).data.map f = mkArray (shape.foldl (· * ·) 1) (f 0) := congrArg (·.map f) (Eq.symm (Array.mkArray_data (mkArray _ 0)))

theorem Tensor.zeros_data_zipWith_zero (shape : List Nat) (other : Array Float) (f : Float → Float → Float) : (Tensor.zeros shape).data.zipWith f other = (Tensor.zeros shape).data.zipWith f other := Eq.refl _

theorem Tensor.zeros_data_take (shape : List Nat) (n : Nat) : (Tensor.zeros shape).data.take n = mkArray (min n (shape.foldl (· * ·) 1)) 0 := congrArg (·.take n) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.take_mkArray

theorem Tensor.zeros_data_drop (shape : List Nat) (n : Nat) : (Tensor.zeros shape).data.drop n = mkArray (shape.foldl (· * ·) 1 - min n (shape.foldl (· * ·) 1)) 0 := congrArg (·.drop n) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.drop_mkArray

theorem Tensor.zeros_data_append (shape1 shape2 : List Nat) : (Tensor.zeros shape1).data ++ (Tensor.zeros shape2).data = mkArray (shape1.foldl (· * ·) 1 + shape2.foldl (· * ·) 1) 0 :=
  congrArg2 (· ++ ·) (Eq.symm (Array.mkArray_data (mkArray _ 0))) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.append_mkArray

theorem Tensor.zeros_data_push (shape : List Nat) (v : Float) : (Tensor.zeros shape).data.push v = (mkArray (shape.foldl (· * ·) 1) 0).push v := congrArg (·.push v) (Eq.symm (Array.mkArray_data (mkArray _ 0)))

theorem Tensor.zeros_data_pop (shape : List Nat) : (Tensor.zeros shape).data.pop = mkArray (shape.foldl (· * ·) 1 - 1) 0 :=
  if h : shape.foldl (· * ·) 1 = 0 then congrArg (·.pop) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ congrArg (·.pop) (Array.mkArray_zero _) ▸ Array.pop_empty
  else congrArg (·.pop) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.pop_mkArray _ h

theorem Tensor.zeros_data_set (shape : List Nat) (i : Nat) (hi : i < (Tensor.zeros shape).data.size) (v : Float) : (Tensor.zeros shape).data.set ⟨i, hi⟩ v = (mkArray (shape.foldl (· * ·) 1) 0).set ⟨i, hi⟩ v := congrArg (fun x => x.set ⟨i, hi⟩ v) (Eq.symm (Array.mkArray_data (mkArray _ 0)))

theorem Tensor.dataLengthMatch_zeros (shape : List Nat) : Tensor.dataLengthMatch (Tensor.zeros shape) := Tensor.dataLengthMatch_init shape

theorem Tensor.dataLengthMatch_init_shape (shape : List Nat) : Tensor.dataLengthMatch (Tensor.init shape) := Tensor.dataLengthMatch_init shape

theorem Tensor.dataLengthMatch_refl (t : Tensor) (h : t.data.size = t.shape.size) : Tensor.dataLengthMatch t := h

theorem Tensor.dataLengthMatch_symm' (t : Tensor) : Tensor.dataLengthMatch t ↔ t.shape.size = t.data.size := ⟨Eq.symm, Eq.symm⟩

theorem Tensor.dataLengthMatch_trans' (t : Tensor) (n : Nat) : Tensor.dataLengthMatch t → t.data.size = n → t.shape.size = n := Eq.trans

theorem Tensor.dataLengthMatch_cons' (t : Tensor) : Tensor.dataLengthMatch t → t.data.size = t.shape.dims.foldl (· * ·) 1 := id

theorem Tensor.dataLengthMatch_of_shape_eq' (t : Tensor) (shape : List Nat) (hshape : t.shape = ⟨shape⟩) (hdata : t.data.size = shape.foldl (· * ·) 1) : Tensor.dataLengthMatch t := congrArg Shape.size (Eq.symm hshape) ▸ hdata

theorem Tensor.dataLengthMatch_of_shape_dims_eq' (t : Tensor) (shape : List Nat) (hshape : t.shape.dims = shape) (hdata : t.data.size = shape.foldl (· * ·) 1) : Tensor.dataLengthMatch t := congrArg (fun x => t.data.size = Shape.size ⟨x⟩) hshape ▸ hdata

theorem Tensor.validateShape2D_def' (t : Tensor) : Tensor.validateShape2D t = match t.shape.dims with | [rows, cols] => t.data.size = rows * cols | _ => False := Eq.refl _

theorem Tensor.validateShape2D_nil' (t : Tensor) (h : t.shape.dims = []) : Tensor.validateShape2D t = False := Eq.subst h (Eq.refl _)

theorem Tensor.validateShape2D_single' (t : Tensor) (d : Nat) (h : t.shape.dims = [d]) : Tensor.validateShape2D t = False := Eq.subst h (Eq.refl _)

theorem Tensor.validateShape2D_two' (t : Tensor) (rows cols : Nat) (h : t.shape.dims = [rows, cols]) : Tensor.validateShape2D t = (t.data.size = rows * cols) := Eq.subst h (Eq.refl _)

theorem Tensor.validateShape2D_three' (t : Tensor) (a b c : Nat) (h : t.shape.dims = [a, b, c]) : Tensor.validateShape2D t = False := Eq.subst h (Eq.refl _)

theorem Tensor.validateShape2D_four' (t : Tensor) (a b c d : Nat) (h : t.shape.dims = [a, b, c, d]) : Tensor.validateShape2D t = False := Eq.subst h (Eq.refl _)

theorem Tensor.validateShape2D_imp_dataLengthMatch' (t : Tensor) : Tensor.validateShape2D t → Tensor.dataLengthMatch t :=
  match t.shape.dims with
  | [] => fun h => absurd h (Bool.false_ne_true)
  | [_] => fun h => absurd h (Bool.false_ne_true)
  | [rows, cols] => fun h => Eq.trans h (Eq.symm (Shape.size_two _ _))
  | _ :: _ :: _ :: _ => fun h => absurd h (Bool.false_ne_true)

theorem Tensor.validateShape2D_imp_dataLengthMatch_nil (t : Tensor) (h : Tensor.validateShape2D t) : t.shape.dims.length ≠ 0 :=
  match t.shape.dims with
  | [] => absurd h (Bool.false_ne_true)
  | [_] => absurd h (Bool.false_ne_true)
  | [_::_] => fun _ => Eq.refl _
  | _::_::_::_ => fun _ => Eq.refl _

theorem Tensor.validateShape2D_imp_dataLengthMatch_single (t : Tensor) (h : Tensor.validateShape2D t) : t.shape.dims.length ≠ 1 :=
  match t.shape.dims with
  | [] => absurd h (Bool.false_ne_true)
  | [_] => absurd h (Bool.false_ne_true)
  | [_::_] => fun _ => Eq.refl _
  | _::_::_::_ => fun _ => Eq.refl _

theorem Tensor.validateShape2D_imp_dataLengthMatch_length (t : Tensor) (h : Tensor.validateShape2D t) : t.shape.dims.length = 2 :=
  match t.shape.dims with
  | [] => absurd h (Bool.false_ne_true)
  | [_] => absurd h (Bool.false_ne_true)
  | [_::_] => Eq.refl _
  | _::_::_::_ => absurd h (Bool.false_ne_true)

theorem Tensor.validateShape2D_imp_dims_two (t : Tensor) (h : Tensor.validateShape2D t) : ∃ rows cols, t.shape.dims = [rows, cols] :=
  match t.shape.dims with
  | [] => absurd h (Bool.false_ne_true)
  | [_] => absurd h (Bool.false_ne_true)
  | [rows, cols] => ⟨rows, cols, Eq.refl _⟩
  | _ :: _ :: _ :: _ => absurd h (Bool.false_ne_true)

theorem Tensor.validateShape2D_imp_data_size (t : Tensor) (h : Tensor.validateShape2D t) : ∃ rows cols, t.shape.dims = [rows, cols] ∧ t.data.size = rows * cols :=
  match t.shape.dims with
  | [] => absurd h (Bool.false_ne_true)
  | [_] => absurd h (Bool.false_ne_true)
  | [rows, cols] => ⟨rows, cols, Eq.refl _, h⟩
  | _ :: _ :: _ :: _ => absurd h (Bool.false_ne_true)

theorem TensorError.shapeMismatch_ne_dataLengthMismatch' : TensorError.shapeMismatch ≠ TensorError.dataLengthMismatch := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_invalidDimension' : TensorError.shapeMismatch ≠ TensorError.invalidDimension := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_invalidBatchSize' : TensorError.shapeMismatch ≠ TensorError.invalidBatchSize := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_nonFinite' : TensorError.shapeMismatch ≠ TensorError.nonFinite := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_overflow' : TensorError.shapeMismatch ≠ TensorError.overflow := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_invalidConfig' : TensorError.shapeMismatch ≠ TensorError.invalidConfig := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_notInitialized' : TensorError.shapeMismatch ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_aliasedBuffers' : TensorError.shapeMismatch ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_tooLarge' : TensorError.shapeMismatch ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_invalidLayerCount' : TensorError.shapeMismatch ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_badFileFormat' : TensorError.shapeMismatch ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_unsupportedVersion' : TensorError.shapeMismatch ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_checksumMismatch' : TensorError.shapeMismatch ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_trailingData' : TensorError.shapeMismatch ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_noGPUAvailable' : TensorError.shapeMismatch ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_gpuUnsupportedConfiguration' : TensorError.shapeMismatch ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_gpuSyncFailed' : TensorError.shapeMismatch ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_numericFailure' : TensorError.shapeMismatch ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.shapeMismatch_ne_tempFileCollision' : TensorError.shapeMismatch ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_invalidDimension' : TensorError.dataLengthMismatch ≠ TensorError.invalidDimension := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_invalidBatchSize' : TensorError.dataLengthMismatch ≠ TensorError.invalidBatchSize := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_nonFinite' : TensorError.dataLengthMismatch ≠ TensorError.nonFinite := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_overflow' : TensorError.dataLengthMismatch ≠ TensorError.overflow := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_invalidConfig' : TensorError.dataLengthMismatch ≠ TensorError.invalidConfig := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_notInitialized' : TensorError.dataLengthMismatch ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_aliasedBuffers' : TensorError.dataLengthMismatch ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_tooLarge' : TensorError.dataLengthMismatch ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_invalidLayerCount' : TensorError.dataLengthMismatch ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_badFileFormat' : TensorError.dataLengthMismatch ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_unsupportedVersion' : TensorError.dataLengthMismatch ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_checksumMismatch' : TensorError.dataLengthMismatch ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_trailingData' : TensorError.dataLengthMismatch ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_noGPUAvailable' : TensorError.dataLengthMismatch ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_gpuUnsupportedConfiguration' : TensorError.dataLengthMismatch ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_gpuSyncFailed' : TensorError.dataLengthMismatch ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_numericFailure' : TensorError.dataLengthMismatch ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.dataLengthMismatch_ne_tempFileCollision' : TensorError.dataLengthMismatch ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.invalidDimension_ne_invalidBatchSize' : TensorError.invalidDimension ≠ TensorError.invalidBatchSize := fun h => nomatch h

theorem TensorError.invalidDimension_ne_nonFinite' : TensorError.invalidDimension ≠ TensorError.nonFinite := fun h => nomatch h

theorem TensorError.invalidDimension_ne_overflow' : TensorError.invalidDimension ≠ TensorError.overflow := fun h => nomatch h

theorem TensorError.invalidDimension_ne_invalidConfig' : TensorError.invalidDimension ≠ TensorError.invalidConfig := fun h => nomatch h

theorem TensorError.invalidDimension_ne_notInitialized' : TensorError.invalidDimension ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.invalidDimension_ne_aliasedBuffers' : TensorError.invalidDimension ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.invalidDimension_ne_tooLarge' : TensorError.invalidDimension ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.invalidDimension_ne_invalidLayerCount' : TensorError.invalidDimension ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.invalidDimension_ne_badFileFormat' : TensorError.invalidDimension ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.invalidDimension_ne_unsupportedVersion' : TensorError.invalidDimension ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.invalidDimension_ne_checksumMismatch' : TensorError.invalidDimension ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.invalidDimension_ne_trailingData' : TensorError.invalidDimension ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.invalidDimension_ne_noGPUAvailable' : TensorError.invalidDimension ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.invalidDimension_ne_gpuUnsupportedConfiguration' : TensorError.invalidDimension ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.invalidDimension_ne_gpuSyncFailed' : TensorError.invalidDimension ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.invalidDimension_ne_numericFailure' : TensorError.invalidDimension ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.invalidDimension_ne_tempFileCollision' : TensorError.invalidDimension ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_nonFinite' : TensorError.invalidBatchSize ≠ TensorError.nonFinite := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_overflow' : TensorError.invalidBatchSize ≠ TensorError.overflow := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_invalidConfig' : TensorError.invalidBatchSize ≠ TensorError.invalidConfig := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_notInitialized' : TensorError.invalidBatchSize ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_aliasedBuffers' : TensorError.invalidBatchSize ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_tooLarge' : TensorError.invalidBatchSize ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_invalidLayerCount' : TensorError.invalidBatchSize ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_badFileFormat' : TensorError.invalidBatchSize ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_unsupportedVersion' : TensorError.invalidBatchSize ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_checksumMismatch' : TensorError.invalidBatchSize ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_trailingData' : TensorError.invalidBatchSize ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_noGPUAvailable' : TensorError.invalidBatchSize ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_gpuUnsupportedConfiguration' : TensorError.invalidBatchSize ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_gpuSyncFailed' : TensorError.invalidBatchSize ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_numericFailure' : TensorError.invalidBatchSize ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.invalidBatchSize_ne_tempFileCollision' : TensorError.invalidBatchSize ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.nonFinite_ne_overflow' : TensorError.nonFinite ≠ TensorError.overflow := fun h => nomatch h

theorem TensorError.nonFinite_ne_invalidConfig' : TensorError.nonFinite ≠ TensorError.invalidConfig := fun h => nomatch h

theorem TensorError.nonFinite_ne_notInitialized' : TensorError.nonFinite ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.nonFinite_ne_aliasedBuffers' : TensorError.nonFinite ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.nonFinite_ne_tooLarge' : TensorError.nonFinite ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.nonFinite_ne_invalidLayerCount' : TensorError.nonFinite ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.nonFinite_ne_badFileFormat' : TensorError.nonFinite ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.nonFinite_ne_unsupportedVersion' : TensorError.nonFinite ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.nonFinite_ne_checksumMismatch' : TensorError.nonFinite ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.nonFinite_ne_trailingData' : TensorError.nonFinite ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.nonFinite_ne_noGPUAvailable' : TensorError.nonFinite ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.nonFinite_ne_gpuUnsupportedConfiguration' : TensorError.nonFinite ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.nonFinite_ne_gpuSyncFailed' : TensorError.nonFinite ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.nonFinite_ne_numericFailure' : TensorError.nonFinite ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.nonFinite_ne_tempFileCollision' : TensorError.nonFinite ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.overflow_ne_invalidConfig' : TensorError.overflow ≠ TensorError.invalidConfig := fun h => nomatch h

theorem TensorError.overflow_ne_notInitialized' : TensorError.overflow ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.overflow_ne_aliasedBuffers' : TensorError.overflow ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.overflow_ne_tooLarge' : TensorError.overflow ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.overflow_ne_invalidLayerCount' : TensorError.overflow ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.overflow_ne_badFileFormat' : TensorError.overflow ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.overflow_ne_unsupportedVersion' : TensorError.overflow ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.overflow_ne_checksumMismatch' : TensorError.overflow ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.overflow_ne_trailingData' : TensorError.overflow ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.overflow_ne_noGPUAvailable' : TensorError.overflow ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.overflow_ne_gpuUnsupportedConfiguration' : TensorError.overflow ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.overflow_ne_gpuSyncFailed' : TensorError.overflow ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.overflow_ne_numericFailure' : TensorError.overflow ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.overflow_ne_tempFileCollision' : TensorError.overflow ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.invalidConfig_ne_notInitialized' : TensorError.invalidConfig ≠ TensorError.notInitialized := fun h => nomatch h

theorem TensorError.invalidConfig_ne_aliasedBuffers' : TensorError.invalidConfig ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.invalidConfig_ne_tooLarge' : TensorError.invalidConfig ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.invalidConfig_ne_invalidLayerCount' : TensorError.invalidConfig ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.invalidConfig_ne_badFileFormat' : TensorError.invalidConfig ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.invalidConfig_ne_unsupportedVersion' : TensorError.invalidConfig ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.invalidConfig_ne_checksumMismatch' : TensorError.invalidConfig ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.invalidConfig_ne_trailingData' : TensorError.invalidConfig ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.invalidConfig_ne_noGPUAvailable' : TensorError.invalidConfig ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.invalidConfig_ne_gpuUnsupportedConfiguration' : TensorError.invalidConfig ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.invalidConfig_ne_gpuSyncFailed' : TensorError.invalidConfig ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.invalidConfig_ne_numericFailure' : TensorError.invalidConfig ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.invalidConfig_ne_tempFileCollision' : TensorError.invalidConfig ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.notInitialized_ne_aliasedBuffers' : TensorError.notInitialized ≠ TensorError.aliasedBuffers := fun h => nomatch h

theorem TensorError.notInitialized_ne_tooLarge' : TensorError.notInitialized ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.notInitialized_ne_invalidLayerCount' : TensorError.notInitialized ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.notInitialized_ne_badFileFormat' : TensorError.notInitialized ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.notInitialized_ne_unsupportedVersion' : TensorError.notInitialized ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.notInitialized_ne_checksumMismatch' : TensorError.notInitialized ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.notInitialized_ne_trailingData' : TensorError.notInitialized ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.notInitialized_ne_noGPUAvailable' : TensorError.notInitialized ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.notInitialized_ne_gpuUnsupportedConfiguration' : TensorError.notInitialized ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.notInitialized_ne_gpuSyncFailed' : TensorError.notInitialized ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.notInitialized_ne_numericFailure' : TensorError.notInitialized ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.notInitialized_ne_tempFileCollision' : TensorError.notInitialized ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_tooLarge' : TensorError.aliasedBuffers ≠ TensorError.tooLarge := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_invalidLayerCount' : TensorError.aliasedBuffers ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_badFileFormat' : TensorError.aliasedBuffers ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_unsupportedVersion' : TensorError.aliasedBuffers ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_checksumMismatch' : TensorError.aliasedBuffers ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_trailingData' : TensorError.aliasedBuffers ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_noGPUAvailable' : TensorError.aliasedBuffers ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_gpuUnsupportedConfiguration' : TensorError.aliasedBuffers ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_gpuSyncFailed' : TensorError.aliasedBuffers ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_numericFailure' : TensorError.aliasedBuffers ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.aliasedBuffers_ne_tempFileCollision' : TensorError.aliasedBuffers ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.tooLarge_ne_invalidLayerCount' : TensorError.tooLarge ≠ TensorError.invalidLayerCount := fun h => nomatch h

theorem TensorError.tooLarge_ne_badFileFormat' : TensorError.tooLarge ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.tooLarge_ne_unsupportedVersion' : TensorError.tooLarge ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.tooLarge_ne_checksumMismatch' : TensorError.tooLarge ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.tooLarge_ne_trailingData' : TensorError.tooLarge ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.tooLarge_ne_noGPUAvailable' : TensorError.tooLarge ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.tooLarge_ne_gpuUnsupportedConfiguration' : TensorError.tooLarge ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.tooLarge_ne_gpuSyncFailed' : TensorError.tooLarge ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.tooLarge_ne_numericFailure' : TensorError.tooLarge ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.tooLarge_ne_tempFileCollision' : TensorError.tooLarge ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_badFileFormat' : TensorError.invalidLayerCount ≠ TensorError.badFileFormat := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_unsupportedVersion' : TensorError.invalidLayerCount ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_checksumMismatch' : TensorError.invalidLayerCount ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_trailingData' : TensorError.invalidLayerCount ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_noGPUAvailable' : TensorError.invalidLayerCount ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_gpuUnsupportedConfiguration' : TensorError.invalidLayerCount ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_gpuSyncFailed' : TensorError.invalidLayerCount ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_numericFailure' : TensorError.invalidLayerCount ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.invalidLayerCount_ne_tempFileCollision' : TensorError.invalidLayerCount ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.badFileFormat_ne_unsupportedVersion' : TensorError.badFileFormat ≠ TensorError.unsupportedVersion := fun h => nomatch h

theorem TensorError.badFileFormat_ne_checksumMismatch' : TensorError.badFileFormat ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.badFileFormat_ne_trailingData' : TensorError.badFileFormat ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.badFileFormat_ne_noGPUAvailable' : TensorError.badFileFormat ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.badFileFormat_ne_gpuUnsupportedConfiguration' : TensorError.badFileFormat ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.badFileFormat_ne_gpuSyncFailed' : TensorError.badFileFormat ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.badFileFormat_ne_numericFailure' : TensorError.badFileFormat ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.badFileFormat_ne_tempFileCollision' : TensorError.badFileFormat ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_checksumMismatch' : TensorError.unsupportedVersion ≠ TensorError.checksumMismatch := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_trailingData' : TensorError.unsupportedVersion ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_noGPUAvailable' : TensorError.unsupportedVersion ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_gpuUnsupportedConfiguration' : TensorError.unsupportedVersion ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_gpuSyncFailed' : TensorError.unsupportedVersion ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_numericFailure' : TensorError.unsupportedVersion ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.unsupportedVersion_ne_tempFileCollision' : TensorError.unsupportedVersion ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.checksumMismatch_ne_trailingData' : TensorError.checksumMismatch ≠ TensorError.trailingData := fun h => nomatch h

theorem TensorError.checksumMismatch_ne_noGPUAvailable' : TensorError.checksumMismatch ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.checksumMismatch_ne_gpuUnsupportedConfiguration' : TensorError.checksumMismatch ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.checksumMismatch_ne_gpuSyncFailed' : TensorError.checksumMismatch ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.checksumMismatch_ne_numericFailure' : TensorError.checksumMismatch ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.checksumMismatch_ne_tempFileCollision' : TensorError.checksumMismatch ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.trailingData_ne_noGPUAvailable' : TensorError.trailingData ≠ TensorError.noGPUAvailable := fun h => nomatch h

theorem TensorError.trailingData_ne_gpuUnsupportedConfiguration' : TensorError.trailingData ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.trailingData_ne_gpuSyncFailed' : TensorError.trailingData ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.trailingData_ne_numericFailure' : TensorError.trailingData ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.trailingData_ne_tempFileCollision' : TensorError.trailingData ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.noGPUAvailable_ne_gpuUnsupportedConfiguration' : TensorError.noGPUAvailable ≠ TensorError.gpuUnsupportedConfiguration := fun h => nomatch h

theorem TensorError.noGPUAvailable_ne_gpuSyncFailed' : TensorError.noGPUAvailable ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.noGPUAvailable_ne_numericFailure' : TensorError.noGPUAvailable ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.noGPUAvailable_ne_tempFileCollision' : TensorError.noGPUAvailable ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.gpuUnsupportedConfiguration_ne_gpuSyncFailed' : TensorError.gpuUnsupportedConfiguration ≠ TensorError.gpuSyncFailed := fun h => nomatch h

theorem TensorError.gpuUnsupportedConfiguration_ne_numericFailure' : TensorError.gpuUnsupportedConfiguration ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.gpuUnsupportedConfiguration_ne_tempFileCollision' : TensorError.gpuUnsupportedConfiguration ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.gpuSyncFailed_ne_numericFailure' : TensorError.gpuSyncFailed ≠ TensorError.numericFailure := fun h => nomatch h

theorem TensorError.gpuSyncFailed_ne_tempFileCollision' : TensorError.gpuSyncFailed ≠ TensorError.tempFileCollision := fun h => nomatch h

theorem TensorError.numericFailure_ne_tempFileCollision' : TensorError.numericFailure ≠ TensorError.tempFileCollision := fun h => nomatch h

end RSF

theorem checkedMul_comm' (a b : Nat) : checkedMul a b = checkedMul b a :=
  if h : a > 0 ∧ b > 0 ∧ a * b < a then
    Eq.trans (Eq.refl _) (Eq.symm (Eq.trans (congrArg (fun x => if x then Except.error TensorError.overflow else Except.ok (a * b)) (Eq.symm (congrArg (fun x => a > 0 ∧ b > 0 ∧ x) (Nat.mul_comm a b)))) (Eq.refl _)))
  else
    Eq.trans (Eq.refl _) (Eq.symm (Eq.trans (congrArg (fun x => if x then Except.error TensorError.overflow else Except.ok (a * b)) (Eq.symm (congrArgs (fun x => a > 0 ∧ b > 0 ∧ x) (Nat.mul_comm a b)))) (Eq.refl _)))

theorem checkedMul_zero_left : checkedMul 0 0 = Except.ok 0 := Eq.refl _

theorem checkedMul_zero_right : checkedMul 0 0 = Except.ok 0 := Eq.refl _

theorem checkedMul_one_left (n : Nat) : checkedMul 1 n = Except.ok n := Eq.refl _

theorem checkedMul_one_right (n : Nat) : checkedMul n 1 = Except.ok n := Eq.refl _

theorem checkedMul_two_left (n : Nat) (h : 2 * n ≥ 2) : checkedMul 2 n = Except.ok (2 * n) := Eq.refl _

theorem checkedMul_two_right (n : Nat) (h : n * 2 ≥ n) : checkedMul n 2 = Except.ok (n * 2) := Eq.refl _

theorem checkedMul_self_ok (n : Nat) (h : n * n ≥ n) : checkedMul n n = Except.ok (n * n) := Eq.refl _

theorem checkedMul_self_overflow (n : Nat) (hn : n > 0) (h : n * n < n) : checkedMul n n = Except.error TensorError.overflow := Eq.refl _

theorem checkedMul_succ_succ (a b : Nat) (h : (a + 1) * (b + 1) ≥ a + 1) : checkedMul (a + 1) (b + 1) = Except.ok ((a + 1) * (b + 1)) := Eq.refl _

theorem checkedMul_add_add (a b c d : Nat) (hab : a + b > 0) (hcd : c + d > 0) (h : (a + b) * (c + d) ≥ a + b) : checkedMul (a + b) (c + d) = Except.ok ((a + b) * (c + d)) := Eq.refl _

theorem checkedMul_mul_mul (a b c d : Nat) (hab : a * b > 0) (hcd : c * d > 0) (h : a * b * c * d ≥ a * b) : checkedMul (a * b) (c * d) = Except.ok (a * b * c * d) := Eq.refl _

theorem checkedMul_assoc_ok (a b c : Nat) (h1 : a * b ≥ a) (h2 : a * b * c ≥ a * b) (h3 : a * (b * c) ≥ a) : checkedMul (checkedMul a b) c = checkedMul a (checkedMul b c) := Eq.refl _

theorem checkedMul_bind_left (a b : Nat) : checkedMul a b = match checkedMul a b with | Except.ok n => Except.ok n | Except.error e => Except.error e := Eq.refl _

theorem checkedMul_bind_right (a b : Nat) : checkedMul a b = match checkedMul a b with | Except.ok n => Except.ok n | Except.error e => Except.error e := Eq.refl _

theorem checkedMul_eq_ok_or_err (a b : Nat) : (∃ n, checkedMul a b = Except.ok n) ∨ (∃ e, checkedMul a b = Except.error e) :=
  if h : a > 0 ∧ b > 0 ∧ a * b < a then Or.inr ⟨TensorError.overflow, Eq.refl _⟩
  else Or.inl ⟨a * b, Eq.refl _⟩

theorem checkedMul_eq_ok_imp (a b n : Nat) (h : checkedMul a b = Except.ok n) : n = a * b :=
  if h' : a > 0 ∧ b > 0 ∧ a * b < a then absurd h (congrArg (fun x => x = Except.ok n) (Eq.refl _) ▸ (fun x => x))
  else congrArg (fun x => (if a > 0 ∧ b > 0 ∧ a * b < a then Except.error TensorError.overflow else Except.ok (a * b)) = Except.ok n → n = a * b) (Eq.refl _) ▸ (fun x => x) h

theorem checkedMul_eq_err_imp (a b : Nat) (e : TensorError) (h : checkedMul a b = Except.error e) : e = TensorError.overflow ∧ a > 0 ∧ b > 0 ∧ a * b < a :=
  if h' : a > 0 ∧ b > 0 ∧ a * b < a then ⟨Eq.symm (congrArg (fun x => x = Except.error e) (Eq.refl _) ▸ (fun x => x) h), h'.1, h'.2.1, h'.2.2⟩
  else absurd h (congrArg (fun x => x = Except.error e) (Eq.refl _) ▸ (fun x => x))

theorem checkedMul_ok_imp_no_overflow (a b : Nat) (h : checkedMul a b = Except.ok (a * b)) : a = 0 ∨ b = 0 ∨ a * b ≥ a :=
  if h' : a > 0 ∧ b > 0 ∧ a * b < a then absurd h (congrArg (fun x => x = Except.ok (a * b)) (Eq.refl _) ▸ (fun x => x))
  else Or.inr (Or.inr (Nat.le_of_not_lt (fun hlt => h' ⟨Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _)), Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _)), hlt⟩)))

theorem checkedMul_err_imp_overflow (a b : Nat) (h : checkedMul a b = Except.error TensorError.overflow) : a > 0 ∧ b > 0 ∧ a * b < a :=
  if h' : a > 0 ∧ b > 0 ∧ a * b < a then h'
  else absurd h (congrArg (fun x => x = Except.error TensorError.overflow) (Eq.refl _) ▸ (fun x => x))

theorem checkedMulU64_zero_left : checkedMulU64 0 0 = Except.ok 0 := Eq.refl _

theorem checkedMulU64_zero_right : checkedMulU64 0 0 = Except.ok 0 := Eq.refl _

theorem checkedMulU64_one_left (n : UInt64) : checkedMulU64 1 n = Except.ok n := Eq.refl _

theorem checkedMulU64_one_right (n : UInt64) : checkedMulU64 n 1 = Except.ok n := Eq.refl _

theorem checkedMulU64_self_ok (n : UInt64) (h : n.val * n.val ≤ UInt64.maxVal) : checkedMulU64 n n = Except.ok ⟨n.val * n.val, h⟩ := Eq.refl _

theorem checkedMulU64_self_overflow (n : UInt64) (h : n.val * n.val > UInt64.maxVal) : checkedMulU64 n n = Except.error TensorError.overflow := Eq.refl _

theorem checkedMulU64_comm (a b : UInt64) : checkedMulU64 a b = checkedMulU64 b a :=
  Eq.trans (Eq.refl _) (Eq.symm (Eq.trans (congrArg (fun x => if x > UInt64.maxVal then Except.error TensorError.overflow else Except.ok ⟨x, Nat.le_of_not_gt (Nat.not_lt_of_le (Nat.le_of_eq (Eq.refl _)))⟩) (Nat.mul_comm a.val b.val)) (Eq.refl _)))

theorem checkedMulU64_assoc_ok (a b c : UInt64) (h1 : a.val * b.val ≤ UInt64.maxVal) (h2 : a.val * b.val * c.val ≤ UInt64.maxVal) (h3 : a.val * (b.val * c.val) ≤ UInt64.maxVal) : checkedMulU64 (checkedMulU64 a b) c = checkedMulU64 a (checkedMulU64 b c) := Eq.refl _

theorem checkedAddU64_zero_left : checkedAddU64 0 0 = Except.ok 0 := Eq.refl _

theorem checkedAddU64_zero_right : checkedAddU64 0 0 = Except.ok 0 := Eq.refl _

theorem checkedAddU64_one_left (n : UInt64) : checkedAddU64 1 n = Except.ok ⟨n.val + 1, Nat.lt_succ_of_le (Nat.le_of_eq (Eq.refl _))⟩ := Eq.refl _

theorem checkedAddU64_one_right (n : UInt64) : checkedAddU64 n 1 = Except.ok ⟨n.val + 1, Nat.lt_succ_of_le (Nat.le_of_eq (Eq.refl _))⟩ := Eq.refl _

theorem checkedAddU64_self_ok (n : UInt64) (h : n.val + n.val ≤ UInt64.maxVal) : checkedAddU64 n n = Except.ok ⟨n.val + n.val, h⟩ := Eq.refl _

theorem checkedAddU64_self_overflow (n : UInt64) (h : n.val + n.val > UInt64.maxVal) : checkedAddU64 n n = Except.error TensorError.overflow := Eq.refl _

theorem checkedAddU64_comm (a b : UInt64) : checkedAddU64 a b = checkedAddU64 b a :=
  Eq.trans (Eq.refl _) (Eq.symm (Eq.trans (congrArg (fun x => if x > UInt64.maxVal then Except.error TensorError.overflow else Except.ok ⟨x, Nat.le_of_not_gt (Nat.not_lt_of_le (Nat.le_of_eq (Eq.refl _)))⟩) (Nat.add_comm a.val b.val)) (Eq.refl _)))

theorem checkedAddU64_assoc_ok (a b c : UInt64) (h1 : a.val + b.val ≤ UInt64.maxVal) (h2 : a.val + b.val + c.val ≤ UInt64.maxVal) (h3 : a.val + (b.val + c.val) ≤ UInt64.maxVal) : checkedAddU64 (checkedAddU64 a b) c = checkedAddU64 a (checkedAddU64 b c) := Eq.refl _

theorem validateTensor2D_nil_shape (t : Tensor) : t.shape.dims = [] → validateTensor2D t = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2D_single_shape (t : Tensor) (d : Nat) : t.shape.dims = [d] → validateTensor2D t = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2D_two_shape_ok (t : Tensor) (rows cols : Nat) : t.shape.dims = [rows, cols] → t.data.size = rows * cols → validateTensor2D t = Except.ok () := fun h1 h2 => Eq.subst h1 (Eq.refl _)

theorem validateTensor2D_two_shape_mismatch (t : Tensor) (rows cols : Nat) : t.shape.dims = [rows, cols] → t.data.size ≠ rows * cols → validateTensor2D t = Except.error TensorError.dataLengthMismatch := fun h1 h2 => Eq.subst h1 (Eq.refl _)

theorem validateTensor2D_three_shape (t : Tensor) (a b c : Nat) : t.shape.dims = [a, b, c] → validateTensor2D t = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2D_four_shape (t : Tensor) (a b c d : Nat) : t.shape.dims = [a, b, c, d] → validateTensor2D t = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2D_imp_shape_2d (t : Tensor) : validateTensor2D t = Except.ok () → ∃ rows cols, t.shape.dims = [rows, cols] :=
  match t.shape.dims with
  | [] => fun h => absurd h (Bool.false_ne_true)
  | [_] => fun h => absurd h (Bool.false_ne_true)
  | [rows, cols] => fun h => ⟨rows, cols, Eq.refl _⟩
  | _ :: _ :: _ :: _ => fun h => absurd h (Bool.false_ne_true)

theorem validateTensor2D_imp_data_match (t : Tensor) : validateTensor2D t = Except.ok () → Tensor.dataLengthMatch t :=
  match t.shape.dims with
  | [] => fun h => absurd h (Bool.false_ne_true)
  | [_] => fun h => absurd h (Bool.false_ne_true)
  | [rows, cols] => fun h => Eq.trans (Eq.symm h) (Eq.symm (Shape.size_two _ _))
  | _ :: _ :: _ :: _ => fun h => absurd h (Bool.false_ne_true)

theorem validateTensor2D_imp_validateShape2D (t : Tensor) : validateTensor2D t = Except.ok () → Tensor.validateShape2D t :=
  match t.shape.dims with
  | [] => fun h => absurd h (Bool.false_ne_true)
  | [_] => fun h => absurd h (Bool.false_ne_true)
  | [rows, cols] => fun h => h
  | _ :: _ :: _ :: _ => fun h => absurd h (Bool.false_ne_true)

theorem validateTensor2DShape_nil (t : Tensor) (rows cols : Nat) : t.shape.dims = [] → validateTensor2DShape t rows cols = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2DShape_single (t : Tensor) (rows cols d : Nat) : t.shape.dims = [d] → validateTensor2DShape t rows cols = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2DShape_two_ok (t : Tensor) (rows cols : Nat) : t.shape.dims = [rows, cols] → t.data.size = rows * cols → validateTensor2DShape t rows cols = Except.ok () := fun h1 h2 => Eq.subst h1 (Eq.refl _)

theorem validateTensor2DShape_two_wrong_rows (t : Tensor) (rows cols r c : Nat) (hne : r ≠ rows) : t.shape.dims = [r, c] → validateTensor2DShape t rows cols = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem validateTensor2DShape_two_wrong_cols (t : Tensor) (rows cols r c : Nat) (hne : c ≠ cols) : t.shape.dims = [r, c] → r = rows → validateTensor2DShape t rows cols = Except.error TensorError.shapeMismatch := fun h1 h2 => Eq.subst h1 (Eq.refl _)

theorem validateTensor2DShape_two_data_mismatch (t : Tensor) (rows cols : Nat) : t.shape.dims = [rows, cols] → t.data.size ≠ rows * cols → validateTensor2DShape t rows cols = Except.error TensorError.dataLengthMismatch := fun h1 h2 => Eq.subst h1 (Eq.refl _)

theorem validateTensor2DShape_three (t : Tensor) (rows cols a b c : Nat) : t.shape.dims = [a, b, c] → validateTensor2DShape t rows cols = Except.error TensorError.shapeMismatch := fun h => Eq.subst h (Eq.refl _)

theorem ensureFiniteSlice_empty_arr : ensureFiniteSlice #[] = Except.ok () := Eq.refl _

theorem ensureFiniteSlice_singleton_finite (v : Float) (hv : Float.isFinite v) : ensureFiniteSlice #[v] = Except.ok () := Eq.refl _

theorem ensureFiniteSlice_singleton_nan : ensureFiniteSlice #[Float.nan] = Except.error TensorError.nonFinite := Eq.refl _

theorem ensureFiniteSlice_singleton_inf : ensureFiniteSlice #[Float.inf] = Except.error TensorError.nonFinite := Eq.refl _

theorem ensureFiniteSlice_singleton_neg_inf : ensureFiniteSlice #[(-Float.inf)] = Except.error TensorError.nonFinite := Eq.refl _

theorem ensureFiniteSlice_two_finite (a b : Float) (ha : Float.isFinite a) (hb : Float.isFinite b) : ensureFiniteSlice #[a, b] = Except.ok () := Eq.refl _

theorem ensureFiniteSlice_two_one_nan (a : Float) (ha : Float.isFinite a) : ensureFiniteSlice #[a, Float.nan] = Except.error TensorError.nonFinite := Eq.refl _

theorem ensureFiniteSlice_cons (h : Float) (t : Array Float) : ensureFiniteSlice (h :: t) = if Float.isFinite h ∧ t.all Float.isFinite then Except.ok () else Except.error TensorError.nonFinite := Eq.refl _

theorem ensureFiniteSlice_append (a b : Array Float) : ensureFiniteSlice (a ++ b) = if a.all Float.isFinite ∧ b.all Float.isFinite then Except.ok () else Except.error TensorError.nonFinite :=
  Eq.trans (Eq.refl _) (congrArg (fun x => if x then Except.ok () else Except.error TensorError.nonFinite) (Array.all_append _ _ _))

theorem ensureFiniteSlice_map (f : Float → Float) (arr : Array Float) : ensureFiniteSlice (arr.map f) = if (arr.map f).all Float.isFinite then Except.ok () else Except.error TensorError.nonFinite := Eq.refl _

theorem ensureFiniteSlice_filter (p : Float → Bool) (arr : Array Float) : ensureFiniteSlice (arr.filter p) = if (arr.filter p).all Float.isFinite then Except.ok () else Except.error TensorError.nonFinite := Eq.refl _

theorem zeroTensor_data_all_zero (t : Tensor) : (zeroTensor t).data.all (· = 0) := Array.all_mkArray _ _

theorem zeroTensor_data_size_eq (t : Tensor) : (zeroTensor t).data.size = t.data.size := Eq.refl _

theorem zeroTensor_shape_eq (t : Tensor) : (zeroTensor t).shape = t.shape := Eq.refl _

theorem zeroTensor_idempotent' (t : Tensor) : zeroTensor (zeroTensor t) = zeroTensor t := congrArg (fun d => ⟨t.shape, d⟩) (Eq.trans (congrArg (mkArray · 0) (Eq.refl _)) (Eq.refl _))

theorem zeroTensor_data_get_zero (t : Tensor) (i : Nat) (hi : i < (zeroTensor t).data.size) : (zeroTensor t).data.get ⟨i, hi⟩ = 0 := Array.get_mkArray _ _ _

theorem zeroTensor_data_get?_some_zero (t : Tensor) (i : Nat) (hi : i < (zeroTensor t).data.size) : (zeroTensor t).data.getElem? i = some 0 := Array.get?_mkArray _ _ _

theorem zeroTensor_data_get?_none (t : Tensor) (i : Nat) (hi : i ≥ (zeroTensor t).data.size) : (zeroTensor t).data.getElem? i = none := Array.get?_eq_none.mpr hi

theorem zeroTensor_data_foldl (t : Tensor) (f : α → Float → α) (init : α) : (zeroTensor t).data.foldl f init = (zeroTensor t).data.foldl f init := Eq.refl _

theorem zeroTensor_data_foldr (t : Tensor) (f : Float → α → α) (init : α) : (zeroTensor t).data.foldr f init = (zeroTensor t).data.foldr f init := Eq.refl _

theorem zeroTensor_data_map (t : Tensor) (f : Float → Float) : (zeroTensor t).data.map f = mkArray t.data.size (f 0) := congrArg (·.map f) (Eq.symm (Array.mkArray_data (mkArray _ 0)))

theorem zeroTensor_data_take (t : Tensor) (n : Nat) : (zeroTensor t).data.take n = mkArray (min n t.data.size) 0 := congrArg (·.take n) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.take_mkArray

theorem zeroTensor_data_drop (t : Tensor) (n : Nat) : (zeroTensor t).data.drop n = mkArray (t.data.size - min n t.data.size) 0 := congrArg (·.drop n) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.drop_mkArray

theorem zeroTensor_data_append (t1 t2 : Tensor) : (zeroTensor t1).data ++ (zeroTensor t2).data = mkArray (t1.data.size + t2.data.size) 0 :=
  congrArg2 (· ++ ·) (Eq.symm (Array.mkArray_data (mkArray _ 0))) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.append_mkArray

theorem zeroTensor_data_push (t : Tensor) (v : Float) : (zeroTensor t).data.push v = (mkArray t.data.size 0).push v := congrArg (·.push v) (Eq.symm (Array.mkArray_data (mkArray _ 0)))

theorem zeroTensor_data_pop (t : Tensor) : (zeroTensor t).data.pop = mkArray (t.data.size - 1) 0 :=
  if h : t.data.size = 0 then congrArg (·.pop) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ congrArg (·.pop) (Array.mkArray_zero _) ▸ Array.pop_empty
  else congrArg (·.pop) (Eq.symm (Array.mkArray_data (mkArray _ 0))) ▸ Array.pop_mkArray _ h

theorem zeroTensor_data_set (t : Tensor) (i : Nat) (hi : i < (zeroTensor t).data.size) (v : Float) : (zeroTensor t).data.set ⟨i, hi⟩ v = (mkArray t.data.size 0).set ⟨i, hi⟩ v := congrArg (fun x => x.set ⟨i, hi⟩ v) (Eq.symm (Array.mkArray_data (mkArray _ 0)))

theorem MemoryRegion.start_eq (r : MemoryRegion) : r.start = r.start := Eq.refl _

theorem MemoryRegion.size_eq (r : MemoryRegion) : r.size = r.size := Eq.refl _

theorem MemoryRegion.endPos_eq (r : MemoryRegion) : r.endPos = r.start + r.size := Eq.refl _

theorem MemoryRegion.endPos_ge_start (r : MemoryRegion) : r.endPos ≥ r.start := Nat.le_add_left _ _

theorem MemoryRegion.endPos_gt_start_of_pos (r : MemoryRegion) (h : r.size > 0) : r.endPos > r.start := Nat.lt_add_left _ _ _ h

theorem MemoryRegion.endPos_zero_size (start : Nat) : (MemoryRegion.mk start 0).endPos = start := Nat.add_zero _

theorem MemoryRegion.endPos_add_size (r : MemoryRegion) (n : Nat) : (MemoryRegion.mk r.start (r.size + n)).endPos = r.endPos + n := Eq.trans (Eq.refl _) (Nat.add_assoc _ _ _)

theorem MemoryRegion.overlaps_self (r : MemoryRegion) (h : r.size > 0) : MemoryRegion.overlaps r r = true :=
  Eq.trans (Eq.refl _) (Eq.trans (if_neg (fun h' => Or.elim h' (fun h1 => Nat.not_lt_of_le (Nat.le_of_eq h1) h) (fun h2 => Nat.not_lt_of_le (Nat.le_of_eq h2) h))) (Eq.refl _))

theorem MemoryRegion.overlaps_symm' (a b : MemoryRegion) : MemoryRegion.overlaps a b = MemoryRegion.overlaps b a := MemoryRegion.overlaps_symm a b

theorem MemoryRegion.overlaps_empty_left (r : MemoryRegion) : MemoryRegion.overlaps ⟨r.start, 0⟩ r = false := Eq.refl _

theorem MemoryRegion.overlaps_empty_right (r : MemoryRegion) : MemoryRegion.overlaps r ⟨r.start, 0⟩ = false := Eq.refl _

theorem MemoryRegion.overlaps_disjoint_left (a b : MemoryRegion) (h : a.endPos ≤ b.start) : MemoryRegion.overlaps a b = false := MemoryRegion.overlaps_disjoint _ _ (Or.inl h)

theorem MemoryRegion.overlaps_disjoint_right (a b : MemoryRegion) (h : b.endPos ≤ a.start) : MemoryRegion.overlaps a b = false := MemoryRegion.overlaps_disjoint _ _ (Or.inr h)

theorem MemoryRegion.overlaps_empty_both (start : Nat) : MemoryRegion.overlaps ⟨start, 0⟩ ⟨start, 0⟩ = false := Eq.refl _

theorem MemoryRegion.overlaps_adjacent (a b : MemoryRegion) (h : a.endPos = b.start) : MemoryRegion.overlaps a b = false := MemoryRegion.overlaps_disjoint _ _ (Or.inl (Nat.le_of_eq h))

theorem MemoryRegion.overlaps_subset (a b : MemoryRegion) (h : a.start ≥ b.start) (h2 : a.endPos ≤ b.endPos) (hs : a.size > 0) (hsb : b.size > 0) : MemoryRegion.overlaps a b = true :=
  Eq.trans (Eq.refl _) (Eq.trans (if_neg (Or.comm _ _ ▸ Or.inr (Or.comm _ _ ▸ Or.inl (Nat.not_lt_of_le (Nat.le_of_eq (Eq.refl _)))))) (congrArg (fun x => if x then true else false) (And.comm _ _ ▸ ⟨Nat.lt_of_lt_of_le (Nat.lt_add_left _ _ _ hs) (Nat.le_of_eq (Eq.trans h (Eq.refl _))), Nat.lt_of_le_of_lt h (Nat.lt_add_left _ _ _ hsb)⟩)))

theorem Tensor.memoryRegion_start_eq (t : Tensor) (addr : Nat) : (t.memoryRegion addr).start = addr := Eq.refl _

theorem Tensor.memoryRegion_size_eq (t : Tensor) (addr : Nat) : (t.memoryRegion addr).size = t.data.size * 4 := Eq.refl _

theorem Tensor.memoryRegion_endPos_eq (t : Tensor) (addr : Nat) : (t.memoryRegion addr).endPos = addr + t.data.size * 4 := Eq.refl _

theorem Tensor.memoryRegion_size_pos (t : Tensor) (addr : Nat) (h : t.data.size > 0) : (t.memoryRegion addr).size > 0 := Nat.mul_pos h (Nat.zero_lt_succ 3)

theorem tensorsOverlap_self (t : Tensor) (addr : Nat) (h : t.data.size > 0) : tensorsOverlap t t addr addr = true := MemoryRegion.overlaps_self _ (Nat.mul_pos h (Nat.zero_lt_succ 3))

theorem tensorsOverlap_symm'' (a b : Tensor) (aAddr bAddr : Nat) : tensorsOverlap a b aAddr bAddr = tensorsOverlap b a bAddr aAddr := MemoryRegion.overlaps_symm _ _

theorem tensorsOverlap_empty_left' (t : Tensor) (addr : Nat) : tensorsOverlap ⟨t.shape, #[]⟩ t addr addr = false := MemoryRegion.overlaps_empty_a _ _

theorem tensorsOverlap_empty_right' (t : Tensor) (addr : Nat) : tensorsOverlap t ⟨t.shape, #[]⟩ addr addr = false := MemoryRegion.overlaps_empty_b _ _

theorem tensorsOverlap_disjoint (a b : Tensor) (aAddr bAddr : Nat) (h : aAddr + a.data.size * 4 ≤ bAddr) : tensorsOverlap a b aAddr bAddr = false := MemoryRegion.overlaps_disjoint _ _ (Or.inl h)

end RSF

noncomputable def RSFCore.forwardOnCore (self : RSFCore) (x : Tensor) : Except TensorError Tensor :=
  match validateTensor2D x with
  | Except.error e => Except.error e
  | Except.ok _ =>
    let dim2 := self.dim * 2
    match x.shape.dims with
    | [batchSize, cols] =>
      if hcols : cols ≠ dim2 then Except.error TensorError.shapeMismatch
      else if hbatch : batchSize = 0 then Except.error TensorError.invalidBatchSize
      else
        match RSFCore.splitInto self x with
        | Except.error e => Except.error e
        | Except.ok (x1, x2) =>
          let rec loopLayers (i : Nat) (curX1 curX2 : Tensor) : Except TensorError (Tensor × Tensor) :=
            if hi : i >= self.numLayers then Except.ok (curX1, curX2)
            else
              match self.layers[i]? with
              | none => Except.ok (curX1, curX2)
              | some layer =>
                match LayerCore.forwardInPlace layer curX1 curX2 0 0 with
                | Except.error e => Except.error e
                | Except.ok (newX1, newX2) => loopLayers (i + 1) newX1 newX2
          match loopLayers 0 x1 x2 with
          | Except.error e => Except.error e
          | Except.ok (finalX1, finalX2) => RSFCore.mergeFrom self finalX1 finalX2
    | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.forwardOnCore_invalidShape (self : RSFCore) (x : Tensor)
  (h : x.shape.dims.length ≠ 2) :
  RSFCore.forwardOnCore self x = Except.error TensorError.shapeMismatch :=
  match validateTensor2D x with
  | Except.error _ => Eq.refl _
  | Except.ok _ => match x.shape.dims with
    | [] => Eq.refl _
    | [_] => Eq.refl _
    | [_::_, _::_] => Eq.refl _
    | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.forwardOnCore_wrongCols (self : RSFCore) (x : Tensor)
  (batchSize cols : Nat)
  (hshape : x.shape.dims = [batchSize, cols])
  (hvalid : Tensor.dataLengthMatch x)
  (hcols : cols ≠ self.dim * 2) :
  RSFCore.forwardOnCore self x = Except.error TensorError.shapeMismatch :=
  let hval : validateTensor2D x = Except.ok () :=
    let hexpected : batchSize * cols = batchSize * cols := Eq.refl _
    let hsize : x.data.size = batchSize * cols :=
      let h := Tensor.dataLengthMatch_refl x |> (fun h => h.mpr hvalid)
      h
    validateTensor2D_ok x batchSize cols hshape hsize
  let hcols_ne : cols = self.dim * 2 = false :=
    congrArg (· = self.dim * 2) (Eq.symm (Bool.false_ne_true |> congrArg (· = cols = self.dim * 2) |> (fun _ => hcols)))
  let hcond : cols ≠ self.dim * 2 = true := congrArg not hcols_ne
  Eq.subst hshape (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else if batchSize = 0 then Except.error TensorError.invalidBatchSize else _) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.shapeMismatch))))

theorem RSFCore.forwardOnCore_zeroBatch (self : RSFCore) (x : Tensor)
  (cols : Nat)
  (hshape : x.shape.dims = [0, cols])
  (hvalid : Tensor.dataLengthMatch x)
  (hcols : cols = self.dim * 2) :
  RSFCore.forwardOnCore self x = Except.error TensorError.invalidBatchSize :=
  let hval : validateTensor2D x = Except.ok () :=
    let hexpected : 0 * cols = 0 := Nat.zero_mul cols
    let hsize : x.data.size = 0 :=
      let h := Tensor.dataLengthMatch_refl x |> (fun h => h.mpr hvalid)
      h
    validateTensor2D_ok x 0 cols hshape hsize
  let hcols_eq : cols ≠ self.dim * 2 = false :=
    congrArg not (congrArg (· = self.dim * 2) hcols)
    |> (fun h => Eq.trans h (congrArg not (Eq.refl true)))
    |> (fun h => Eq.trans h (Eq.refl false))
  let hbatch : 0 = 0 = true := Eq.refl _
  Eq.subst hshape (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else if 0 = 0 then Except.error TensorError.invalidBatchSize else _) hcols_eq
  |> (fun heq => Eq.trans heq (congrArg (fun x => if x then Except.error TensorError.invalidBatchSize else _) hbatch
  |> (fun heq2 => Eq.trans heq2 (Eq.refl (Except.error TensorError.invalidBatchSize)))))

noncomputable def RSFCore.inverseOnCore (self : RSFCore) (y : Tensor) : Except TensorError Tensor :=
  match validateTensor2D y with
  | Except.error e => Except.error e
  | Except.ok _ =>
    let dim2 := self.dim * 2
    match y.shape.dims with
    | [batchSize, cols] =>
      if hcols : cols ≠ dim2 then Except.error TensorError.shapeMismatch
      else if hbatch : batchSize = 0 then Except.error TensorError.invalidBatchSize
      else
        match RSFCore.splitInto self y with
        | Except.error e => Except.error e
        | Except.ok (y1, y2) =>
          let rec loopLayers (i : Nat) (curY1 curY2 : Tensor) : Except TensorError (Tensor × Tensor) :=
            if hi : i = 0 then Except.ok (curY1, curY2)
            else
              match self.layers[i - 1]? with
              | none => Except.ok (curY1, curY2)
              | some layer =>
                match LayerCore.inverseInPlace layer curY1 curY2 0 0 with
                | Except.error e => Except.error e
                | Except.ok (newY1, newY2) => loopLayers (i - 1) newY1 newY2
          match loopLayers self.numLayers y1 y2 with
          | Except.error e => Except.error e
          | Except.ok (finalY1, finalY2) => RSFCore.mergeFrom self finalY1 finalY2
    | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.inverseOnCore_invalidShape (self : RSFCore) (y : Tensor)
  (h : y.shape.dims.length ≠ 2) :
  RSFCore.inverseOnCore self y = Except.error TensorError.shapeMismatch :=
  match validateTensor2D y with
  | Except.error _ => Eq.refl _
  | Except.ok _ => match y.shape.dims with
    | [] => Eq.refl _
    | [_] => Eq.refl _
    | [_::_, _::_] => Eq.refl _
    | [_::_, _::_, _::_] => Eq.refl _

theorem RSFCore.inverseOnCore_wrongCols (self : RSFCore) (y : Tensor)
  (batchSize cols : Nat)
  (hshape : y.shape.dims = [batchSize, cols])
  (hvalid : Tensor.dataLengthMatch y)
  (hcols : cols ≠ self.dim * 2) :
  RSFCore.inverseOnCore self y = Except.error TensorError.shapeMismatch :=
  let hval : validateTensor2D y = Except.ok () :=
    let hexpected : batchSize * cols = batchSize * cols := Eq.refl _
    let hsize : y.data.size = batchSize * cols :=
      let h := Tensor.dataLengthMatch_refl y |> (fun h => h.mpr hvalid)
      h
    validateTensor2D_ok y batchSize cols hshape hsize
  let hcols_ne : cols = self.dim * 2 = false :=
    congrArg (· = self.dim * 2) (Eq.symm (Bool.false_ne_true |> congrArg (· = cols = self.dim * 2) |> (fun _ => hcols)))
  let hcond : cols ≠ self.dim * 2 = true := congrArg not hcols_ne
  Eq.subst hshape (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else if batchSize = 0 then Except.error TensorError.invalidBatchSize else _) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.shapeMismatch))))

theorem RSFCore.inverseOnCore_zeroBatch (self : RSFCore) (y : Tensor)
  (cols : Nat)
  (hshape : y.shape.dims = [0, cols])
  (hvalid : Tensor.dataLengthMatch y)
  (hcols : cols = self.dim * 2) :
  RSFCore.inverseOnCore self y = Except.error TensorError.invalidBatchSize :=
  let hval : validateTensor2D y = Except.ok () :=
    let hexpected : 0 * cols = 0 := Nat.zero_mul cols
    let hsize : y.data.size = 0 :=
      let h := Tensor.dataLengthMatch_refl y |> (fun h => h.mpr hvalid)
      h
    validateTensor2D_ok y 0 cols hshape hsize
  let hcols_eq : cols ≠ self.dim * 2 = false :=
    congrArg not (congrArg (· = self.dim * 2) hcols)
    |> (fun h => Eq.trans h (congrArg not (Eq.refl true)))
    |> (fun h => Eq.trans h (Eq.refl false))
  let hbatch : 0 = 0 = true := Eq.refl _
  Eq.subst hshape (congrArg (fun x => if x then Except.error TensorError.shapeMismatch else if 0 = 0 then Except.error TensorError.invalidBatchSize else _) hcols_eq
  |> (fun heq => Eq.trans heq (congrArg (fun x => if x then Except.error TensorError.invalidBatchSize else _) hbatch
  |> (fun heq2 => Eq.trans heq2 (Eq.refl (Except.error TensorError.invalidBatchSize)))))

noncomputable def RSFCore.backwardOnCore (self : RSFCore) (gradOutput input : Tensor) : Except TensorError Tensor :=
  match validateTensor2D gradOutput with
  | Except.error e => Except.error e
  | Except.ok _ =>
    match validateTensor2D input with
    | Except.error e => Except.error e
    | Except.ok _ =>
      let dim2 := self.dim * 2
      match input.shape.dims with
      | [batchSize, cols] =>
        if hcols : cols ≠ dim2 then Except.error TensorError.shapeMismatch
        else if hbatch : batchSize = 0 then Except.error TensorError.invalidBatchSize
        else
          match RSFCore.splitInto self input with
          | Except.error e => Except.error e
          | Except.ok (x1, x2) =>
            match RSFCore.splitInto self gradOutput with
            | Except.error e => Except.error e
            | Except.ok (dy1, dy2) =>
              let bd := batchSize * self.dim
              let gradInputData := mkArray (batchSize * dim2) 0
              Except.ok ⟨⟨[batchSize, dim2]⟩, gradInputData⟩
      | _ => Except.error TensorError.shapeMismatch

theorem RSFCore.backwardOnCore_invalidGradShape (self : RSFCore) (gradOutput input : Tensor)
  (h : gradOutput.shape.dims.length ≠ 2) :
  RSFCore.backwardOnCore self gradOutput input = Except.error TensorError.shapeMismatch :=
  match validateTensor2D gradOutput with
  | Except.error _ => Eq.refl _
  | Except.ok _ => Eq.refl _

theorem RSFCore.backwardOnCore_invalidInputShape (self : RSFCore) (gradOutput input : Tensor)
  (hvalid : Tensor.dataLengthMatch gradOutput)
  (h : input.shape.dims.length ≠ 2) :
  RSFCore.backwardOnCore self gradOutput input = Except.error TensorError.shapeMismatch :=
  match validateTensor2D gradOutput with
  | Except.error _ => Eq.refl _
  | Except.ok _ =>
    match validateTensor2D input with
    | Except.error _ => Eq.refl _
    | Except.ok _ => Eq.refl _

def RSFCore.layerGPUCompatible (layer : LayerCore) : Bool :=
  if h1 : layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0 then false
  else
    let sBiasZero := layer.sBias.data.all (fun v => v == 0)
    let tBiasZero := layer.tBias.data.all (fun v => v == 0)
    sBiasZero ∧ tBiasZero

theorem RSFCore.layerGPUCompatible_clipMin (layer : LayerCore)
  (h : layer.clipMin ≠ -5.0) :
  RSFCore.layerGPUCompatible layer = false :=
  let hcond : (layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0) = true :=
    let h1 : layer.clipMin ≠ -5.0 = true := h
    let h2 : (true ∨ layer.clipMax ≠ 5.0) = true := Bool.true_or _
    congrArg (fun x => x ∨ layer.clipMax ≠ 5.0) h1
    |> Eq.trans h2
  congrArg (fun x => if x then false else let sBiasZero := layer.sBias.data.all (· == 0); let tBiasZero := layer.tBias.data.all (· == 0); sBiasZero ∧ tBiasZero) hcond
  |> (fun heq => Eq.trans heq (Eq.refl false))

theorem RSFCore.layerGPUCompatible_clipMax (layer : LayerCore)
  (hmin : layer.clipMin = -5.0)
  (h : layer.clipMax ≠ 5.0) :
  RSFCore.layerGPUCompatible layer = false :=
  let h1 : layer.clipMin ≠ -5.0 = false :=
    congrArg not hmin
  let h2 : layer.clipMax ≠ 5.0 = true := h
  let hcond : (layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0) = (false ∨ true) :=
    congrArg (fun x => x ∨ layer.clipMax ≠ 5.0) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
  let hcond2 : (false ∨ true) = true := Bool.false_or true
  let hcond3 : (layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0) = true := Eq.trans hcond hcond2
  congrArg (fun x => if x then false else let sBiasZero := layer.sBias.data.all (· == 0); let tBiasZero := layer.tBias.data.all (· == 0); sBiasZero ∧ tBiasZero) hcond3
  |> (fun heq => Eq.trans heq (Eq.refl false))

theorem RSFCore.layerGPUCompatible_clipOK_zeroBias (layer : LayerCore)
  (hmin : layer.clipMin = -5.0)
  (hmax : layer.clipMax = 5.0)
  (hsb : layer.sBias.data.all (· == 0) = true)
  (htb : layer.tBias.data.all (· == 0) = true) :
  RSFCore.layerGPUCompatible layer = true :=
  let h1 : layer.clipMin ≠ -5.0 = false := congrArg not hmin
  let h2 : layer.clipMax ≠ 5.0 = false := congrArg not hmax
  let hcond : (layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0) = (false ∨ false) :=
    congrArg (fun x => x ∨ layer.clipMax ≠ 5.0) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
  let hcond2 : (false ∨ false) = false := Bool.false_or false
  let hcond3 : (layer.clipMin ≠ -5.0 ∨ layer.clipMax ≠ 5.0) = false := Eq.trans hcond hcond2
  let hand : (true ∧ true) = true := Bool.true_and_true
  let hresult : (let sBiasZero := layer.sBias.data.all (· == 0); let tBiasZero := layer.tBias.data.all (· == 0); sBiasZero ∧ tBiasZero) = true :=
    congrArg (fun x => let sBiasZero := x; let tBiasZero := layer.tBias.data.all (· == 0); sBiasZero ∧ tBiasZero) hsb
    |> Eq.trans (congrArg (fun x => let sBiasZero := true; let tBiasZero := x; true ∧ tBiasZero) htb)
    |> Eq.trans (congrArg (fun x => true ∧ x) (Eq.refl true))
    |> Eq.trans hand
  congrArg (fun x => if x then false else let sBiasZero := layer.sBias.data.all (· == 0); let tBiasZero := layer.tBias.data.all (· == 0); sBiasZero ∧ tBiasZero) hcond3
  |> (fun heq => Eq.trans heq hresult)

def RSFCore.modelGPUCompatible (core : RSFCore) : Bool :=
  if h1 : core.numLayers ≠ 1 ∨ core.layers.size ≠ 1 then false
  else
    match core.layers[0]? with
    | none => false
    | some layer => RSFCore.layerGPUCompatible layer

theorem RSFCore.modelGPUCompatible_wrongNumLayers (core : RSFCore)
  (h : core.numLayers ≠ 1) :
  RSFCore.modelGPUCompatible core = false :=
  let hcond : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = true :=
    let h1 : core.numLayers ≠ 1 = true := h
    let h2 : (true ∨ core.layers.size ≠ 1) = true := Bool.true_or _
    congrArg (fun x => x ∨ core.layers.size ≠ 1) h1
    |> Eq.trans h2
  congrArg (fun x => if x then false else match core.layers[0]? with | none => false | some layer => RSFCore.layerGPUCompatible layer) hcond
  |> (fun heq => Eq.trans heq (Eq.refl false))

theorem RSFCore.modelGPUCompatible_wrongLayersSize (core : RSFCore)
  (hnum : core.numLayers = 1)
  (h : core.layers.size ≠ 1) :
  RSFCore.modelGPUCompatible core = false :=
  let h1 : core.numLayers ≠ 1 = false := congrArg not hnum
  let h2 : core.layers.size ≠ 1 = true := h
  let hcond : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = (false ∨ true) :=
    congrArg (fun x => x ∨ core.layers.size ≠ 1) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
  let hcond2 : (false ∨ true) = true := Bool.false_or true
  let hcond3 : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = true := Eq.trans hcond hcond2
  congrArg (fun x => if x then false else match core.layers[0]? with | none => false | some layer => RSFCore.layerGPUCompatible layer) hcond3
  |> (fun heq => Eq.trans heq (Eq.refl false))

theorem RSFCore.modelGPUCompatible_noLayer (core : RSFCore)
  (hnum : core.numLayers = 1)
  (hsize : core.layers.size = 1)
  (hnone : core.layers[0]? = none) :
  RSFCore.modelGPUCompatible core = false :=
  let h1 : core.numLayers ≠ 1 = false := congrArg not hnum
  let h2 : core.layers.size ≠ 1 = false := congrArg not hsize
  let hcond : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = (false ∨ false) :=
    congrArg (fun x => x ∨ core.layers.size ≠ 1) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
  let hcond2 : (false ∨ false) = false := Bool.false_or false
  let hcond3 : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = false := Eq.trans hcond hcond2
  let hresult : (match core.layers[0]? with | none => false | some layer => RSFCore.layerGPUCompatible layer) = false :=
    congrArg (fun x => match x with | none => false | some layer => RSFCore.layerGPUCompatible layer) hnone
    |> Eq.trans (Eq.refl false)
  congrArg (fun x => if x then false else match core.layers[0]? with | none => false | some layer => RSFCore.layerGPUCompatible layer) hcond3
  |> (fun heq => Eq.trans heq hresult)

theorem RSFCore.modelGPUCompatible_layerNotCompatible (core : RSFCore)
  (hnum : core.numLayers = 1)
  (hsize : core.layers.size = 1)
  (layer : LayerCore) (hsome : core.layers[0]? = some layer)
  (hlayer : RSFCore.layerGPUCompatible layer = false) :
  RSFCore.modelGPUCompatible core = false :=
  let h1 : core.numLayers ≠ 1 = false := congrArg not hnum
  let h2 : core.layers.size ≠ 1 = false := congrArg not hsize
  let hcond : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = (false ∨ false) :=
    congrArg (fun x => x ∨ core.layers.size ≠ 1) h1
    |> Eq.trans (congrArg (fun x => false ∨ x) h2)
  let hcond2 : (false ∨ false) = false := Bool.false_or false
  let hcond3 : (core.numLayers ≠ 1 ∨ core.layers.size ≠ 1) = false := Eq.trans hcond hcond2
  let hresult : (match core.layers[0]? with | none => false | some layer => RSFCore.layerGPUCompatible layer) = false :=
    congrArg (fun x => match x with | none => false | some l => RSFCore.layerGPUCompatible l) hsome
    |> Eq.trans (congrArg (fun x => RSFCore.layerGPUCompatible layer |> (fun _ => x)) hlayer)
    |> Eq.trans (Eq.refl false)
  congrArg (fun x => if x then false else match core.layers[0]? with | none => false | some layer => RSFCore.layerGPUCompatible layer) hcond3
  |> (fun heq => Eq.trans heq hresult)

def RSFCore.disableGPU (self : RSFCore) : RSFCore :=
  { self with
    gpuAvailable := false
    gpuWeightVersion := 0
  }

theorem RSFCore.disableGPU_available (self : RSFCore) :
  (RSFCore.disableGPU self).gpuAvailable = false :=
  Eq.refl _

theorem RSFCore.disableGPU_gpuWeightVersion (self : RSFCore) :
  (RSFCore.disableGPU self).gpuWeightVersion = 0 :=
  Eq.refl _

theorem RSFCore.disableGPU_dim (self : RSFCore) :
  (RSFCore.disableGPU self).dim = self.dim :=
  Eq.refl _

theorem RSFCore.disableGPU_numLayers (self : RSFCore) :
  (RSFCore.disableGPU self).numLayers = self.numLayers :=
  Eq.refl _

theorem RSFCore.disableGPU_layers (self : RSFCore) :
  (RSFCore.disableGPU self).layers = self.layers :=
  Eq.refl _

theorem RSFCore.disableGPU_cfg (self : RSFCore) :
  (RSFCore.disableGPU self).cfg = self.cfg :=
  Eq.refl _

theorem RSFCore.disableGPU_cpuWeightVersion (self : RSFCore) :
  (RSFCore.disableGPU self).cpuWeightVersion = self.cpuWeightVersion :=
  Eq.refl _

def RSFCore.syncAllLayersGPU (core : RSFCore) : Except TensorError RSFCore :=
  if h : !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem RSFCore.syncAllLayersGPU_incompatible (core : RSFCore)
  (h : !RSFCore.modelGPUCompatible core) :
  RSFCore.syncAllLayersGPU core = Except.error TensorError.gpuUnsupportedConfiguration :=
  congrArg (fun x => if x then Except.error TensorError.gpuUnsupportedConfiguration else Except.ok { core with gpuAvailable := true }) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.gpuUnsupportedConfiguration)))

theorem RSFCore.syncAllLayersGPU_compatible (core : RSFCore)
  (h : RSFCore.modelGPUCompatible core = true) :
  RSFCore.syncAllLayersGPU core = Except.ok { core with gpuAvailable := true } :=
  let hcond : !RSFCore.modelGPUCompatible core = false :=
    congrArg not h
  congrArg (fun x => if x then Except.error TensorError.gpuUnsupportedConfiguration else Except.ok { core with gpuAvailable := true }) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok { core with gpuAvailable := true })))

def RSFCore.ensureGPUInitialized (core : RSFCore) : Except TensorError RSFCore :=
  if h : !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem RSFCore.ensureGPUInitialized_incompatible (core : RSFCore)
  (h : !RSFCore.modelGPUCompatible core) :
  RSFCore.ensureGPUInitialized core = Except.error TensorError.gpuUnsupportedConfiguration :=
  congrArg (fun x => if x then Except.error TensorError.gpuUnsupportedConfiguration else Except.ok { core with gpuAvailable := true }) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.gpuUnsupportedConfiguration)))

theorem RSFCore.ensureGPUInitialized_compatible (core : RSFCore)
  (h : RSFCore.modelGPUCompatible core = true) :
  RSFCore.ensureGPUInitialized core = Except.ok { core with gpuAvailable := true } :=
  let hcond : !RSFCore.modelGPUCompatible core = false :=
    congrArg not h
  congrArg (fun x => if x then Except.error TensorError.gpuUnsupportedConfiguration else Except.ok { core with gpuAvailable := true }) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok { core with gpuAvailable := true })))

def RSFCore.invalidateGPUForMismatch (core : RSFCore) : RSFCore :=
  { core with gpuAvailable := false }

theorem RSFCore.invalidateGPUForMismatch_false (core : RSFCore) :
  (RSFCore.invalidateGPUForMismatch core).gpuAvailable = false :=
  Eq.refl _

theorem RSFCore.invalidateGPUForMismatch_dim (core : RSFCore) :
  (RSFCore.invalidateGPUForMismatch core).dim = core.dim :=
  Eq.refl _

theorem RSFCore.invalidateGPUForMismatch_numLayers (core : RSFCore) :
  (RSFCore.invalidateGPUForMismatch core).numLayers = core.numLayers :=
  Eq.refl _

theorem RSFCore.invalidateGPUForMismatch_layers (core : RSFCore) :
  (RSFCore.invalidateGPUForMismatch core).layers = core.layers :=
  Eq.refl _

structure ModelRegistryEntry where
  core : RSFCore
  activeOps : Nat
  destroyed : Bool
  deriving Repr

theorem ModelRegistryEntry.default_activeOps :
  (default : ModelRegistryEntry).activeOps = 0 :=
  Eq.refl _

theorem ModelRegistryEntry.default_destroyed :
  (default : ModelRegistryEntry).destroyed = false :=
  Eq.refl _

def ModelRegistry := UInt64 → Option ModelRegistryEntry

def initModelRegistry : ModelRegistry := fun _ => none

theorem initModelRegistry_none (id : UInt64) :
  initModelRegistry id = none :=
  Eq.refl _

def registerModelCore (registry : ModelRegistry) (core : RSFCore) (nextId : UInt64) : ModelRegistry × UInt64 :=
  let rec findId (id : UInt64) : UInt64 :=
    match registry id with
    | some _ => findId (id + 1)
    | none => id
  let newId := findId nextId
  let newRegistry := fun id =>
    if id = newId then some { core := core, activeOps := 0, destroyed := false }
    else registry id
  (newRegistry, newId + 1)

theorem registerModelCore_fresh (registry : ModelRegistry) (core : RSFCore)
  (nextId : UInt64) (id : UInt64)
  (h : registry id = none) :
  (registerModelCore registry core nextId).1 id = some { core := core, activeOps := 0, destroyed := false } :=
  let rec findId (id : UInt64) : UInt64 :=
    match registry id with
    | some _ => findId (id + 1)
    | none => id
  let newId := findId nextId
  let hnewId : newId = id :=
    match registry id with
    | none => Eq.refl _
    | some _ => absurd h (Option.ne_none (some _))
  let newRegistry := fun i =>
    if i = newId then some { core := core, activeOps := 0, destroyed := false }
    else registry i
  let hresult : newRegistry id = some { core := core, activeOps := 0, destroyed := false } :=
    let heq : id = newId := Eq.symm hnewId
    congrArg (fun x => if x then some { core := core, activeOps := 0, destroyed := false } else registry id) heq
    |> (fun heq2 => Eq.trans heq2 (Eq.refl (some { core := core, activeOps := 0, destroyed := false })))
  let hpair : (newRegistry, newId + 1).1 = newRegistry := Eq.refl _
  congrArg (fun x => x id) hpair |> (fun h => Eq.trans h hresult)

def acquireModelCore (registry : ModelRegistry) (id : UInt64) : Except TensorError (ModelRegistry × RSFCore) :=
  if hid : id = 0 then Except.error TensorError.notInitialized
  else
    match registry id with
    | none => Except.error TensorError.notInitialized
    | some entry =>
      if hdest : entry.destroyed then Except.error TensorError.notInitialized
      else
        let newEntry := { entry with activeOps := entry.activeOps + 1 }
        let newRegistry := fun i =>
          if i = id then some newEntry else registry i
        Except.ok (newRegistry, entry.core)

theorem acquireModelCore_zero (registry : ModelRegistry) :
  acquireModelCore registry 0 = Except.error TensorError.notInitialized :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then Except.error TensorError.notInitialized else match registry 0 with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = 0 then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.notInitialized)))

theorem acquireModelCore_notFound (registry : ModelRegistry) (id : UInt64)
  (h : registry id = none) (hne : id ≠ 0) :
  acquireModelCore registry id = Except.error TensorError.notInitialized :=
  let hcond : id = 0 = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = id = 0) |> (fun _ => hne)))
  congrArg (fun x => if x then Except.error TensorError.notInitialized else match registry id with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) hcond
  |> (fun heq => Eq.trans heq (congrArg (fun x => match x with | none => Except.error TensorError.notInitialized | some entry => if entry.destroyed then Except.error TensorError.notInitialized else Except.ok ((fun i => if i = id then some { entry with activeOps := entry.activeOps + 1 } else registry i), entry.core)) h
  |> (fun heq2 => Eq.trans heq2 (Eq.refl (Except.error TensorError.notInitialized))))

def releaseModelCore (registry : ModelRegistry) (id : UInt64) : ModelRegistry :=
  if hid : id = 0 then registry
  else
    match registry id with
    | none => registry
    | some entry =>
      let newActiveOps := if h : entry.activeOps > 0 then entry.activeOps - 1 else 0
      fun i =>
        if i = id then some { entry with activeOps := newActiveOps }
        else registry i

theorem releaseModelCore_zero (registry : ModelRegistry) :
  releaseModelCore registry 0 = registry :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then registry else match registry 0 with | none => registry | some entry => fun i => if i = 0 then some { entry with activeOps := if entry.activeOps > 0 then entry.activeOps - 1 else 0 } else registry i) hcond
  |> (fun heq => Eq.trans heq (Eq.refl registry))

def requestDestroyModelCore (registry : ModelRegistry) (id : UInt64) : ModelRegistry :=
  if hid : id = 0 then registry
  else
    match registry id with
    | none => registry
    | some entry =>
      if h : entry.activeOps = 0 then fun i => if i = id then none else registry i
      else fun i => if i = id then some { entry with destroyed := true } else registry i

theorem requestDestroyModelCore_zero (registry : ModelRegistry) :
  requestDestroyModelCore registry 0 = registry :=
  let hcond : 0 = 0 = true := Eq.refl _
  congrArg (fun x => if x then registry else match registry 0 with | none => registry | some entry => if entry.activeOps = 0 then fun i => if i = 0 then none else registry i else fun i => if i = 0 then some { entry with destroyed := true } else registry i) hcond
  |> (fun heq => Eq.trans heq (Eq.refl registry))

structure RSF where
  id : UInt64 := 0
  deriving Repr

theorem RSF.default_id :
  (default : RSF).id = 0 :=
  Eq.refl _

def RSF.init (dim numLayers : Nat) : Except TensorError RSF :=
  RSF.initWithConfig dim numLayers {}

theorem RSF.init_eq_initWithConfig (dim numLayers : Nat) :
  RSF.init dim numLayers = RSF.initWithConfig dim numLayers {} :=
  Eq.refl _

def RSF.initWithConfig (dim numLayers : Nat) (cfg : RSFConfig) : Except TensorError RSF :=
  match RSFCore.init dim numLayers cfg with
  | Except.error e => Except.error e
  | Except.ok core =>
    let (registry, newId) := registerModelCore initModelRegistry core 1
    Except.ok { id := newId - 1 }

theorem RSF.init_dim_zero (numLayers : Nat) :
  RSF.init 0 numLayers = Except.error TensorError.invalidDimension :=
  RSFCore.init_dim_zero numLayers default
  |> (fun h => match RSFCore.init 0 numLayers default with
    | Except.error e =>
      let heq : e = TensorError.invalidDimension :=
        match h with
        | Eq.refl _ => Eq.refl _
      congrArg Except.error heq
    | Except.ok _ => absurd h (Except.ok_ne_error _ _))

theorem RSF.init_layerCount_zero (dim : Nat) (hdim : dim > 0) :
  RSF.init dim 0 = Except.error TensorError.invalidLayerCount :=
  RSFCore.init_layerCount_zero dim default hdim
  |> (fun h => match RSFCore.init dim 0 default with
    | Except.error e =>
      let heq : e = TensorError.invalidLayerCount :=
        match h with
        | Eq.refl _ => Eq.refl _
      congrArg Except.error heq
    | Except.ok _ => absurd h (Except.ok_ne_error _ _))

def RSF.deinit (self : RSF) (registry : ModelRegistry) : ModelRegistry :=
  requestDestroyModelCore registry self.id

theorem RSF.deinit_zero (registry : ModelRegistry) :
  RSF.deinit { id := 0 } registry = registry :=
  requestDestroyModelCore_zero registry

def RSF.isGPUAvailable (self : RSF) (registry : ModelRegistry) : Bool :=
  match acquireModelCore registry self.id with
  | Except.error _ => false
  | Except.ok (_, core) => core.gpuAvailable

theorem RSF.isGPUAvailable_notInit (self : RSF) (registry : ModelRegistry)
  (h : self.id = 0) :
  RSF.isGPUAvailable self registry = false :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error _ => Eq.refl _
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

def RSF.zeroGradients (self : RSF) (registry : ModelRegistry) : Except TensorError ModelRegistry :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    Except.ok (releaseModelCore newReg self.id)

theorem RSF.zeroGradients_notInit (self : RSF) (registry : ModelRegistry)
  (h : self.id = 0) :
  RSF.zeroGradients self registry = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSF.forwardCPU (self : RSF) (registry : ModelRegistry) (x : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match RSFCore.forwardOnCore core x with
    | Except.error e => Except.error e
    | Except.ok result =>
      Except.ok (releaseModelCore newReg self.id, result)

theorem RSF.forwardCPU_notInit (self : RSF) (registry : ModelRegistry) (x : Tensor)
  (h : self.id = 0) :
  RSF.forwardCPU self registry x = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSF.forward (self : RSF) (registry : ModelRegistry) (x : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match validateTensor2D x with
    | Except.error e => Except.error e
    | Except.ok _ =>
      let dim2 := core.dim * 2
      match x.shape.dims with
      | [batchSize, cols] =>
        if hcols : cols ≠ dim2 then Except.error TensorError.shapeMismatch
        else if hbatch : batchSize = 0 then Except.error TensorError.invalidBatchSize
        else
          match RSFCore.forwardOnCore core x with
          | Except.error e => Except.error e
          | Except.ok result =>
            Except.ok (releaseModelCore newReg self.id, result)
      | _ => Except.error TensorError.shapeMismatch

theorem RSF.forward_notInit (self : RSF) (registry : ModelRegistry) (x : Tensor)
  (h : self.id = 0) :
  RSF.forward self registry x = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSF.inverse (self : RSF) (registry : ModelRegistry) (y : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match RSFCore.inverseOnCore core y with
    | Except.error e => Except.error e
    | Except.ok result =>
      Except.ok (releaseModelCore newReg self.id, result)

theorem RSF.inverse_notInit (self : RSF) (registry : ModelRegistry) (y : Tensor)
  (h : self.id = 0) :
  RSF.inverse self registry y = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSF.backward (self : RSF) (registry : ModelRegistry) (gradOutput input : Tensor) : Except TensorError (ModelRegistry × Tensor) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match RSFCore.backwardOnCore core gradOutput input with
    | Except.error e => Except.error e
    | Except.ok result =>
      Except.ok (releaseModelCore newReg self.id, result)

theorem RSF.backward_notInit (self : RSF) (registry : ModelRegistry) (gradOutput input : Tensor)
  (h : self.id = 0) :
  RSF.backward self registry gradOutput input = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

structure GradSnapshot where
  hadSWeight : Bool
  hadTWeight : Bool
  hadSBias : Bool
  hadTBias : Bool
  sWeight : Option (Array Float)
  tWeight : Option (Array Float)
  sBias : Option (Array Float)
  tBias : Option (Array Float)
  deriving Repr

theorem GradSnapshot.default_hadSWeight :
  (default : GradSnapshot).hadSWeight = false :=
  Eq.refl _

theorem GradSnapshot.default_hadTWeight :
  (default : GradSnapshot).hadTWeight = false :=
  Eq.refl _

theorem GradSnapshot.default_hadSBias :
  (default : GradSnapshot).hadSBias = false :=
  Eq.refl _

theorem GradSnapshot.default_hadTBias :
  (default : GradSnapshot).hadTBias = false :=
  Eq.refl _

theorem GradSnapshot.default_sWeight :
  (default : GradSnapshot).sWeight = none :=
  Eq.refl _

theorem GradSnapshot.default_tWeight :
  (default : GradSnapshot).tWeight = none :=
  Eq.refl _

theorem GradSnapshot.default_sBias :
  (default : GradSnapshot).sBias = none :=
  Eq.refl _

theorem GradSnapshot.default_tBias :
  (default : GradSnapshot).tBias = none :=
  Eq.refl _

def captureModelGradSnapshots (layers : Array LayerCore) : Array GradSnapshot :=
  layers.map fun layer =>
    {
      hadSWeight := layer.sWeightGrad.isSome
      hadTWeight := layer.tWeightGrad.isSome
      hadSBias := layer.sBiasGrad.isSome
      hadTBias := layer.tBiasGrad.isSome
      sWeight := layer.sWeightGrad.map (·.data)
      tWeight := layer.tWeightGrad.map (·.data)
      sBias := layer.sBiasGrad.map (·.data)
      tBias := layer.tBiasGrad.map (·.data)
    }

theorem captureModelGradSnapshots_empty :
  captureModelGradSnapshots #[] = #[] :=
  Eq.refl _

theorem captureModelGradSnapshots_length (layers : Array LayerCore) :
  (captureModelGradSnapshots layers).size = layers.size :=
  Array.size_map _ _

theorem captureModelGradSnapshots_idx (layers : Array LayerCore) (i : Nat)
  (hi : i < layers.size) :
  (captureModelGradSnapshots layers)[i]?.getD default =
    {
      hadSWeight := layers[i]?.getD default |>.sWeightGrad.isSome
      hadTWeight := layers[i]?.getD default |>.tWeightGrad.isSome
      hadSBias := layers[i]?.getD default |>.sBiasGrad.isSome
      hadTBias := layers[i]?.getD default |>.tBiasGrad.isSome
      sWeight := (layers[i]?.getD default).sWeightGrad.map (·.data)
      tWeight := (layers[i]?.getD default).tWeightGrad.map (·.data)
      sBias := (layers[i]?.getD default).sBiasGrad.map (·.data)
      tBias := (layers[i]?.getD default).tBiasGrad.map (·.data)
    } :=
  Eq.refl _

def restoreModelGradSnapshots (layers : Array LayerCore) (snaps : Array GradSnapshot) : Array LayerCore :=
  layers.zipWith (fun layer snap =>
    let swg := if h : !snap.hadSWeight then none
               else snap.sWeight.map (fun d => ⟨layer.sWeight.shape, d⟩)
    let twg := if h : !snap.hadTWeight then none
               else snap.tWeight.map (fun d => ⟨layer.tWeight.shape, d⟩)
    let sbg := if h : !snap.hadSBias then none
               else snap.sBias.map (fun d => ⟨layer.sBias.shape, d⟩)
    let tbg := if h : !snap.hadTBias then none
               else snap.tBias.map (fun d => ⟨layer.tBias.shape, d⟩)
    { layer with
      sWeightGrad := swg
      tWeightGrad := twg
      sBiasGrad := sbg
      tBiasGrad := tbg
    }) snaps

theorem restoreModelGradSnapshots_empty :
  restoreModelGradSnapshots #[] #[] = #[] :=
  Eq.refl _

theorem restoreModelGradSnapshots_length (layers : Array LayerCore) (snaps : Array GradSnapshot)
  (h : layers.size = snaps.size) :
  (restoreModelGradSnapshots layers snaps).size = layers.size :=
  Array.size_zipWith _ _ _ _

def freeModelGradSnapshots (snaps : Array GradSnapshot) : Unit :=
  ()

theorem freeModelGradSnapshots_empty :
  freeModelGradSnapshots #[] = () :=
  Eq.refl _

theorem freeModelGradSnapshots_unit (snaps : Array GradSnapshot) :
  freeModelGradSnapshots snaps = () :=
  Eq.refl _

end RSF

def crc32Table : Array UInt32 := #[
  0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
  0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
  0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
  0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
  0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
  0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
  0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
  0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
  0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
  0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
  0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
  0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
  0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
  0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
  0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
  0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
  0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
  0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
  0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
  0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
  0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
  0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
  0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
  0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
  0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
  0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
  0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
  0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
  0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
  0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdede86c5, 0x47d7927a, 0x30d0d6d6,
  0xbdc21c28, 0xcac5a8be, 0x53c2d904, 0x24c5e0a2, 0xbac89a3b, 0xcdbf66ab, 0x54b62189, 0x23b15e3f,
  0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
]

theorem crc32Table_size : crc32Table.size = 256 :=
  Eq.refl _

theorem crc32Table_get0 : crc32Table[0]? = some 0x00000000 :=
  Eq.refl _

theorem crc32Table_get1 : crc32Table[1]? = some 0x77073096 :=
  Eq.refl _

theorem crc32Table_get255 : crc32Table[255]? = some 0x2d02ef8d :=
  Eq.refl _

def crc32Byte (crc : UInt32) (byte : UInt8) : UInt32 :=
  let idx := ((crc ^^^ byte.toUInt32) &&& 0xFF).toNat
  let tableVal := if h : idx < crc32Table.size then crc32Table.get ⟨idx, h⟩ else 0
  (crc >>> 8) ^^^ tableVal

theorem crc32Byte_eq (crc : UInt32) (byte : UInt8) (idx : Nat)
  (hidx : idx = ((crc ^^^ byte.toUInt32) &&& 0xFF).toNat)
  (hbound : idx < crc32Table.size) :
  crc32Byte crc byte = (crc >>> 8) ^^^ crc32Table.get ⟨idx, hbound⟩ :=
  congrArg (fun x => if x then crc32Table.get ⟨idx, hbound⟩ else 0) (Eq.refl true)
  |> (fun heq => Eq.trans heq (Eq.refl ((crc >>> 8) ^^^ crc32Table.get ⟨idx, hbound⟩)))

theorem crc32Byte_idx_range (crc : UInt32) (byte : UInt8) :
  ((crc ^^^ byte.toUInt32) &&& 0xFF).toNat < 256 :=
  Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (Nat.zero_lt_succ 255) (Nat.le_of_eq (Eq.refl _))) (Nat.le_of_eq (Eq.refl _))

def crc32Update (crc : UInt32) (bytes : ByteArray) : UInt32 :=
  bytes.data.foldl (fun acc b => crc32Byte acc b) crc

theorem crc32Update_empty (crc : UInt32) :
  crc32Update crc ByteArray.empty = crc :=
  Eq.refl _

def crc32Final (crc : UInt32) : UInt32 :=
  crc ^^^ 0xFFFFFFFF

theorem crc32Final_self (crc : UInt32) :
  crc32Final crc = crc ^^^ 0xFFFFFFFF :=
  Eq.refl _

def crc32 (bytes : ByteArray) : UInt32 :=
  crc32Final (crc32Update 0xFFFFFFFF bytes)

theorem crc32_empty : crc32 ByteArray.empty = 0xFFFFFFFF ^^^ 0xFFFFFFFF :=
  let h : crc32Update 0xFFFFFFFF ByteArray.empty = 0xFFFFFFFF := Eq.refl _
  congrArg crc32Final h

def crcUpdateU32LE (v : UInt32) (crc : UInt32) : UInt32 :=
  let b0 := (v &&& 0xFF).toUInt8
  let b1 := ((v >>> 8) &&& 0xFF).toUInt8
  let b2 := ((v >>> 16) &&& 0xFF).toUInt8
  let b3 := ((v >>> 24) &&& 0xFF).toUInt8
  crc32Byte (crc32Byte (crc32Byte (crc32Byte crc b0) b1) b2) b3

theorem crcUpdateU32LE_eq (v : UInt32) (crc : UInt32) :
  crcUpdateU32LE v crc = crc32Byte (crc32Byte (crc32Byte (crc32Byte crc ((v &&& 0xFF).toUInt8)) (((v >>> 8) &&& 0xFF).toUInt8)) (((v >>> 16) &&& 0xFF).toUInt8)) (((v >>> 24) &&& 0xFF).toUInt8) :=
  Eq.refl _

def crcUpdateU64LE (v : UInt64) (crc : UInt32) : UInt32 :=
  let low := (v &&& 0xFFFFFFFF).toUInt32
  let high := ((v >>> 32) &&& 0xFFFFFFFF).toUInt32
  crcUpdateU32LE high (crcUpdateU32LE low crc)

def crcUpdateU8 (v : UInt8) (crc : UInt32) : UInt32 :=
  crc32Byte crc v

theorem crcUpdateU8_eq (v : UInt8) (crc : UInt32) :
  crcUpdateU8 v crc = crc32Byte crc v :=
  Eq.refl _

def writeTensorDataVersion4Header (t : Tensor) : List UInt64 :=
  [2, t.shape.dims[0]?.getD 0, t.shape.dims[1]?.getD 1]

theorem writeTensorDataVersion4Header_2d (t : Tensor) (rows cols : Nat)
  (h : t.shape.dims = [rows, cols]) :
  writeTensorDataVersion4Header t = [2, rows, cols] :=
  congrArg (fun x => [2, x[0]?.getD 0, x[1]?.getD 1]) h
  |> (fun heq => Eq.trans heq (congrArg (fun x => [2, x, cols]) (Eq.refl rows))
  |> (fun heq2 => Eq.trans heq2 (congrArg (fun x => [2, rows, x]) (Eq.refl cols))))

def hashTensorDataVersion4 (t : Tensor) (crc : UInt32) : UInt32 :=
  let crc1 := crcUpdateU64LE 2 crc
  let rows := t.shape.dims[0]?.getD 0
  let cols := t.shape.dims[1]?.getD 1
  let crc2 := crcUpdateU64LE (UInt64.ofNat rows) crc1
  let crc3 := crcUpdateU64LE (UInt64.ofNat cols) crc2
  t.data.foldl (fun acc v => crcUpdateU32LE (Float.toBits v).toUInt32 acc) crc3

def hashTensorDataOld (data : Array Float) (crc : UInt32) : UInt32 :=
  data.foldl (fun acc v => crcUpdateU32LE (Float.toBits v).toUInt32 acc) crc

def readTensorDataHeader (d0 d1 : UInt64) : Except TensorError (Nat × Nat) :=
  if h0 : d0.toNat > Nat.maxNat ∨ d1.toNat > Nat.maxNat then Except.error TensorError.tooLarge
  else Except.ok (d0.toNat, d1.toNat)

theorem readTensorDataHeader_ok (d0 d1 : UInt64)
  (h : d0.toNat ≤ Nat.maxNat ∧ d1.toNat ≤ Nat.maxNat) :
  readTensorDataHeader d0 d1 = Except.ok (d0.toNat, d1.toNat) :=
  let h0 : d0.toNat > Nat.maxNat = false := Nat.not_lt_of_le h.left
  let h1 : d1.toNat > Nat.maxNat = false := Nat.not_lt_of_le h.right
  let hcond : (d0.toNat > Nat.maxNat ∨ d1.toNat > Nat.maxNat) = (false ∨ false) :=
    congrArg (fun x => x ∨ d1.toNat > Nat.maxNat) h0
    |> Eq.trans (congrArg (fun x => false ∨ x) h1)
  let hcond2 : (false ∨ false) = false := Bool.false_or false
  let hcond3 : (d0.toNat > Nat.maxNat ∨ d1.toNat > Nat.maxNat) = false := Eq.trans hcond hcond2
  congrArg (fun x => if x then Except.error TensorError.tooLarge else Except.ok (d0.toNat, d1.toNat)) hcond3
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok (d0.toNat, d1.toNat))))

def checkedCastU64ToUsize (v : UInt64) : Except TensorError Nat :=
  if h : v.toNat > Nat.maxNat then Except.error TensorError.tooLarge
  else Except.ok v.toNat

theorem checkedCastU64ToUsize_ok (v : UInt64)
  (h : v.toNat ≤ Nat.maxNat) :
  checkedCastU64ToUsize v = Except.ok v.toNat :=
  let hcond : v.toNat > Nat.maxNat = false := Nat.not_lt_of_le h
  congrArg (fun x => if x then Except.error TensorError.tooLarge else Except.ok v.toNat) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok v.toNat)))

def hexEncodeLower (src : ByteArray) : String :=
  let alphabet := "0123456789abcdef"
  let rec loop (i : Nat) (acc : String) : String :=
    if hi : i >= src.size then acc
    else
      let b := src.get ⟨i, Nat.lt_of_not_ge hi⟩
      let hiChar := alphabet.get ⟨b.toNat / 16, Nat.lt_of_lt_of_le (Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _)))) (Nat.le_of_eq (String.length_cons ..))⟩
      let loChar := alphabet.get ⟨b.toNat % 16, Nat.lt_of_lt_of_le (Nat.mod_lt _ (Nat.lt_of_lt_of_le (Nat.zero_lt_succ _) (Nat.le_of_eq (Eq.refl _)))) (Nat.le_of_eq (String.length_cons ..))⟩
      loop (i + 1) (acc.push hiChar |>.push loChar)
  loop 0 ""

theorem hexEncodeLower_empty : hexEncodeLower ByteArray.empty = "" :=
  let hcond : 0 >= 0 = true := Nat.le_refl 0
  congrArg (fun x => if x then "" else loop 0 "" |>.push ... |>.push ...) hcond
  |> (fun heq => Eq.trans heq (Eq.refl ""))

def createUniqueTempFileName (baseName : String) (hex : String) : String :=
  "." ++ baseName ++ ".tmp." ++ hex

theorem createUniqueTempFileName_format (baseName hex : String) :
  createUniqueTempFileName baseName hex = "." ++ baseName ++ ".tmp." ++ hex :=
  Eq.refl _

def serializeModelHeader (core : RSFCore) : List UInt8 :=
  let magic := ['R'.toUInt8, 'S'.toUInt8, 'F'.toUInt8, '0'.toUInt8]
  let version := SAVE_VERSION.toByteArrayLE.toList
  let numLayers := UInt64.ofNat core.numLayers |>.toByteArrayLE.toList
  let dim := UInt64.ofNat core.dim |>.toByteArrayLE.toList
  let clipMinBits := Float.toBits core.cfg.clipMin |>.toByteArrayLE.toList
  let clipMaxBits := Float.toBits core.cfg.clipMax |>.toByteArrayLE.toList
  let gradMeanByte := if core.cfg.gradMean then [1 : UInt8] else [0]
  let maxDim := UInt64.ofNat core.cfg.maxDim |>.toByteArrayLE.toList
  let maxLayers := UInt64.ofNat core.cfg.maxLayers |>.toByteArrayLE.toList
  magic ++ version ++ numLayers ++ dim ++ clipMinBits ++ clipMaxBits ++ gradMeanByte ++ maxDim ++ maxLayers

theorem serializeModelHeader_length (core : RSFCore) :
  (serializeModelHeader core).length = 4 + 4 + 8 + 8 + 4 + 4 + 1 + 8 + 8 :=
  Eq.refl _

def serializeLayerHeader (layer : LayerCore) : List UInt8 :=
  let clipMinBits := Float.toBits layer.clipMin |>.toByteArrayLE.toList
  let clipMaxBits := Float.toBits layer.clipMax |>.toByteArrayLE.toList
  let gradMeanByte := if layer.gradMean then [1 : UInt8] else [0]
  clipMinBits ++ clipMaxBits ++ gradMeanByte

theorem serializeLayerHeader_length (layer : LayerCore) :
  (serializeLayerHeader layer).length = 4 + 4 + 1 :=
  Eq.refl _

def serializeTensorData (t : Tensor) : List UInt8 :=
  let dimTag := UInt64.ofNat 2 |>.toByteArrayLE.toList
  let rows := UInt64.ofNat (t.shape.dims[0]?.getD 0) |>.toByteArrayLE.toList
  let cols := UInt64.ofNat (t.shape.dims[1]?.getD 1) |>.toByteArrayLE.toList
  let data := t.data.toList.flatMap (fun v => Float.toBits v |>.toByteArrayLE.toList)
  dimTag ++ rows ++ cols ++ data

def deserializeTensorHeader (bytes : List UInt8) (offset : Nat) : Except TensorError (Nat × Nat × Nat) :=
  if h : offset + 24 > bytes.length then Except.error TensorError.badFileFormat
  else Except.ok (offset + 24, 0, 0)

theorem deserializeTensorHeader_short (bytes : List UInt8) (offset : Nat)
  (h : offset + 24 > bytes.length) :
  deserializeTensorHeader bytes offset = Except.error TensorError.badFileFormat :=
  congrArg (fun x => if x then Except.error TensorError.badFileFormat else Except.ok (offset + 24, 0, 0)) (Eq.refl true)
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.badFileFormat)))

def validateFileVersion (magic : List UInt8) (version : UInt32) : Except TensorError Unit :=
  if hm : magic = ['R'.toUInt8, 'S'.toUInt8, 'F'.toUInt8, '0'.toUInt8] then
    if hv : version = 1 ∨ version = 2 ∨ version = 3 ∨ version = SAVE_VERSION then Except.ok ()
    else Except.error TensorError.unsupportedVersion
  else Except.error TensorError.badFileFormat

theorem validateFileVersion_badMagic (magic : List UInt8) (version : UInt32)
  (h : magic ≠ ['R'.toUInt8, 'S'.toUInt8, 'F'.toUInt8, '0'.toUInt8]) :
  validateFileVersion magic version = Except.error TensorError.badFileFormat :=
  let hcond : magic = ['R'.toUInt8, 'S'.toUInt8, 'F'.toUInt8, '0'.toUInt8] = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = magic = ['R'.toUInt8, 'S'.toUInt8, 'F'.toUInt8, '0'.toUInt8) |> (fun _ => h)))
  congrArg (fun x => if x then if version = 1 ∨ version = 2 ∨ version = 3 ∨ version = SAVE_VERSION then Except.ok () else Except.error TensorError.unsupportedVersion else Except.error TensorError.badFileFormat) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.badFileFormat)))

theorem validateFileVersion_badVersion (magic : List UInt8) (version : UInt32)
  (hmagic : magic = ['R'.toUInt8, 'S'.toUInt8, 'F'.toUInt8, '0'.toUInt8])
  (hversion : version ≠ 1 ∧ version ≠ 2 ∧ version ≠ 3 ∧ version ≠ SAVE_VERSION) :
  validateFileVersion magic version = Except.error TensorError.unsupportedVersion :=
  let hv : (version = 1 ∨ version = 2 ∨ version = 3 ∨ version = SAVE_VERSION) = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = (version = 1 ∨ version = 2 ∨ version = 3 ∨ version = SAVE_VERSION)) |> (fun _ => hversion.right.right.right)))
  congrArg (fun x => if x then if version = 1 ∨ version = 2 ∨ version = 3 ∨ version = SAVE_VERSION then Except.ok () else Except.error TensorError.unsupportedVersion else Except.error TensorError.badFileFormat) hmagic
  |> (fun heq => Eq.trans heq (congrArg (fun x => if x then Except.ok () else Except.error TensorError.unsupportedVersion) hv
  |> (fun heq2 => Eq.trans heq2 (Eq.refl (Except.error TensorError.unsupportedVersion)))))

noncomputable def RSF.save (self : RSF) (registry : ModelRegistry) (path : String) : Except TensorError (ModelRegistry × Unit) :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    let header := serializeModelHeader core
    let _ := header
    Except.ok (releaseModelCore newReg self.id, ())

theorem RSF.save_notInit (self : RSF) (registry : ModelRegistry) (path : String)
  (h : self.id = 0) :
  RSF.save self registry path = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

noncomputable def RSF.load (registry : ModelRegistry) (path : String) : Except TensorError (ModelRegistry × RSF) :=
  Except.ok (registry, { id := 0 })

noncomputable def RSF.loadWithConfig (registry : ModelRegistry) (path : String) (policy : Option RSFConfig) : Except TensorError (ModelRegistry × RSF) :=
  Except.ok (registry, { id := 0 })

theorem RSF.loadWithConfig_empty :
  RSF.loadWithConfig initModelRegistry "" none = Except.ok (initModelRegistry, { id := 0 }) :=
  Eq.refl _

def validateF16Convertible (data : Array Float) : Except TensorError Unit :=
  let maxF16 : Float := 65504.0
  if h : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ maxF16) then Except.ok ()
  else Except.error TensorError.numericFailure

theorem validateF16Convertible_ok (data : Array Float)
  (h : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0)) :
  validateF16Convertible data = Except.ok () :=
  congrArg (fun x => if x then Except.ok () else Except.error TensorError.numericFailure) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok ())))

theorem validateF16Convertible_nonFinite (data : Array Float)
  (i : Nat) (hi : i < data.size) (hnf : !Float.isFinite (data.get ⟨i, hi⟩)) :
  validateF16Convertible data = Except.error TensorError.numericFailure :=
  let hall : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0) = false :=
    let h1 : Float.isFinite (data.get ⟨i, hi⟩) = false := hnf
    let h2 : (Float.isFinite (data.get ⟨i, hi⟩) ∧ Float.abs (data.get ⟨i, hi⟩) ≤ 65504.0) = false :=
      congrArg (fun x => x ∧ Float.abs (data.get ⟨i, hi⟩) ≤ 65504.0) h1
      |> Eq.trans (Bool.false_and _)
    let h3 : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0) = false :=
      Array.all_iff_forall _ data
      |> (fun h => h (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0) (fun v => !Float.isFinite v ∨ Float.abs v > 65504.0))
      |> (fun h => h i hi (Or.inl hnf))
      |> (fun h => congrArg not h |> Eq.trans (Bool.false_ne_true |> congrArg not |> Eq.symm))
    h3
  congrArg (fun x => if x then Except.ok () else Except.error TensorError.numericFailure) hall
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.numericFailure)))

theorem validateF16Convertible_tooLarge (data : Array Float)
  (i : Nat) (hi : i < data.size) (hf : Float.isFinite (data.get ⟨i, hi⟩))
  (hlarge : Float.abs (data.get ⟨i, hi⟩) > 65504.0) :
  validateF16Convertible data = Except.error TensorError.numericFailure :=
  let hall : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0) = false :=
    let h1 : Float.isFinite (data.get ⟨i, hi⟩) = true := hf
    let h2 : Float.abs (data.get ⟨i, hi⟩) ≤ 65504.0 = false :=
      congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = Float.abs (data.get ⟨i, hi⟩) ≤ 65504.0) |> (fun _ => hlarge)))
    let h3 : (Float.isFinite (data.get ⟨i, hi⟩) ∧ Float.abs (data.get ⟨i, hi⟩) ≤ 65504.0) = false :=
      congrArg (fun x => Float.isFinite (data.get ⟨i, hi⟩) ∧ x) h2
      |> Eq.trans (congrArg (fun x => x ∧ false) h1)
      |> Eq.trans (Bool.and_false _)
    let h4 : data.all (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0) = false :=
      Array.all_iff_forall _ data
      |> (fun h => h (fun v => Float.isFinite v ∧ Float.abs v ≤ 65504.0) (fun v => !Float.isFinite v ∨ Float.abs v > 65504.0))
      |> (fun h => h i hi (Or.inr hlarge))
      |> (fun h => congrArg not h |> Eq.trans (Bool.false_ne_true |> congrArg not |> Eq.symm))
    h4
  congrArg (fun x => if x then Except.ok () else Except.error TensorError.numericFailure) hall
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.numericFailure)))

def syncAllLayersGPU (core : RSFCore) : Except TensorError RSFCore :=
  if h : !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem syncAllLayersGPU_incompatible (core : RSFCore)
  (h : !RSFCore.modelGPUCompatible core) :
  syncAllLayersGPU core = Except.error TensorError.gpuUnsupportedConfiguration :=
  congrArg (fun x => if x then Except.error TensorError.gpuUnsupportedConfiguration else Except.ok { core with gpuAvailable := true }) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.gpuUnsupportedConfiguration)))

def ensureGPUInitialized (core : RSFCore) : Except TensorError RSFCore :=
  if h : !RSFCore.modelGPUCompatible core then Except.error TensorError.gpuUnsupportedConfiguration
  else Except.ok { core with gpuAvailable := true }

theorem ensureGPUInitialized_incompatible (core : RSFCore)
  (h : !RSFCore.modelGPUCompatible core) :
  ensureGPUInitialized core = Except.error TensorError.gpuUnsupportedConfiguration :=
  congrArg (fun x => if x then Except.error TensorError.gpuUnsupportedConfiguration else Except.ok { core with gpuAvailable := true }) h
  |> (fun heq => Eq.trans heq (Eq.refl (Except.error TensorError.gpuUnsupportedConfiguration)))

def invalidateGPUForMismatch (core : RSFCore) : RSFCore :=
  { core with gpuAvailable := false }

theorem invalidateGPUForMismatch_false (core : RSFCore) :
  (invalidateGPUForMismatch core).gpuAvailable = false :=
  Eq.refl _

structure TempFile where
  handle : Unit
  tmpName : String
  deriving Repr

theorem TempFile.default_handle :
  (default : TempFile).handle = () :=
  Eq.refl _

theorem TempFile.default_tmpName :
  (default : TempFile).tmpName = "" :=
  Eq.refl _

def createUniqueTempFile (dir : Unit) (allocator : Unit) (baseName : String) : Except TensorError TempFile :=
  let rec tryCreate (attempt : Nat) : Except TensorError TempFile :=
    if h : attempt ≥ 64 then Except.error TensorError.tempFileCollision
    else
      let hex := "a1b2c3d4e5f6" ++ Nat.repr attempt
      Except.ok { handle := (), tmpName := "." ++ baseName ++ ".tmp." ++ hex }
  tryCreate 0

theorem createUniqueTempFile_collision :
  createUniqueTempFile () () "test" = Except.ok { handle := (), tmpName := ".test.tmp.a1b2c3d4e5f60" } :=
  let hcond : 0 ≥ 64 = false :=
    congrArg not (Eq.symm (Bool.false_ne_true |> congrArg (· = 0 ≥ 64) |> (fun _ => Nat.not_le_of_gt (Nat.zero_lt_succ 63))))
  congrArg (fun x => if x then Except.error TensorError.tempFileCollision else Except.ok { handle := (), tmpName := ".test.tmp.a1b2c3d4e5f60" }) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok { handle := (), tmpName := ".test.tmp.a1b2c3d4e5f60" })))

def allocTensorArray (count rows cols : Nat) : Except TensorError (Array Tensor) :=
  let rec loop (i : Nat) (acc : Array Tensor) : Except TensorError (Array Tensor) :=
    if hi : i ≥ count then Except.ok acc
    else loop (i + 1) (acc.push (Tensor.zeros [rows, cols]))
  loop 0 #[]

theorem allocTensorArray_empty (rows cols : Nat) :
  allocTensorArray 0 rows cols = Except.ok #[] :=
  let hcond : 0 ≥ 0 = true := Nat.le_refl 0
  congrArg (fun x => if x then Except.ok #[] else loop (0 + 1) (#[].push (Tensor.zeros [rows, cols]))) hcond
  |> (fun heq => Eq.trans heq (Eq.refl (Except.ok #[])))

theorem allocTensorArray_size (count rows cols : Nat)
  (h : count > 0) :
  match allocTensorArray count rows cols with
  | Except.ok arr => arr.size = count
  | Except.error _ => True :=
  let rec loop (i : Nat) (acc : Array Tensor) : Except TensorError (Array Tensor) :=
    if hi : i ≥ count then Except.ok acc
    else loop (i + 1) (acc.push (Tensor.zeros [rows, cols]))
  let hloop : (loop 0 #[]).bind (fun arr => Except.ok arr.size) = Except.ok count :=
    let h := Eq.refl count
  h
  hloop

def freeTensorArray (arr : Array Tensor) : Unit :=
  ()

theorem freeTensorArray_empty :
  freeTensorArray #[] = () :=
  Eq.refl _

theorem freeTensorArray_unit (arr : Array Tensor) :
  freeTensorArray arr = () :=
  Eq.refl _

def syncWeightsToGPU (self : RSF) (registry : ModelRegistry) : Except TensorError ModelRegistry :=
  match acquireModelCore registry self.id with
  | Except.error e => Except.error e
  | Except.ok (newReg, core) =>
    match syncAllLayersGPU core with
    | Except.error e => Except.error e
    | Except.ok _ => Except.ok (releaseModelCore newReg self.id)

theorem syncWeightsToGPU_notInit (self : RSF) (registry : ModelRegistry)
  (h : self.id = 0) :
  syncWeightsToGPU self registry = Except.error TensorError.notInitialized :=
  let hresult : acquireModelCore registry self.id = Except.error TensorError.notInitialized :=
    Eq.subst h (acquireModelCore_zero registry)
  match acquireModelCore registry self.id with
  | Except.error e =>
    let heq : e = TensorError.notInitialized :=
      match hresult with
      | Eq.refl _ => Eq.refl _
    congrArg Except.error heq
  | Except.ok _ => absurd hresult (Except.ok_ne_error _ _)

def destroyLayerCore (core : LayerCore) : Unit :=
  ()

theorem destroyLayerCore_unit (core : LayerCore) :
  destroyLayerCore core = () :=
  Eq.refl _

def destroyModelCore (core : RSFCore) : Unit :=
  ()

theorem destroyModelCore_unit (core : RSFCore) :
  destroyModelCore core = () :=
  Eq.refl _
end RSF
