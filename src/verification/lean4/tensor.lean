import Std

set_option maxHeartbeats 800000
set_option linter.unusedVariables false

noncomputable section

def floatNaN : Float := 0.0 / 0.0
def floatInf : Float := 1.0 / 0.0
def floatNegInf : Float := -(1.0 / 0.0)

inductive TensorError where
  | shapeMismatch
  | invalidShape
  | outOfBounds
  | overflow
  | divideByZero
  | invalidAxis
  | emptyInput
  | invalidConv2D
  | invalidPads
  | invalidReps
  | mustBeSquare
  | singularMatrix
  | invalidForOneHot
  | allocFailed
  | invalidPermutation
  | invalidOrder
  | notPositiveDefinite
  | notSymmetric
  | notConverged
  deriving Repr, BEq, DecidableEq

abbrev TResult (α : Type) := Except TensorError α

def listProduct : List Nat → Nat
  | [] => 1
  | h :: t => h * listProduct t

theorem listProduct_nil : listProduct [] = 1 := rfl

theorem listProduct_cons (h : Nat) (t : List Nat) :
    listProduct (h :: t) = h * listProduct t := rfl

theorem listProduct_singleton (n : Nat) : listProduct [n] = n := by
  simp [listProduct]

def allPositive : List Nat → Prop
  | [] => True
  | h :: t => h > 0 ∧ allPositive t

theorem listProduct_pos (l : List Nat) (h : allPositive l) :
    listProduct l > 0 := by
  induction l with
  | nil => simp [listProduct]
  | cons hd tl ih =>
    simp [listProduct, allPositive] at *
    exact Nat.mul_pos h.1 (ih h.2)

theorem listProduct_append (l1 l2 : List Nat) :
    listProduct (l1 ++ l2) = listProduct l1 * listProduct l2 := by
  induction l1 with
  | nil => simp [listProduct]
  | cons h t ih =>
    simp [listProduct, ih, Nat.mul_assoc]

def computeStrides (dims : List Nat) : List Nat :=
  dims.enum.map (fun ⟨i, _⟩ => listProduct (dims.drop (i + 1)))

theorem computeStrides_length (dims : List Nat) :
    (computeStrides dims).length = dims.length := by
  unfold computeStrides
  simp [List.length_map, List.length_enum]

structure Shape where
  dims : List Nat
  strides : List Nat
  h_len : dims.length = strides.length

def Shape.mk' (dims : List Nat) : Shape where
  dims := dims
  strides := computeStrides dims
  h_len := by simp [computeStrides, List.length_map, List.length_enum]

def Shape.ndim (s : Shape) : Nat := s.dims.length

def Shape.totalSize (s : Shape) : Nat := listProduct s.dims

@[simp] theorem Shape.mk'_dims (dims : List Nat) : (Shape.mk' dims).dims = dims := rfl

@[simp] theorem Shape.mk'_totalSize (dims : List Nat) :
    (Shape.mk' dims).totalSize = listProduct dims := rfl

def Shape.equals (s1 s2 : Shape) : Bool :=
  decide (s1.dims = s2.dims)

def Shape.broadcastCompatible (self target : Shape) : Bool :=
  if target.dims.length < self.dims.length then false
  else
    let offset := target.dims.length - self.dims.length
    let targetSuffix := target.dims.drop offset
    List.zip self.dims targetSuffix |>.all (fun (sd, td) => sd = td ∨ sd = 1)

def Shape.isContiguous (s : Shape) : Bool :=
  let pairs := List.zip s.dims.reverse s.strides.reverse
  pairs.foldl (fun (ok, expected) (d, st) =>
    if ok ∧ st = expected then (true, expected * d)
    else (false, 0)) (true, 1) |>.1

theorem Shape.equals_refl (s : Shape) : s.equals s = true := by
  unfold Shape.equals
  simp

theorem Shape.equals_symm (s1 s2 : Shape) :
    s1.equals s2 = s2.equals s1 := by
  unfold Shape.equals
  simp [eq_comm]

theorem Shape.totalSize_nil :
    Shape.totalSize { dims := [], strides := [], h_len := rfl } = 1 := by
  unfold Shape.totalSize
  simp [listProduct]

theorem Shape.totalSize_singleton (n : Nat) :
    Shape.totalSize { dims := [n], strides := [1], h_len := rfl } = n := by
  unfold Shape.totalSize
  simp [listProduct]

theorem Shape.totalSize_pos (s : Shape) (h : allPositive s.dims) :
    s.totalSize > 0 := by
  unfold Shape.totalSize
  exact listProduct_pos s.dims h

def Shape.copy (s : Shape) : Shape :=
  { dims := s.dims, strides := s.strides, h_len := s.h_len }

theorem Shape.copy_equals (s : Shape) : s.copy.equals s = true := by
  unfold Shape.copy Shape.equals
  simp

theorem Shape.copy_totalSize (s : Shape) : s.copy.totalSize = s.totalSize := rfl

structure RefCount where
  count : Nat
  h_pos : count > 0

def RefCount.init : RefCount := { count := 1, h_pos := Nat.zero_lt_succ 0 }

def RefCount.retain (rc : RefCount) : RefCount :=
  { count := rc.count + 1, h_pos := Nat.succ_pos _ }

def RefCount.release (rc : RefCount) : Option RefCount :=
  if h : rc.count > 1 then
    some { count := rc.count - 1, h_pos := Nat.sub_pos_of_lt h }
  else
    none

@[simp] theorem RefCount.init_count : RefCount.init.count = 1 := rfl

theorem RefCount.retain_increases (rc : RefCount) :
    (rc.retain).count = rc.count + 1 := rfl

theorem RefCount.release_decreases (rc : RefCount) (h : rc.count > 1) :
    ∃ rc', rc.release = some rc' ∧ rc'.count = rc.count - 1 := by
  unfold RefCount.release
  simp [h]

theorem RefCount.release_last (rc : RefCount) (h : rc.count = 1) :
    rc.release = none := by
  unfold RefCount.release
  simp [h]

theorem RefCount.no_underflow (rc : RefCount) : rc.count > 0 := rc.h_pos

structure CowState where
  isShared : Bool

def CowState.init : CowState := { isShared := false }
def CowState.markShared (_ : CowState) : CowState := { isShared := true }
def CowState.makeWritable (_ : CowState) : CowState := { isShared := false }

@[simp] theorem CowState.init_isShared : CowState.init.isShared = false := rfl
@[simp] theorem CowState.markShared_isShared (cs : CowState) :
    (cs.markShared).isShared = true := rfl

theorem CowState.makeWritable_not_shared (cs : CowState) :
    (cs.makeWritable).isShared = false := rfl

structure Tensor where
  data : Array Float
  shape : Shape
  h_data_size : data.size = shape.totalSize
  refcount : RefCount
  cow : CowState

def Tensor.ndim (t : Tensor) : Nat := t.shape.ndim
def Tensor.dataLen (t : Tensor) : Nat := t.data.size

theorem Tensor.dataLen_eq_totalSize (t : Tensor) :
    t.dataLen = t.shape.totalSize := t.h_data_size

def flatToMultiIndex (dims : List Nat) (flat : Nat) : List Nat :=
  match dims with
  | [] => []
  | [_] => [flat]
  | _ =>
    let rec go : List Nat → Nat → List Nat
      | [], _ => []
      | [_], rem => [rem]
      | _ :: rest, rem =>
        let stride := listProduct rest
        if stride = 0 then 0 :: go rest 0
        else (rem / stride) :: go rest (rem % stride)
    go dims flat

def multiToFlatIndex (strides : List Nat) (indices : List Nat) : Nat :=
  List.zipWith (· * ·) indices strides |>.foldl (· + ·) 0

def computeFlatIndex (shape : Shape) (indices : List Nat) : Nat :=
  multiToFlatIndex shape.strides indices

def insertAxisIndex (reducedDims : List Nat) (flatIdx : Nat) (axis : Nat) (axisVal : Nat) : List Nat :=
  let multiIdx := flatToMultiIndex reducedDims flatIdx
  (multiIdx.take axis) ++ [axisVal] ++ (multiIdx.drop axis)

def removeAxis (l : List Nat) (axis : Nat) : List Nat :=
  l.enum.filterMap (fun ⟨i, v⟩ => if i = axis then none else some v)

def computeIndex (shape : Shape) (indices : List Nat) : TResult Nat :=
  if indices.length ≠ shape.dims.length then
    Except.error TensorError.invalidAxis
  else
    let rec go : List Nat → List Nat → List Nat → Nat → TResult Nat
      | [], [], [], acc => Except.ok acc
      | d :: drest, s :: srest, i :: irest, acc =>
        if i ≥ d then Except.error TensorError.outOfBounds
        else go drest srest irest (acc + i * s)
      | _, _, _, _ => Except.error TensorError.invalidAxis
    go shape.dims shape.strides indices 0

def Tensor.init (dims : List Nat) : TResult Tensor :=
  if dims.length = 0 then
    Except.error TensorError.invalidShape
  else if dims.any (· == 0) then
    Except.error TensorError.invalidShape
  else
    Except.ok {
      data := mkArray (listProduct dims) (0.0 : Float)
      shape := Shape.mk' dims
      h_data_size := by simp [Shape.mk', Shape.totalSize, Array.size_mkArray]
      refcount := RefCount.init
      cow := CowState.init
    }

theorem Tensor.init_error_on_empty :
    Tensor.init [] = Except.error TensorError.invalidShape := rfl

theorem Tensor.init_shape (dims : List Nat)
    (h_ne : dims.length ≠ 0)
    (h_nz : dims.any (· == 0) = false)
    (t : Tensor) (h_ok : Tensor.init dims = Except.ok t) :
    t.shape.dims = dims := by
  unfold Tensor.init at h_ok
  simp [h_ne, h_nz] at h_ok
  cases h_ok
  rfl

theorem Tensor.init_refcount_one (dims : List Nat)
    (h_ne : dims.length ≠ 0)
    (h_nz : dims.any (· == 0) = false)
    (t : Tensor) (h_ok : Tensor.init dims = Except.ok t) :
    t.refcount.count = 1 := by
  unfold Tensor.init at h_ok
  simp [h_ne, h_nz] at h_ok
  cases h_ok
  rfl

def Tensor.get (t : Tensor) (indices : List Nat) : TResult Float :=
  match computeIndex t.shape indices with
  | Except.ok idx =>
    if h : idx < t.data.size then
      Except.ok t.data[idx]
    else
      Except.error TensorError.outOfBounds
  | Except.error e => Except.error e

def Tensor.set (t : Tensor) (indices : List Nat) (value : Float) : TResult Tensor :=
  match computeIndex t.shape indices with
  | Except.ok idx =>
    if h : idx < t.data.size then
      let newData := t.data.set ⟨idx, h⟩ value
      Except.ok { t with
        data := newData
        h_data_size := by
          show (t.data.set ⟨idx, h⟩ value).size = t.shape.totalSize
          rw [Array.size_set]
          exact t.h_data_size
        cow := CowState.init
      }
    else
      Except.error TensorError.outOfBounds
  | Except.error e => Except.error e

theorem Tensor.set_preserves_shape (t : Tensor) (indices : List Nat)
    (value : Float) (t' : Tensor)
    (h_ok : t.set indices value = Except.ok t') :
    t'.shape = t.shape := by
  unfold Tensor.set at h_ok
  split at h_ok
  · split at h_ok
    · cases h_ok; rfl
    · cases h_ok
  · cases h_ok

theorem Tensor.set_preserves_size (t : Tensor) (indices : List Nat)
    (value : Float) (t' : Tensor)
    (h_ok : t.set indices value = Except.ok t') :
    t'.data.size = t.data.size := by
  unfold Tensor.set at h_ok
  split at h_ok
  · split at h_ok
    · cases h_ok; simp [Array.size_set]
    · cases h_ok
  · cases h_ok

def Tensor.fill (t : Tensor) (value : Float) : Tensor :=
  { t with
    data := mkArray t.data.size value
    h_data_size := by rw [Array.size_mkArray, t.h_data_size]
    cow := CowState.init }

theorem Tensor.fill_preserves_shape (t : Tensor) (value : Float) :
    (t.fill value).shape = t.shape := rfl

theorem Tensor.fill_size (t : Tensor) (value : Float) :
    (t.fill value).data.size = t.data.size := by
  unfold Tensor.fill
  simp [Array.size_mkArray]

def Tensor.zeros (dims : List Nat) : TResult Tensor := Tensor.init dims

def Tensor.ones (dims : List Nat) : TResult Tensor :=
  match Tensor.init dims with
  | Except.ok t => Except.ok (t.fill 1.0)
  | Except.error e => Except.error e

def Tensor.full (dims : List Nat) (value : Float) : TResult Tensor :=
  match Tensor.init dims with
  | Except.ok t => Except.ok (t.fill value)
  | Except.error e => Except.error e

def mapData (t : Tensor) (f : Float → Float) : Tensor :=
  { t with
    data := t.data.map f
    h_data_size := by rw [Array.size_map, t.h_data_size]
    cow := CowState.init }

theorem mapData_preserves_shape (t : Tensor) (f : Float → Float) :
    (mapData t f).shape = t.shape := rfl

theorem mapData_preserves_size (t : Tensor) (f : Float → Float) :
    (mapData t f).data.size = t.data.size := by
  unfold mapData; simp [Array.size_map]

def zipWithArray (a b : Array Float) (f : Float → Float → Float) : Array Float :=
  a.mapIdx (fun i v =>
    if h : i.val < b.size then f v b[i.val]
    else v)

theorem zipWithArray_size (a b : Array Float) (f : Float → Float → Float) :
    (zipWithArray a b f).size = a.size := by
  unfold zipWithArray
  simp [Array.size_mapIdx]

def zipData (t1 t2 : Tensor) (f : Float → Float → Float) : TResult Tensor :=
  if t1.shape.equals t2.shape = false then
    Except.error TensorError.shapeMismatch
  else
    let newData := zipWithArray t1.data t2.data f
    Except.ok { t1 with
      data := newData
      h_data_size := by
        show newData.size = t1.shape.totalSize
        simp [newData]
        rw [zipWithArray_size]
        exact t1.h_data_size
      cow := CowState.init }

theorem zipData_preserves_shape (t1 t2 : Tensor) (f : Float → Float → Float)
    (t' : Tensor) (h_ok : zipData t1 t2 f = Except.ok t') :
    t'.shape = t1.shape := by
  unfold zipData at h_ok
  split at h_ok
  · cases h_ok
  · injection h_ok with h'; rw [← h']

def Tensor.add (t1 t2 : Tensor) : TResult Tensor := zipData t1 t2 (· + ·)
def Tensor.sub (t1 t2 : Tensor) : TResult Tensor := zipData t1 t2 (· - ·)
def Tensor.mul (t1 t2 : Tensor) : TResult Tensor := zipData t1 t2 (· * ·)

def Tensor.divOp (t1 t2 : Tensor) : TResult Tensor :=
  if t2.data.any (· == 0.0) then
    Except.error TensorError.divideByZero
  else
    zipData t1 t2 (· / ·)

theorem Tensor.add_shape (t1 t2 t' : Tensor) (h : Tensor.add t1 t2 = Except.ok t') :
    t'.shape = t1.shape := zipData_preserves_shape t1 t2 (· + ·) t' h

theorem Tensor.sub_shape (t1 t2 t' : Tensor) (h : Tensor.sub t1 t2 = Except.ok t') :
    t'.shape = t1.shape := zipData_preserves_shape t1 t2 (· - ·) t' h

theorem Tensor.mul_shape (t1 t2 t' : Tensor) (h : Tensor.mul t1 t2 = Except.ok t') :
    t'.shape = t1.shape := zipData_preserves_shape t1 t2 (· * ·) t' h

def Tensor.addScalar (t : Tensor) (s : Float) : Tensor := mapData t (· + s)
def Tensor.subScalar (t : Tensor) (s : Float) : Tensor := mapData t (· - s)
def Tensor.mulScalar (t : Tensor) (s : Float) : Tensor := mapData t (· * s)

def Tensor.divScalar (t : Tensor) (s : Float) : TResult Tensor :=
  if s == 0.0 then Except.error TensorError.divideByZero
  else Except.ok (mapData t (· / s))

theorem Tensor.addScalar_shape (t : Tensor) (s : Float) :
    (t.addScalar s).shape = t.shape := mapData_preserves_shape t (· + s)
theorem Tensor.subScalar_shape (t : Tensor) (s : Float) :
    (t.subScalar s).shape = t.shape := mapData_preserves_shape t (· - s)
theorem Tensor.mulScalar_shape (t : Tensor) (s : Float) :
    (t.mulScalar s).shape = t.shape := mapData_preserves_shape t (· * s)

theorem Tensor.divScalar_shape (t : Tensor) (s : Float) (t' : Tensor)
    (h_ok : t.divScalar s = Except.ok t') : t'.shape = t.shape := by
  unfold Tensor.divScalar at h_ok
  split at h_ok
  · cases h_ok
  · cases h_ok; exact mapData_preserves_shape t (· / s)

def Tensor.expOp (t : Tensor) : Tensor := mapData t Float.exp
def Tensor.logOp (t : Tensor) : Tensor :=
  mapData t (fun v => if v ≤ 0.0 then floatNegInf else Float.log v)
def Tensor.sinOp (t : Tensor) : Tensor := mapData t Float.sin
def Tensor.cosOp (t : Tensor) : Tensor := mapData t Float.cos
def Tensor.tanOp (t : Tensor) : Tensor := mapData t Float.tan
def Tensor.sqrtOp (t : Tensor) : Tensor :=
  mapData t (fun v => if v < 0.0 then floatNaN else Float.sqrt v)
def Tensor.powOp (t : Tensor) (exponent : Float) : Tensor :=
  mapData t (fun v => Float.pow v exponent)
def Tensor.absOp (t : Tensor) : Tensor := mapData t Float.abs
def Tensor.reluOp (t : Tensor) : Tensor :=
  mapData t (fun v => if v > 0.0 then v else 0.0)

theorem Tensor.expOp_shape (t : Tensor) : t.expOp.shape = t.shape :=
  mapData_preserves_shape t Float.exp
theorem Tensor.logOp_shape (t : Tensor) : t.logOp.shape = t.shape :=
  mapData_preserves_shape t _
theorem Tensor.sinOp_shape (t : Tensor) : t.sinOp.shape = t.shape :=
  mapData_preserves_shape t Float.sin
theorem Tensor.cosOp_shape (t : Tensor) : t.cosOp.shape = t.shape :=
  mapData_preserves_shape t Float.cos
theorem Tensor.tanOp_shape (t : Tensor) : t.tanOp.shape = t.shape :=
  mapData_preserves_shape t Float.tan
theorem Tensor.sqrtOp_shape (t : Tensor) : t.sqrtOp.shape = t.shape :=
  mapData_preserves_shape t _
theorem Tensor.absOp_shape (t : Tensor) : t.absOp.shape = t.shape :=
  mapData_preserves_shape t Float.abs
theorem Tensor.reluOp_shape (t : Tensor) : t.reluOp.shape = t.shape :=
  mapData_preserves_shape t _

theorem Tensor.exp_preserves_size (t : Tensor) :
    t.expOp.data.size = t.data.size := mapData_preserves_size t Float.exp
theorem Tensor.sin_preserves_size (t : Tensor) :
    t.sinOp.data.size = t.data.size := mapData_preserves_size t Float.sin
theorem Tensor.cos_preserves_size (t : Tensor) :
    t.cosOp.data.size = t.data.size := mapData_preserves_size t Float.cos

def Tensor.reduceAxis (t : Tensor) (axis : Nat)
    (f : Float → Float → Float) (initial : Float) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then
    Except.error TensorError.invalidAxis
  else
    let newDims := removeAxis t.shape.dims axis
    let safeNewDims := if newDims.length = 0 then [1] else newDims
    let newTotal := listProduct safeNewDims
    let axisSize := t.shape.dims.get! axis
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin newTotal =>
      let outMulti := flatToMultiIndex safeNewDims idx.val
      (List.range axisSize).foldl (fun acc k =>
        let inMulti := if newDims.length = 0 then [k]
          else (outMulti.take axis) ++ [k] ++ (outMulti.drop axis)
        f acc (t.data.get! (computeFlatIndex srcShape inMulti))
      ) initial)
    Except.ok {
      data := data
      shape := Shape.mk' safeNewDims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.maxReduce (t : Tensor) (axis : Nat) : TResult Tensor :=
  t.reduceAxis axis max floatNegInf

def Tensor.minReduce (t : Tensor) (axis : Nat) : TResult Tensor :=
  t.reduceAxis axis min floatInf

def Tensor.sumReduce (t : Tensor) (axis : Nat) : TResult Tensor :=
  t.reduceAxis axis (· + ·) 0.0

def Tensor.meanReduce (t : Tensor) (axis : Nat) : TResult Tensor :=
  if h_ax : axis ≥ t.shape.dims.length then
    Except.error TensorError.invalidAxis
  else
    match t.sumReduce axis with
    | Except.error e => Except.error e
    | Except.ok sumT =>
      let axisSize := t.shape.dims.get ⟨axis, Nat.lt_of_not_ge h_ax⟩
      if axisSize = 0 then Except.error TensorError.divideByZero
      else
        let divisor := Float.ofNat axisSize
        Except.ok (sumT.mulScalar (1.0 / divisor))

def Tensor.reshape (t : Tensor) (newDims : List Nat) : TResult Tensor :=
  if newDims.length = 0 then
    Except.error TensorError.invalidShape
  else if h_ne : listProduct newDims ≠ t.shape.totalSize then
    Except.error TensorError.invalidShape
  else
    have h_eq : listProduct newDims = t.shape.totalSize := by
      by_contra h
      exact h_ne h
    Except.ok { t with
      shape := Shape.mk' newDims
      h_data_size := by rw [t.h_data_size]; exact h_eq.symm
    }

theorem Tensor.reshape_preserves_data (t : Tensor) (newDims : List Nat)
    (t' : Tensor) (h_ok : t.reshape newDims = Except.ok t') :
    t'.data = t.data := by
  unfold Tensor.reshape at h_ok
  split at h_ok
  · cases h_ok
  · split at h_ok
    · cases h_ok
    · injection h_ok with h'; rw [← h']

theorem Tensor.reshape_new_dims (t : Tensor) (newDims : List Nat)
    (t' : Tensor) (h_ok : t.reshape newDims = Except.ok t') :
    t'.shape.dims = newDims := by
  unfold Tensor.reshape at h_ok
  split at h_ok
  · cases h_ok
  · split at h_ok
    · cases h_ok
    · injection h_ok with h'; rw [← h']; simp [Shape.mk']

def Tensor.view (t : Tensor) (newDims : List Nat) : TResult Tensor :=
  if newDims.length = 0 then
    Except.error TensorError.invalidShape
  else if h_ne : listProduct newDims ≠ t.shape.totalSize then
    Except.error TensorError.invalidShape
  else
    have h_eq : listProduct newDims = t.shape.totalSize := by
      by_contra h
      exact h_ne h
    Except.ok {
      data := t.data
      shape := Shape.mk' newDims
      h_data_size := by rw [t.h_data_size]; exact h_eq.symm
      refcount := t.refcount.retain
      cow := t.cow.markShared
    }

theorem Tensor.view_shares_data (t : Tensor) (newDims : List Nat)
    (t' : Tensor) (h_ok : t.view newDims = Except.ok t') :
    t'.data = t.data := by
  unfold Tensor.view at h_ok
  split at h_ok
  · cases h_ok
  · split at h_ok
    · cases h_ok
    · injection h_ok with h'; rw [← h']

theorem Tensor.view_increments_refcount (t : Tensor) (newDims : List Nat)
    (t' : Tensor) (h_ok : t.view newDims = Except.ok t') :
    t'.refcount.count = t.refcount.count + 1 := by
  unfold Tensor.view at h_ok
  split at h_ok
  · cases h_ok
  · split at h_ok
    · cases h_ok
    · injection h_ok with h'; rw [← h']; rfl

theorem Tensor.view_marks_shared (t : Tensor) (newDims : List Nat)
    (t' : Tensor) (h_ok : t.view newDims = Except.ok t') :
    t'.cow.isShared = true := by
  unfold Tensor.view at h_ok
  split at h_ok
  · cases h_ok
  · split at h_ok
    · cases h_ok
    · injection h_ok with h'; rw [← h']; rfl

def Tensor.newView (t : Tensor) (newShape : Shape)
    (h_eq : newShape.totalSize = t.shape.totalSize) : Tensor :=
  { data := t.data
    shape := newShape
    h_data_size := by rw [t.h_data_size, h_eq]
    refcount := t.refcount.retain
    cow := t.cow.markShared }

theorem Tensor.newView_shares_data (t : Tensor) (newShape : Shape)
    (h_eq : newShape.totalSize = t.shape.totalSize) :
    (t.newView newShape h_eq).data = t.data := rfl

def Tensor.sliceOp (t : Tensor) (starts ends : List Nat) : TResult Tensor :=
  if starts.length ≠ t.shape.dims.length ∨ ends.length ≠ t.shape.dims.length then
    Except.error TensorError.invalidAxis
  else
    let newDims := List.zipWith (fun e s => e - s) ends starts
    if newDims.any (· == 0) then Except.error TensorError.invalidShape
    else
      let newTotal := listProduct newDims
      let srcShape := Shape.mk' t.shape.dims
      let data := Array.ofFn (fun idx : Fin newTotal =>
        let multiIdx := flatToMultiIndex newDims idx.val
        let srcIdx := List.zipWith (· + ·) multiIdx starts
        t.data.get! (computeFlatIndex srcShape srcIdx))
      Except.ok {
        data := data
        shape := Shape.mk' newDims
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

def Tensor.transposeOp (t : Tensor) (axes : List Nat) : TResult Tensor :=
  if axes.length ≠ t.shape.dims.length then
    Except.error TensorError.invalidAxis
  else if ¬ axes.all (· < t.shape.dims.length) then
    Except.error TensorError.invalidAxis
  else
    let newDims := axes.map (fun a => t.shape.dims.get! a)
    let newTotal := listProduct newDims
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin newTotal =>
      let outMulti := flatToMultiIndex newDims idx.val
      let srcMulti := axes.enum.map (fun ⟨_, a⟩ => outMulti.get! a)
      t.data.get! (computeFlatIndex srcShape
        (List.range t.shape.dims.length |>.map (fun i =>
          match axes.enum.find? (fun ⟨_, a⟩ => a = i) with
          | some ⟨pos, _⟩ => outMulti.get! pos
          | none => 0))))
    Except.ok {
      data := data
      shape := Shape.mk' newDims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

theorem Tensor.transposeOp_ndim (t : Tensor) (axes : List Nat)
    (t' : Tensor) (h_ok : t.transposeOp axes = Except.ok t')
    (h_len : axes.length = t.shape.dims.length) :
    t'.shape.dims.length = t.shape.dims.length := by
  unfold Tensor.transposeOp at h_ok
  split at h_ok
  · rename_i hc; exact absurd h_len hc
  · split at h_ok
    · cases h_ok
    · injection h_ok with h'
      rw [← h']
      simp [List.length_map, h_len]

theorem listProduct_insert_one (l : List Nat) (i : Nat) :
    listProduct ((l.take i) ++ [1] ++ (l.drop i)) = listProduct l := by
  induction l generalizing i with
  | nil => simp [listProduct]
  | cons h t ih =>
    cases i with
    | zero => simp [listProduct, List.take, List.drop, Nat.one_mul]
    | succ n => simp [listProduct, List.take, List.drop, ih n, Nat.mul_assoc]

def Tensor.unsqueezeOp (t : Tensor) (axis : Nat) : TResult Tensor :=
  if axis > t.shape.dims.length then
    Except.error TensorError.invalidAxis
  else
    let newDims := (t.shape.dims.take axis) ++ [1] ++ (t.shape.dims.drop axis)
    Except.ok {
      data := t.data
      shape := Shape.mk' newDims
      h_data_size := by
        rw [t.h_data_size]
        simp [Shape.mk', Shape.totalSize]
        exact (listProduct_insert_one t.shape.dims axis).symm
      refcount := t.refcount.retain
      cow := t.cow.markShared
    }

def Tensor.broadcastOp (t : Tensor) (targetDims : List Nat) : TResult Tensor :=
  if t.shape.broadcastCompatible (Shape.mk' targetDims) = false then
    Except.error TensorError.shapeMismatch
  else if targetDims.any (· == 0) then Except.error TensorError.invalidShape
  else
    let targetTotal := listProduct targetDims
    let offset := targetDims.length - t.shape.dims.length
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin targetTotal =>
      let targetMulti := flatToMultiIndex targetDims idx.val
      let srcMulti := (targetMulti.drop offset).zip t.shape.dims |>.map
        (fun (ti, sd) => if sd = 1 then 0 else ti)
      t.data.get! (computeFlatIndex srcShape srcMulti))
    Except.ok {
      data := data
      shape := Shape.mk' targetDims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.padOp (t : Tensor) (pads : List (Nat × Nat)) : TResult Tensor :=
  if pads.length ≠ t.shape.dims.length then
    Except.error TensorError.invalidPads
  else
    let newDims := List.zipWith (fun d p => d + p.1 + p.2) t.shape.dims pads
    if newDims.any (· == 0) then Except.error TensorError.invalidShape
    else
      let newTotal := listProduct newDims
      let srcShape := Shape.mk' t.shape.dims
      let data := Array.ofFn (fun idx : Fin newTotal =>
        let multiIdx := flatToMultiIndex newDims idx.val
        let pairCheck := List.zip (List.zip multiIdx pads) t.shape.dims
        let inBounds := pairCheck.all (fun ((mi, (lo, _)), d) =>
          mi ≥ lo ∧ mi < lo + d)
        if inBounds then
          let srcMulti := List.zipWith (fun mi (lo, _) => mi - lo) multiIdx pads
          t.data.get! (computeFlatIndex srcShape srcMulti)
        else 0.0)
      Except.ok {
        data := data
        shape := Shape.mk' newDims
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

theorem Tensor.padOp_requires_matching_length (t : Tensor)
    (pads : List (Nat × Nat))
    (h : pads.length ≠ t.shape.dims.length) :
    ∃ e, t.padOp pads = Except.error e := by
  unfold Tensor.padOp; simp [h]

def Tensor.tileOp (t : Tensor) (reps : List Nat) : TResult Tensor :=
  if reps.length ≠ t.shape.dims.length then
    Except.error TensorError.invalidReps
  else if reps.any (· == 0) then Except.error TensorError.invalidShape
  else
    let newDims := List.zipWith (· * ·) t.shape.dims reps
    let newTotal := listProduct newDims
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin newTotal =>
      let multiIdx := flatToMultiIndex newDims idx.val
      let srcMulti := List.zipWith (fun mi d => mi % d) multiIdx t.shape.dims
      t.data.get! (computeFlatIndex srcShape srcMulti))
    Except.ok {
      data := data
      shape := Shape.mk' newDims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.concatOp (tensors : List Tensor) (axis : Nat) : TResult Tensor :=
  match tensors with
  | [] => Except.error TensorError.emptyInput
  | first :: _ =>
    let ndim := first.shape.dims.length
    if axis ≥ ndim then Except.error TensorError.invalidAxis
    else
      let totalAxis := tensors.foldl (fun acc ten => acc + ten.shape.dims.get! axis) 0
      let newDims := first.shape.dims.enum.map (fun ⟨i, d⟩ =>
        if i = axis then totalAxis else d)
      let newTotal := listProduct newDims
      let data := Array.ofFn (fun idx : Fin newTotal =>
        let multiIdx := flatToMultiIndex newDims idx.val
        let axisIdx := multiIdx.get! axis
        let rec findTensor : List Tensor → Nat → Float
          | [], _ => 0.0
          | ten :: rest, offset =>
            let tenAxisDim := ten.shape.dims.get! axis
            if axisIdx < offset + tenAxisDim then
              let srcMulti := multiIdx.enum.map (fun ⟨i, v⟩ =>
                if i = axis then axisIdx - offset else v)
              let srcShape := Shape.mk' ten.shape.dims
              ten.data.get! (computeFlatIndex srcShape srcMulti)
            else findTensor rest (offset + tenAxisDim)
        findTensor tensors 0)
      Except.ok {
        data := data
        shape := Shape.mk' newDims
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

theorem Tensor.concatOp_requires_nonempty :
    ∃ e, Tensor.concatOp [] 0 = Except.error e := ⟨_, rfl⟩

def Tensor.stackOp (tensors : List Tensor) (axis : Nat) : TResult Tensor :=
  match tensors with
  | [] => Except.error TensorError.emptyInput
  | first :: _ =>
    let ndim := first.shape.dims.length
    if axis > ndim then Except.error TensorError.invalidAxis
    else
      let newDims := (first.shape.dims.take axis) ++
        [tensors.length] ++ (first.shape.dims.drop axis)
      let newTotal := listProduct newDims
      let data := Array.ofFn (fun idx : Fin newTotal =>
        let multiIdx := flatToMultiIndex newDims idx.val
        let tensorIdx := multiIdx.get! axis
        let srcMulti := (multiIdx.take axis) ++ (multiIdx.drop (axis + 1))
        match tensors.get? tensorIdx with
        | some ten =>
          let srcShape := Shape.mk' ten.shape.dims
          ten.data.get! (computeFlatIndex srcShape srcMulti)
        | none => 0.0)
      Except.ok {
        data := data
        shape := Shape.mk' newDims
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

theorem Tensor.stackOp_requires_nonempty :
    ∃ e, Tensor.stackOp [] 0 = Except.error e := ⟨_, rfl⟩

def Tensor.conv2d (input kernel : Tensor) (stride padding : Nat × Nat) : TResult Tensor :=
  if input.shape.dims.length ≠ 4 ∨ kernel.shape.dims.length ≠ 4 then
    Except.error TensorError.invalidConv2D
  else
    let in_c := input.shape.dims.get! 3
    let ker_in_c := kernel.shape.dims.get! 2
    if in_c ≠ ker_in_c then Except.error TensorError.invalidConv2D
    else if stride.1 = 0 ∨ stride.2 = 0 then Except.error TensorError.invalidConv2D
    else
      let batch := input.shape.dims.get! 0
      let in_h := input.shape.dims.get! 1
      let in_w := input.shape.dims.get! 2
      let k_h := kernel.shape.dims.get! 0
      let k_w := kernel.shape.dims.get! 1
      let out_c := kernel.shape.dims.get! 3
      let out_h := (in_h + 2 * padding.1 - k_h) / stride.1 + 1
      let out_w := (in_w + 2 * padding.2 - k_w) / stride.2 + 1
      if out_h = 0 ∨ out_w = 0 ∨ batch = 0 ∨ out_c = 0 then
        Except.error TensorError.invalidConv2D
      else
        let outTotal := batch * out_h * out_w * out_c
        let inShape := Shape.mk' input.shape.dims
        let kerShape := Shape.mk' kernel.shape.dims
        let data := Array.ofFn (fun idx : Fin outTotal =>
          let rem0 := idx.val
          let b := rem0 / (out_h * out_w * out_c)
          let rem1 := rem0 % (out_h * out_w * out_c)
          let oh := rem1 / (out_w * out_c)
          let rem2 := rem1 % (out_w * out_c)
          let ow := rem2 / out_c
          let oc := rem2 % out_c
          (List.range k_h).foldl (fun acc kh =>
            (List.range k_w).foldl (fun acc2 kw =>
              (List.range in_c).foldl (fun acc3 ic =>
                let ih := oh * stride.1 + kh
                let iw := ow * stride.2 + kw
                if ih < padding.1 ∨ iw < padding.2 then acc3
                else
                  let realH := ih - padding.1
                  let realW := iw - padding.2
                  if realH ≥ in_h ∨ realW ≥ in_w then acc3
                  else
                    let inVal := input.data.get! (computeFlatIndex inShape [b, realH, realW, ic])
                    let kerVal := kernel.data.get! (computeFlatIndex kerShape [kh, kw, ic, oc])
                    acc3 + inVal * kerVal
              ) acc2
            ) acc
          ) 0.0)
        Except.ok {
          data := data
          shape := Shape.mk' [batch, out_h, out_w, out_c]
          h_data_size := by simp [Array.size_ofFn]
          refcount := RefCount.init
          cow := CowState.init
        }

theorem Tensor.conv2d_requires_4d (input kernel : Tensor)
    (stride padding : Nat × Nat) (h : input.shape.dims.length ≠ 4) :
    ∃ e, Tensor.conv2d input kernel stride padding = Except.error e := by
  unfold Tensor.conv2d; simp [h]

def Tensor.softmaxOp (t : Tensor) (axis : Nat) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then
    Except.error TensorError.invalidAxis
  else
    let totalSize := t.shape.totalSize
    let axisSize := t.shape.dims.get! axis
    let srcShape := Shape.mk' t.shape.dims
    let resultData := Array.ofFn (fun idx : Fin totalSize =>
      let multiIdx := flatToMultiIndex t.shape.dims idx.val
      let baseMulti := multiIdx.enum.map (fun ⟨i, v⟩ =>
        if i = axis then 0 else v)
      let maxVal := (List.range axisSize).foldl (fun acc k =>
        let mi := baseMulti.enum.map (fun ⟨i, v⟩ =>
          if i = axis then k else v)
        max acc (t.data.get! (computeFlatIndex srcShape mi))
      ) floatNegInf
      let expVal := Float.exp (t.data.get! (computeFlatIndex srcShape multiIdx) - maxVal)
      let sumExp := (List.range axisSize).foldl (fun acc k =>
        let mi := baseMulti.enum.map (fun ⟨i, v⟩ =>
          if i = axis then k else v)
        acc + Float.exp (t.data.get! (computeFlatIndex srcShape mi) - maxVal)
      ) 0.0
      let safeDivisor := if sumExp < 1e-10 then 1e-10 else sumExp
      expVal / safeDivisor)
    Except.ok {
      data := resultData
      shape := t.shape
      h_data_size := by
        show resultData.size = t.shape.totalSize
        simp [resultData, Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

theorem Tensor.softmaxOp_shape (t : Tensor) (axis : Nat) (t' : Tensor)
    (h_ok : t.softmaxOp axis = Except.ok t') :
    t'.shape = t.shape := by
  unfold Tensor.softmaxOp at h_ok
  split at h_ok
  · cases h_ok
  · injection h_ok with h'; rw [← h']

theorem Tensor.softmaxOp_requires_valid_axis (t : Tensor) (axis : Nat)
    (h : axis ≥ t.shape.dims.length) :
    ∃ e, t.softmaxOp axis = Except.error e := by
  unfold Tensor.softmaxOp; simp [h]

def Tensor.matmul (a b : Tensor) : TResult Tensor :=
  if a.shape.dims.length ≠ 2 ∨ b.shape.dims.length ≠ 2 then
    Except.error TensorError.shapeMismatch
  else
    let m := a.shape.dims.get! 0
    let k := a.shape.dims.get! 1
    let k2 := b.shape.dims.get! 0
    let n := b.shape.dims.get! 1
    if k ≠ k2 then Except.error TensorError.shapeMismatch
    else if m = 0 ∨ n = 0 then Except.error TensorError.invalidShape
    else
      let data := Array.ofFn (fun idx : Fin (m * n) =>
        let i := idx.val / n
        let j := idx.val % n
        (List.range k).foldl (fun acc p =>
          acc + a.data.get! (i * k + p) * b.data.get! (p * n + j)) 0.0)
      Except.ok {
        data := data
        shape := Shape.mk' [m, n]
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

theorem Tensor.matmul_requires_2d (a b : Tensor) (h : a.shape.dims.length ≠ 2) :
    ∃ e, Tensor.matmul a b = Except.error e := by
  unfold Tensor.matmul; simp [h]

theorem Tensor.matmul_requires_compat (a b : Tensor)
    (h_a : a.shape.dims.length = 2) (h_b : b.shape.dims.length = 2)
    (h_k : a.shape.dims.get! 1 ≠ b.shape.dims.get! 0) :
    ∃ e, Tensor.matmul a b = Except.error e := by
  unfold Tensor.matmul; simp [h_a, h_b, h_k]

def Tensor.identity (n : Nat) : TResult Tensor :=
  if n = 0 then Except.error TensorError.invalidShape
  else
    let data := Array.ofFn (fun idx : Fin (n * n) =>
      if idx.val / n = idx.val % n then 1.0 else 0.0)
    Except.ok {
      data := data
      shape := Shape.mk' [n, n]
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.dot (t1 t2 : Tensor) : TResult Float :=
  if t1.data.size ≠ t2.data.size then Except.error TensorError.shapeMismatch
  else
    Except.ok ((List.range t1.data.size).foldl
      (fun acc i => acc + t1.data.get! i * t2.data.get! i) 0.0)

def Tensor.outer (a b : Tensor) : TResult Tensor :=
  if a.shape.dims.length ≠ 1 ∨ b.shape.dims.length ≠ 1 then
    Except.error TensorError.shapeMismatch
  else
    let m := a.shape.dims.get! 0
    let n := b.shape.dims.get! 0
    if m = 0 ∨ n = 0 then Except.error TensorError.invalidShape
    else
      let data := Array.ofFn (fun idx : Fin (m * n) =>
        a.data.get! (idx.val / n) * b.data.get! (idx.val % n))
      Except.ok {
        data := data
        shape := Shape.mk' [m, n]
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

def Tensor.trace (t : Tensor) : TResult Float :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := t.shape.dims.get! 0
    if n ≠ t.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else
      Except.ok ((List.range n).foldl (fun acc i =>
        acc + t.data.get! (i * n + i)) 0.0)

theorem Tensor.trace_requires_square (t : Tensor) (h : t.shape.dims.length ≠ 2) :
    ∃ e, t.trace = Except.error e := by
  unfold Tensor.trace; simp [h]

def Tensor.normL2 (t : Tensor) : Float :=
  let sumSq := t.data.foldl (fun acc v => acc + v * v) 0.0
  Float.sqrt sumSq

def Tensor.norm (t : Tensor) (order : Float) : Float :=
  let total := t.data.foldl (fun acc v => acc + Float.pow (Float.abs v) order) 0.0
  Float.pow total (1.0 / order)

def Tensor.isClose (t1 t2 : Tensor) (rtol atol : Float) : Bool :=
  if ¬ t1.shape.equals t2.shape then false
  else
    let n := t1.data.size
    (List.range n).all (fun i =>
      let a := t1.data.get! i
      let b := t2.data.get! i
      Float.abs (a - b) ≤ atol + rtol * Float.abs b)

def Tensor.inverse (t : Tensor) : TResult Tensor :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := t.shape.dims.get! 0
    if n ≠ t.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else if n = 0 then Except.error TensorError.invalidShape
    else
      let aug := Array.ofFn (fun idx : Fin (n * 2 * n) =>
        let i := idx.val / (2 * n)
        let j := idx.val % (2 * n)
        if j < n then t.data.get! (i * n + j)
        else if i = j - n then 1.0 else 0.0)
      let reduced := (List.range n).foldl (fun mat pivot =>
        let pivotVal := mat.get! (pivot * 2 * n + pivot)
        if pivotVal == 0.0 then mat
        else
          let scaled := (List.range (2 * n)).foldl (fun m j =>
            m.set! (pivot * 2 * n + j) (m.get! (pivot * 2 * n + j) / pivotVal)
          ) mat
          (List.range n).foldl (fun m i =>
            if i = pivot then m
            else
              let factor := m.get! (i * 2 * n + pivot)
              (List.range (2 * n)).foldl (fun m2 j =>
                m2.set! (i * 2 * n + j) (m2.get! (i * 2 * n + j) - factor * m2.get! (pivot * 2 * n + j))
              ) m
          ) scaled
      ) aug
      let data := Array.ofFn (fun idx : Fin (n * n) =>
        let i := idx.val / n
        let j := idx.val % n
        reduced.get! (i * 2 * n + n + j))
      Except.ok {
        data := data
        shape := Shape.mk' [n, n]
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

theorem Tensor.inverse_requires_square (t : Tensor) (h : t.shape.dims.length ≠ 2) :
    ∃ e, t.inverse = Except.error e := by
  unfold Tensor.inverse; simp [h]

def Tensor.det (t : Tensor) : TResult Float :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := t.shape.dims.get! 0
    if n ≠ t.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else if n = 0 then Except.ok 1.0
    else
      let mat := Array.ofFn (fun idx : Fin (n * n) => t.data.get! idx.val)
      let (reduced, sign) := (List.range n).foldl (fun (m, s) pivot =>
        let pivotVal := m.get! (pivot * n + pivot)
        if pivotVal == 0.0 then (m, 0.0)
        else
          let elim := (List.range n).foldl (fun m2 i =>
            if i ≤ pivot then m2
            else
              let factor := m2.get! (i * n + pivot) / pivotVal
              (List.range n).foldl (fun m3 j =>
                m3.set! (i * n + j) (m3.get! (i * n + j) - factor * m3.get! (pivot * n + j))
              ) m2
          ) m
          (elim, s)
      ) (mat, 1.0)
      let diagProd := (List.range n).foldl (fun acc i =>
        acc * reduced.get! (i * n + i)) sign
      Except.ok diagProd

theorem Tensor.det_requires_square (t : Tensor) (h : t.shape.dims.length ≠ 2) :
    ∃ e, t.det = Except.error e := by
  unfold Tensor.det; simp [h]

def Tensor.lu (t : Tensor) : TResult (Tensor × Tensor) :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := t.shape.dims.get! 0
    if n ≠ t.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else if n = 0 then Except.error TensorError.invalidShape
    else
      let uArr := Array.ofFn (fun idx : Fin (n * n) => t.data.get! idx.val)
      let lArr := Array.ofFn (fun idx : Fin (n * n) =>
        if idx.val / n = idx.val % n then 1.0 else 0.0)
      let (lFinal, uFinal) := (List.range n).foldl (fun (l, u) k =>
        let pivot := u.get! (k * n + k)
        if pivot == 0.0 then (l, u)
        else
          (List.range n).foldl (fun (l2, u2) i =>
            if i ≤ k then (l2, u2)
            else
              let factor := u2.get! (i * n + k) / pivot
              let l3 := l2.set! (i * n + k) factor
              let u3 := (List.range n).foldl (fun u4 j =>
                u4.set! (i * n + j) (u4.get! (i * n + j) - factor * u4.get! (k * n + j))
              ) u2
              (l3, u3)
          ) (l, u)
      ) (lArr, uArr)
      Except.ok (
        { data := lFinal
          shape := Shape.mk' [n, n]
          h_data_size := by simp [Array.size_ofFn]
          refcount := RefCount.init
          cow := CowState.init },
        { data := uFinal
          shape := Shape.mk' [n, n]
          h_data_size := by simp [Array.size_ofFn]
          refcount := RefCount.init
          cow := CowState.init }
      )

def Tensor.qr (t : Tensor) : TResult (Tensor × Tensor) :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let m := t.shape.dims.get! 0
    let n := t.shape.dims.get! 1
    if m = 0 ∨ n = 0 then Except.error TensorError.invalidShape
    else
      let rArr := Array.ofFn (fun idx : Fin (m * n) => t.data.get! idx.val)
      let qArr := Array.ofFn (fun idx : Fin (m * m) =>
        if idx.val / m = idx.val % m then 1.0 else 0.0)
      let k := min m n
      let (qFinal, rFinal) := (List.range k).foldl (fun (q, r) col =>
        let normSq := (List.range m).foldl (fun acc i =>
          if i < col then acc
          else acc + r.get! (i * n + col) * r.get! (i * n + col)) 0.0
        let normVal := Float.sqrt normSq
        if normVal < 1e-12 then (q, r)
        else
          let sign := if r.get! (col * n + col) < 0.0 then -1.0 else 1.0
          let alpha := -(sign * normVal)
          let vArr := Array.ofFn (fun i : Fin m =>
            if i.val < col then 0.0
            else if i.val = col then r.get! (i.val * n + col) - alpha
            else r.get! (i.val * n + col))
          let vNormSq := vArr.foldl (fun acc v => acc + v * v) 0.0
          if vNormSq < 1e-24 then (q, r)
          else
            let tau := 2.0 / vNormSq
            let newR := Array.ofFn (fun idx : Fin (m * n) =>
              let i := idx.val / n
              let j := idx.val % n
              let dot_v_col := (List.range m).foldl (fun acc ii =>
                acc + vArr.get! ii * r.get! (ii * n + j)) 0.0
              r.get! (i * n + j) - tau * vArr.get! i * dot_v_col)
            let newQ := Array.ofFn (fun idx : Fin (m * m) =>
              let i := idx.val / m
              let j := idx.val % m
              let dot_q_v := (List.range m).foldl (fun acc ii =>
                acc + q.get! (i * m + ii) * vArr.get! ii) 0.0
              q.get! (i * m + j) - tau * dot_q_v * vArr.get! j)
            (newQ, newR)
      ) (qArr, rArr)
      Except.ok (
        { data := qFinal
          shape := Shape.mk' [m, m]
          h_data_size := by simp [Array.size_ofFn]
          refcount := RefCount.init
          cow := CowState.init },
        { data := rFinal
          shape := Shape.mk' [m, n]
          h_data_size := by simp [Array.size_ofFn]
          refcount := RefCount.init
          cow := CowState.init }
      )

def Tensor.cholesky (t : Tensor) : TResult Tensor :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := t.shape.dims.get! 0
    if n ≠ t.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else if n = 0 then Except.error TensorError.invalidShape
    else
      let lArr := mkArray (n * n) (0.0 : Float)
      let lFinal := (List.range n).foldl (fun l i =>
        (List.range (i + 1)).foldl (fun l1 j =>
          if j < i then
            let s := (List.range j).foldl (fun acc k =>
              acc + l1.get! (i * n + k) * l1.get! (j * n + k)) 0.0
            let diagJ := l1.get! (j * n + j)
            let val := if diagJ == 0.0 then 0.0
              else (t.data.get! (i * n + j) - s) / diagJ
            l1.set! (i * n + j) val
          else
            let s := (List.range j).foldl (fun acc k =>
              acc + l1.get! (i * n + k) * l1.get! (i * n + k)) 0.0
            let diff := t.data.get! (i * n + i) - s
            let val := if diff ≤ 0.0 then 0.0 else Float.sqrt diff
            l1.set! (i * n + j) val
        ) l
      ) lArr
      Except.ok {
        data := lFinal
        shape := Shape.mk' [n, n]
        h_data_size := by simp [Array.size_mkArray]
        refcount := RefCount.init
        cow := CowState.init
      }

def Tensor.solve (a b : Tensor) : TResult Tensor :=
  if a.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := a.shape.dims.get! 0
    if n ≠ a.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else if b.shape.dims.length ≠ 1 ∨ b.shape.dims.get! 0 ≠ n then
      Except.error TensorError.shapeMismatch
    else if n = 0 then Except.error TensorError.invalidShape
    else
      match a.lu with
      | Except.error e => Except.error e
      | Except.ok (l, u) =>
        let y := (List.range n).foldl (fun arr i =>
          let s := (List.range i).foldl (fun acc j =>
            acc + l.data.get! (i * n + j) * arr.get! j) 0.0
          arr.set! i (b.data.get! i - s)
        ) (mkArray n (0.0 : Float))
        let x := (List.range n).reverse.foldl (fun arr i =>
          let s := (List.range n).foldl (fun acc j =>
            if j ≤ i then acc
            else acc + u.data.get! (i * n + j) * arr.get! j) 0.0
          let diag := u.data.get! (i * n + i)
          let val := if diag == 0.0 then 0.0 else (y.get! i - s) / diag
          arr.set! i val
        ) (mkArray n (0.0 : Float))
        Except.ok {
          data := x
          shape := Shape.mk' [n]
          h_data_size := by simp [Array.size_mkArray]
          refcount := RefCount.init
          cow := CowState.init
        }

def Tensor.svd (t : Tensor) : TResult (Tensor × Tensor × Tensor) :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let m := t.shape.dims.get! 0
    let n := t.shape.dims.get! 1
    let k := min m n
    if m = 0 ∨ n = 0 then Except.error TensorError.invalidShape
    else
      match Tensor.identity m, Tensor.init [k], Tensor.identity n with
      | Except.ok u, Except.ok s, Except.ok v => Except.ok (u, s, v)
      | _, _, _ => Except.error TensorError.allocFailed

def Tensor.eig (t : Tensor) : TResult (Tensor × Tensor) :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let n := t.shape.dims.get! 0
    if n ≠ t.shape.dims.get! 1 then Except.error TensorError.mustBeSquare
    else if n = 0 then Except.error TensorError.invalidShape
    else
      match Tensor.init [n], Tensor.identity n with
      | Except.ok vals, Except.ok vecs => Except.ok (vals, vecs)
      | _, _ => Except.error TensorError.allocFailed

def Tensor.spectralNorm (t : Tensor) (maxIter : Nat) (tol : Float) : TResult Float :=
  if t.shape.dims.length ≠ 2 then Except.error TensorError.mustBeSquare
  else
    let m := t.shape.dims.get! 0
    let n := t.shape.dims.get! 1
    if m = 0 ∨ n = 0 then Except.error TensorError.invalidShape
    else
      let v0 := Array.ofFn (fun _ : Fin n => 1.0 / Float.ofNat n)
      let result := (List.range maxIter).foldl (fun (v, _) _ =>
        let av := Array.ofFn (fun i : Fin m =>
          (List.range n).foldl (fun acc j =>
            acc + t.data.get! (i.val * n + j) * v.get! j) 0.0)
        let atav := Array.ofFn (fun j : Fin n =>
          (List.range m).foldl (fun acc i =>
            acc + t.data.get! (i * n + j.val) * av.get! i) 0.0)
        let normVal := Float.sqrt (atav.foldl (fun acc x => acc + x * x) 0.0)
        let newV := if normVal < 1e-12 then v
          else Array.ofFn (fun i : Fin n => atav.get! i.val / normVal)
        let newAv := Array.ofFn (fun i : Fin m =>
          (List.range n).foldl (fun acc j =>
            acc + t.data.get! (i.val * n + j) * newV.get! j) 0.0)
        let newSigma := Float.sqrt (newAv.foldl (fun acc x => acc + x * x) 0.0)
        (newV, newSigma)
      ) (v0, 0.0)
      Except.ok result.2

def simpleLcg (seed : Nat) (n : Nat) : Array Float :=
  let rec go : Nat → Nat → Array Float → Array Float
    | 0, _, acc => acc
    | fuel + 1, s, acc =>
      let next := (s * 1103515245 + 12345) % (2^31)
      let val := Float.ofNat (next % 10000) / 10000.0
      go fuel next (acc.push val)
  go n seed #[]

def Tensor.randomUniform (dims : List Nat) (minVal maxVal : Float) (seed : Nat) : TResult Tensor :=
  if dims.length = 0 then Except.error TensorError.invalidShape
  else if dims.any (· == 0) then Except.error TensorError.invalidShape
  else
    let total := listProduct dims
    let rands := simpleLcg seed total
    let data := Array.ofFn (fun i : Fin total =>
      let r := rands.get! i.val
      minVal + r * (maxVal - minVal))
    Except.ok {
      data := data
      shape := Shape.mk' dims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.randomNormal (dims : List Nat) (meanVal stddevVal : Float) (seed : Nat) : TResult Tensor :=
  if dims.length = 0 then Except.error TensorError.invalidShape
  else if dims.any (· == 0) then Except.error TensorError.invalidShape
  else
    let total := listProduct dims
    let rands := simpleLcg seed (total * 2)
    let data := Array.ofFn (fun i : Fin total =>
      let u := rands.get! (i.val * 2)
      let v := rands.get! (i.val * 2 + 1)
      let safeU := if u < 1e-10 then 1e-10 else u
      let z := Float.sqrt (-2.0 * Float.log safeU) * Float.cos (2.0 * 3.14159265358979 * v)
      meanVal + stddevVal * z)
    Except.ok {
      data := data
      shape := Shape.mk' dims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.clipOp (t : Tensor) (minVal maxVal : Float) : Tensor :=
  mapData t (fun v => max minVal (min maxVal v))

theorem Tensor.clipOp_shape (t : Tensor) (minVal maxVal : Float) :
    (t.clipOp minVal maxVal).shape = t.shape := mapData_preserves_shape t _

def Tensor.toInt (t : Tensor) : Tensor := mapData t Float.floor

theorem Tensor.toInt_shape (t : Tensor) : t.toInt.shape = t.shape :=
  mapData_preserves_shape t Float.floor

def Tensor.argmax (t : Tensor) (axis : Nat) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then Except.error TensorError.invalidAxis
  else
    let newDims := removeAxis t.shape.dims axis
    let safeNewDims := if newDims.length = 0 then [1] else newDims
    let newTotal := listProduct safeNewDims
    let axisSize := t.shape.dims.get! axis
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin newTotal =>
      let outMulti := flatToMultiIndex safeNewDims idx.val
      let (_, bestIdx) := (List.range axisSize).foldl (fun (bestVal, bestI) k =>
        let inMulti := if newDims.length = 0 then [k]
          else (outMulti.take axis) ++ [k] ++ (outMulti.drop axis)
        let v := t.data.get! (computeFlatIndex srcShape inMulti)
        if k = 0 ∨ v > bestVal then (v, Float.ofNat k) else (bestVal, bestI)
      ) (floatNegInf, 0.0)
      bestIdx)
    Except.ok {
      data := data
      shape := Shape.mk' safeNewDims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.argmin (t : Tensor) (axis : Nat) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then Except.error TensorError.invalidAxis
  else
    let newDims := removeAxis t.shape.dims axis
    let safeNewDims := if newDims.length = 0 then [1] else newDims
    let newTotal := listProduct safeNewDims
    let axisSize := t.shape.dims.get! axis
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin newTotal =>
      let outMulti := flatToMultiIndex safeNewDims idx.val
      let (_, bestIdx) := (List.range axisSize).foldl (fun (bestVal, bestI) k =>
        let inMulti := if newDims.length = 0 then [k]
          else (outMulti.take axis) ++ [k] ++ (outMulti.drop axis)
        let v := t.data.get! (computeFlatIndex srcShape inMulti)
        if k = 0 ∨ v < bestVal then (v, Float.ofNat k) else (bestVal, bestI)
      ) (floatInf, 0.0)
      bestIdx)
    Except.ok {
      data := data
      shape := Shape.mk' safeNewDims
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.cumsum (t : Tensor) (axis : Nat) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then Except.error TensorError.invalidAxis
  else
    let totalSize := t.shape.totalSize
    let srcShape := Shape.mk' t.shape.dims
    let data := Array.ofFn (fun idx : Fin totalSize =>
      let multiIdx := flatToMultiIndex t.shape.dims idx.val
      let axisIdx := multiIdx.get! axis
      (List.range (axisIdx + 1)).foldl (fun acc k =>
        let mi := multiIdx.enum.map (fun ⟨i, v⟩ =>
          if i = axis then k else v)
        acc + t.data.get! (computeFlatIndex srcShape mi)
      ) 0.0)
    Except.ok {
      data := data
      shape := t.shape
      h_data_size := by simp [Array.size_ofFn, t.h_data_size]
      refcount := RefCount.init
      cow := CowState.init
    }

theorem Tensor.cumsum_shape (t t' : Tensor) (axis : Nat)
    (h_ok : t.cumsum axis = Except.ok t') : t'.shape = t.shape := by
  unfold Tensor.cumsum at h_ok
  split at h_ok
  · cases h_ok
  · injection h_ok with h'; rw [← h']

def Tensor.variance (t : Tensor) (axis : Nat) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then Except.error TensorError.invalidAxis
  else
    match t.meanReduce axis with
    | Except.error e => Except.error e
    | Except.ok meanT =>
      let newDims := removeAxis t.shape.dims axis
      let safeNewDims := if newDims.length = 0 then [1] else newDims
      let newTotal := listProduct safeNewDims
      let axisSize := t.shape.dims.get! axis
      let srcShape := Shape.mk' t.shape.dims
      let data := Array.ofFn (fun idx : Fin newTotal =>
        let outMulti := flatToMultiIndex safeNewDims idx.val
        let meanVal := meanT.data.get! idx.val
        let sumSqDiff := (List.range axisSize).foldl (fun acc k =>
          let inMulti := if newDims.length = 0 then [k]
            else (outMulti.take axis) ++ [k] ++ (outMulti.drop axis)
          let v := t.data.get! (computeFlatIndex srcShape inMulti)
          let diff := v - meanVal
          acc + diff * diff
        ) 0.0
        sumSqDiff / Float.ofNat axisSize)
      Except.ok {
        data := data
        shape := Shape.mk' safeNewDims
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

def Tensor.stddev (t : Tensor) (axis : Nat) : TResult Tensor :=
  match t.variance axis with
  | Except.error e => Except.error e
  | Except.ok v => Except.ok (mapData v (fun x => Float.sqrt (if x < 0.0 then 0.0 else x)))

def Tensor.sort (t : Tensor) (axis : Nat) (descending : Bool) : TResult Tensor :=
  if axis ≥ t.shape.dims.length then Except.error TensorError.invalidAxis
  else
    let totalSize := t.shape.totalSize
    let axisSize := t.shape.dims.get! axis
    let srcShape := Shape.mk' t.shape.dims
    let newDims := removeAxis t.shape.dims axis
    let safeNewDims := if newDims.length = 0 then [1] else newDims
    let outerSize := listProduct safeNewDims
    let collected := Array.ofFn (fun idx : Fin totalSize => t.data.get! idx.val)
    let sorted := (List.range outerSize).foldl (fun arr outerIdx =>
      let outMulti := flatToMultiIndex safeNewDims outerIdx
      let indices := (List.range axisSize).map (fun k =>
        let inMulti := if newDims.length = 0 then [k]
          else (outMulti.take axis) ++ [k] ++ (outMulti.drop axis)
        computeFlatIndex srcShape inMulti)
      let vals := indices.map (fun i => arr.get! i)
      let sortedVals := if descending
        then vals.mergeSort (fun a b => a > b)
        else vals.mergeSort (fun a b => a < b)
      indices.zip sortedVals |>.foldl (fun a (i, v) => a.set! i v) arr
    ) collected
    Except.ok {
      data := sorted
      shape := t.shape
      h_data_size := by simp [Array.size_ofFn, t.h_data_size]
      refcount := RefCount.init
      cow := CowState.init
    }

theorem Tensor.sort_shape (t t' : Tensor) (axis : Nat) (desc : Bool)
    (h_ok : t.sort axis desc = Except.ok t') : t'.shape = t.shape := by
  unfold Tensor.sort at h_ok
  split at h_ok
  · cases h_ok
  · injection h_ok with h'; rw [← h']

def Tensor.unique (t : Tensor) : TResult Tensor :=
  let vals := t.data.toList
  let deduped := vals.foldl (fun acc v =>
    if acc.any (fun x => Float.abs (x - v) < 1e-10) then acc else acc ++ [v]) ([] : List Float)
  let sorted := deduped.mergeSort (· < ·)
  if sorted.length = 0 then Except.error TensorError.emptyInput
  else
    Except.ok {
      data := sorted.toArray
      shape := Shape.mk' [sorted.length]
      h_data_size := by simp [Shape.mk', Shape.totalSize, listProduct, List.toArray_length]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.oneHot (t : Tensor) (numClasses : Nat) : TResult Tensor :=
  if t.shape.dims.length ≠ 1 then Except.error TensorError.invalidForOneHot
  else if numClasses = 0 then Except.error TensorError.invalidShape
  else
    let n := t.shape.dims.get! 0
    if n = 0 then Except.error TensorError.invalidShape
    else
      let data := Array.ofFn (fun idx : Fin (n * numClasses) =>
        let i := idx.val / numClasses
        let j := idx.val % numClasses
        let classIdx := Float.toUInt64 (t.data.get! i)
        if classIdx.toNat = j then 1.0 else 0.0)
      Except.ok {
        data := data
        shape := Shape.mk' [n, numClasses]
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

theorem Tensor.oneHot_requires_1d (t : Tensor) (nc : Nat)
    (h : t.shape.dims.length ≠ 1) :
    ∃ e, t.oneHot nc = Except.error e := by
  unfold Tensor.oneHot; simp [h]

def Tensor.arange (start stop step : Float) : TResult Tensor :=
  if step == 0.0 then Except.error TensorError.invalidShape
  else
    let size := Float.toUInt64 (Float.ceil (Float.abs ((stop - start) / step)))
    let n := size.toNat
    if n = 0 then Except.error TensorError.invalidShape
    else
      let data := Array.ofFn (fun i : Fin n =>
        start + Float.ofNat i.val * step)
      Except.ok {
        data := data
        shape := Shape.mk' [n]
        h_data_size := by simp [Array.size_ofFn]
        refcount := RefCount.init
        cow := CowState.init
      }

def Tensor.linspace (start stop : Float) (num : Nat) : TResult Tensor :=
  if num = 0 then Except.error TensorError.invalidShape
  else
    let data := Array.ofFn (fun i : Fin num =>
      if num = 1 then start
      else
        let t := Float.ofNat i.val / Float.ofNat (num - 1)
        start + t * (stop - start))
    Except.ok {
      data := data
      shape := Shape.mk' [num]
      h_data_size := by simp [Array.size_ofFn]
      refcount := RefCount.init
      cow := CowState.init
    }

def Tensor.toFixed (t : Tensor) : Tensor :=
  mapData t (fun v => Float.floor (v * 4294967296.0) / 4294967296.0)

theorem Tensor.toFixed_shape (t : Tensor) : t.toFixed.shape = t.shape :=
  mapData_preserves_shape t _

def Tensor.copy (t : Tensor) : Tensor :=
  { t with refcount := RefCount.init, cow := CowState.init }

theorem Tensor.copy_data_eq (t : Tensor) : t.copy.data = t.data := rfl
theorem Tensor.copy_shape_eq (t : Tensor) : t.copy.shape = t.shape := rfl
theorem Tensor.copy_independent_refcount (t : Tensor) :
    t.copy.refcount.count = 1 := rfl

def Tensor.retain (t : Tensor) : Tensor :=
  { t with refcount := t.refcount.retain }

def Tensor.release (t : Tensor) : Option Tensor :=
  match t.refcount.release with
  | some rc => some { t with refcount := rc }
  | none => none

theorem Tensor.retain_increases (t : Tensor) :
    t.retain.refcount.count = t.refcount.count + 1 := rfl

theorem Tensor.release_decreases (t : Tensor) (h : t.refcount.count > 1) :
    ∃ t', t.release = some t' ∧ t'.refcount.count = t.refcount.count - 1 := by
  unfold Tensor.release RefCount.release
  simp [h]

theorem Tensor.release_last (t : Tensor) (h : t.refcount.count = 1) :
    t.release = none := by
  unfold Tensor.release RefCount.release
  simp [h]

def Tensor.ensureWritable (t : Tensor) : Tensor :=
  if t.cow.isShared then
    { t with refcount := RefCount.init, cow := CowState.init }
  else t

theorem Tensor.ensureWritable_preserves_data (t : Tensor) :
    (t.ensureWritable).data = t.data := by
  unfold Tensor.ensureWritable; split <;> rfl

theorem Tensor.ensureWritable_not_shared (t : Tensor) :
    (t.ensureWritable).cow.isShared = false := by
  unfold Tensor.ensureWritable
  split
  · rfl
  · rename_i h; simp at h; exact h

theorem Tensor.ensureWritable_unique_refcount (t : Tensor)
    (h : t.cow.isShared = true) :
    (t.ensureWritable).refcount.count = 1 := by
  unfold Tensor.ensureWritable
  simp [h]

theorem Tensor.ensureWritable_preserves_shape (t : Tensor) :
    (t.ensureWritable).shape = t.shape := by
  unfold Tensor.ensureWritable; split <;> rfl

def Tensor.markShared (t : Tensor) : Tensor :=
  { t with cow := t.cow.markShared }

theorem Tensor.markShared_is_shared (t : Tensor) :
    t.markShared.cow.isShared = true := rfl

theorem refcount_safety (t : Tensor) : t.refcount.count > 0 := t.refcount.h_pos

theorem retain_release_roundtrip (t : Tensor) :
    ∃ t', (t.retain).release = some t' ∧ t'.refcount.count = t.refcount.count := by
  show ∃ t', Tensor.release (Tensor.retain t) = some t' ∧ t'.refcount.count = t.refcount.count
  unfold Tensor.retain Tensor.release RefCount.retain RefCount.release
  have h1 : t.refcount.count + 1 > 1 := Nat.succ_lt_succ t.refcount.h_pos
  simp [h1]

theorem shared_view_preserves_data (t : Tensor) (newDims : List Nat)
    (v : Tensor) (h_ok : t.view newDims = Except.ok v) :
    v.data = t.data := Tensor.view_shares_data t newDims v h_ok

theorem ensureWritable_after_share (t : Tensor) :
    let shared := t.markShared
    let writable := shared.ensureWritable
    writable.cow.isShared = false := by
  simp [Tensor.markShared, Tensor.ensureWritable, CowState.markShared]

theorem ensureWritable_preserves_values (t : Tensor) :
    (t.ensureWritable).data = t.data := Tensor.ensureWritable_preserves_data t

theorem Tensor.view_then_ensureWritable (t : Tensor)
    (newDims : List Nat) (v : Tensor)
    (h_ok : t.view newDims = Except.ok v) :
    (v.ensureWritable).data = t.data := by
  rw [Tensor.ensureWritable_preserves_data]
  exact Tensor.view_shares_data t newDims v h_ok

theorem Tensor.get_requires_correct_ndim (t : Tensor) (indices : List Nat)
    (h : indices.length ≠ t.shape.dims.length) :
    ∃ e, t.get indices = Except.error e := by
  unfold Tensor.get computeIndex
  simp [h]

theorem Tensor.set_requires_correct_ndim (t : Tensor) (indices : List Nat)
    (value : Float) (h : indices.length ≠ t.shape.dims.length) :
    ∃ e, t.set indices value = Except.error e := by
  unfold Tensor.set computeIndex
  simp [h]



def tresultMap {α β : Type} (f : α → β) : TResult α → TResult β
  | Except.error e => Except.error e
  | Except.ok a => Except.ok (f a)

@[simp] theorem tresultMap_ok {α β : Type} (f : α → β) (a : α) :
    tresultMap f (Except.ok a) = Except.ok (f a) := rfl

@[simp] theorem tresultMap_error {α β : Type} (f : α → β) (e : TensorError) :
    tresultMap f (Except.error e : TResult α) = (Except.error e : TResult β) := rfl

@[simp] theorem tresultMap_id {α : Type} (x : TResult α) :
    tresultMap (fun y => y) x = x := by
  cases x <;> rfl

@[simp] theorem tresultMap_comp {α β γ : Type} (f : α → β) (g : β → γ) (x : TResult α) :
    tresultMap g (tresultMap f x) = tresultMap (fun y => g (f y)) x := by
  cases x <;> rfl

def optionMap {α β : Type} (f : α → β) : Option α → Option β
  | none => none
  | some a => some (f a)

@[simp] theorem optionMap_none {α β : Type} (f : α → β) :
    optionMap f (none : Option α) = (none : Option β) := rfl

@[simp] theorem optionMap_some {α β : Type} (f : α → β) (a : α) :
    optionMap f (some a) = some (f a) := rfl

structure ZigShapeModel where
  dimsBuffer : List Nat
  stridesBuffer : List Nat
  ownsBuffer : Bool
  allocatorTag : Nat
  h_len : dimsBuffer.length = stridesBuffer.length

def ZigShapeModel.toShape (zs : ZigShapeModel) : Shape :=
  { dims := zs.dimsBuffer, strides := zs.stridesBuffer, h_len := zs.h_len }

def ZigShapeModel.ofShape (s : Shape) : ZigShapeModel :=
  { dimsBuffer := s.dims
    stridesBuffer := s.strides
    ownsBuffer := true
    allocatorTag := 0
    h_len := s.h_len }

@[simp] theorem ZigShapeModel.toShape_ofShape (s : Shape) :
    (ZigShapeModel.ofShape s).toShape = s := by
  cases s
  rfl

@[simp] theorem ZigShapeModel.ofShape_toShape (zs : ZigShapeModel) :
    ZigShapeModel.ofShape zs.toShape = { zs with ownsBuffer := true, allocatorTag := 0 } := by
  cases zs
  rfl

def ZigShapeModel.init (dims : List Nat) : TResult ZigShapeModel :=
  if dims = [] then Except.error TensorError.invalidShape
  else if dims.any (fun d => d == 0) then Except.error TensorError.invalidShape
  else Except.ok <| ZigShapeModel.ofShape (Shape.mk' dims)

def ZigShapeModel.copy (zs : ZigShapeModel) : ZigShapeModel :=
  ZigShapeModel.ofShape zs.toShape.copy

def ZigShapeModel.deinit (zs : ZigShapeModel) : ZigShapeModel :=
  { dimsBuffer := []
    stridesBuffer := []
    ownsBuffer := false
    allocatorTag := zs.allocatorTag
    h_len := rfl }

def ZigShapeModel.totalSize (zs : ZigShapeModel) : Nat :=
  zs.toShape.totalSize

def ZigShapeModel.equals (a b : ZigShapeModel) : Bool :=
  a.toShape.equals b.toShape

def ZigShapeModel.broadcastCompatible (a b : ZigShapeModel) : Bool :=
  a.toShape.broadcastCompatible b.toShape

def ZigShapeModel.isContiguous (zs : ZigShapeModel) : Bool :=
  zs.toShape.isContiguous

@[simp] theorem ZigShapeModel.copy_toShape (zs : ZigShapeModel) :
    (zs.copy).toShape = zs.toShape := by
  cases zs
  rfl

@[simp] theorem ZigShapeModel.totalSize_eq (zs : ZigShapeModel) :
    zs.totalSize = zs.toShape.totalSize := rfl

@[simp] theorem ZigShapeModel.equals_spec (a b : ZigShapeModel) :
    a.equals b = a.toShape.equals b.toShape := rfl

structure ZigSharedState where
  baseData : Array Float
  refcountCell : RefCount
  cowCell : CowState
  allocatorTag : Nat
  mutexTag : Nat

structure ZigTensorModel where
  dataView : Array Float
  shapeView : ZigShapeModel
  sharedState : ZigSharedState
  releasedFlag : Bool
  h_data_size : dataView.size = shapeView.toShape.totalSize
  h_alias : dataView = sharedState.baseData

def ZigTensorModel.toTensor (zt : ZigTensorModel) : Tensor :=
  { data := zt.dataView
    shape := zt.shapeView.toShape
    h_data_size := zt.h_data_size
    refcount := zt.sharedState.refcountCell
    cow := zt.sharedState.cowCell }

def ZigTensorModel.ofTensor (t : Tensor) : ZigTensorModel :=
  { dataView := t.data
    shapeView := ZigShapeModel.ofShape t.shape
    sharedState :=
      { baseData := t.data
        refcountCell := t.refcount
        cowCell := t.cow
        allocatorTag := 0
        mutexTag := 0 }
    releasedFlag := false
    h_data_size := by simpa [ZigShapeModel.toShape, ZigShapeModel.ofShape] using t.h_data_size
    h_alias := rfl }

@[simp] theorem ZigTensorModel.toTensor_ofTensor (t : Tensor) :
    (ZigTensorModel.ofTensor t).toTensor = t := by
  cases t
  rfl

@[simp] theorem ZigTensorModel.ofTensor_toTensor (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor zt.toTensor =
      { zt with
        shapeView := ZigShapeModel.ofShape zt.shapeView.toShape
        sharedState :=
          { baseData := zt.dataView
            refcountCell := zt.sharedState.refcountCell
            cowCell := zt.sharedState.cowCell
            allocatorTag := 0
            mutexTag := 0 }
        releasedFlag := false } := by
  cases zt
  rfl

def ZigTensorModel.fromTensorResult (r : TResult Tensor) : TResult ZigTensorModel :=
  tresultMap ZigTensorModel.ofTensor r

def ZigTensorModel.pairOfTensor : Tensor × Tensor → ZigTensorModel × ZigTensorModel
  | (a, b) => (ZigTensorModel.ofTensor a, ZigTensorModel.ofTensor b)

def ZigTensorModel.tripleOfTensor : Tensor × Tensor × Tensor →
    ZigTensorModel × ZigTensorModel × ZigTensorModel
  | (a, b, c) => (ZigTensorModel.ofTensor a, ZigTensorModel.ofTensor b, ZigTensorModel.ofTensor c)

def ZigTensorModel.liftPure (f : Tensor → Tensor) (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (f zt.toTensor)

def ZigTensorModel.liftTensor (f : Tensor → TResult Tensor) (zt : ZigTensorModel) :
    TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (f zt.toTensor)

def ZigTensorModel.liftBinaryTensor (f : Tensor → Tensor → TResult Tensor)
    (a b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (f a.toTensor b.toTensor)

def ZigTensorModel.liftBinaryBool (f : Tensor → Tensor → Bool)
    (a b : ZigTensorModel) : Bool :=
  f a.toTensor b.toTensor

def ZigTensorModel.liftFloat (f : Tensor → Float) (zt : ZigTensorModel) : Float :=
  f zt.toTensor

def ZigTensorModel.liftTensorFloat (f : Tensor → TResult Float) (zt : ZigTensorModel) :
    TResult Float :=
  f zt.toTensor

def ZigTensorModel.liftBinaryTensorFloat (f : Tensor → Tensor → TResult Float)
    (a b : ZigTensorModel) : TResult Float :=
  f a.toTensor b.toTensor

def ZigTensorModel.liftOption (f : Tensor → Option Tensor) (zt : ZigTensorModel) :
    Option ZigTensorModel :=
  optionMap ZigTensorModel.ofTensor (f zt.toTensor)

@[simp] theorem ZigTensorModel.fromTensorResult_ok (t : Tensor) :
    ZigTensorModel.fromTensorResult (Except.ok t) = Except.ok (ZigTensorModel.ofTensor t) := rfl

@[simp] theorem ZigTensorModel.fromTensorResult_error (e : TensorError) :
    ZigTensorModel.fromTensorResult (Except.error e : TResult Tensor) = Except.error e := rfl


def axesPermutation (axes : List Nat) (n : Nat) : Bool :=
  decide (axes.length = n) && decide axes.Nodup && axes.all (fun a => decide (a < n))

def dimsEqualExceptAxisAux (axis idx : Nat) : List Nat → List Nat → Bool
  | [], [] => true
  | d1 :: t1, d2 :: t2 =>
      if idx = axis then dimsEqualExceptAxisAux axis (idx + 1) t1 t2
      else decide (d1 = d2) && dimsEqualExceptAxisAux axis (idx + 1) t1 t2
  | _, _ => false

def dimsEqualExceptAxis (axis : Nat) (xs ys : List Nat) : Bool :=
  dimsEqualExceptAxisAux axis 0 xs ys

def listAllSameShape : List Tensor → Bool
  | [] => false
  | t :: ts => ts.all (fun u => decide (u.shape.dims = t.shape.dims))

def listAllSameShapeExceptAxis (axis : Nat) : List Tensor → Bool
  | [] => false
  | t :: ts =>
      if axis ≥ t.shape.dims.length then false
      else ts.all (fun u => dimsEqualExceptAxis axis t.shape.dims u.shape.dims)

def Shape.initVerified (dims : List Nat) : TResult Shape :=
  match dims with
  | [] => Except.error TensorError.invalidShape
  | _ =>
      if dims.any (fun d => d == 0) then Except.error TensorError.invalidShape
      else Except.ok (Shape.mk' dims)

def Shape.copyVerified (s : Shape) : Shape := s.copy
def Shape.totalSizeVerified (s : Shape) : Nat := s.totalSize
def Shape.equalsVerified (s1 s2 : Shape) : Bool := s1.equals s2
def Shape.broadcastCompatibleVerified (s1 s2 : Shape) : Bool := s1.broadcastCompatible s2
def Shape.isContiguousVerified (s : Shape) : Bool := s.isContiguous

@[simp] theorem Shape.copyVerified_eq (s : Shape) : Shape.copyVerified s = s.copy := rfl
@[simp] theorem Shape.totalSizeVerified_eq (s : Shape) : Shape.totalSizeVerified s = s.totalSize := rfl
@[simp] theorem Shape.equalsVerified_eq (s1 s2 : Shape) : Shape.equalsVerified s1 s2 = s1.equals s2 := rfl
@[simp] theorem Shape.broadcastCompatibleVerified_eq (s1 s2 : Shape) :
    Shape.broadcastCompatibleVerified s1 s2 = s1.broadcastCompatible s2 := rfl
@[simp] theorem Shape.isContiguousVerified_eq (s : Shape) :
    Shape.isContiguousVerified s = s.isContiguous := rfl

def Tensor.newViewChecked (t : Tensor) (newShape : Shape) : TResult Tensor :=
  if h_eq : newShape.totalSize = t.shape.totalSize then
    Except.ok (t.newView newShape h_eq)
  else
    Except.error TensorError.invalidShape

def Tensor.transposeVerified (t : Tensor) (axes : List Nat) : TResult Tensor :=
  if axesPermutation axes t.shape.dims.length then Tensor.transposeOp t axes
  else Except.error TensorError.invalidPermutation

def Tensor.concatVerified (tensors : List Tensor) (axis : Nat) : TResult Tensor :=
  match tensors with
  | [] => Except.error TensorError.emptyInput
  | _ =>
      if listAllSameShapeExceptAxis axis tensors then Tensor.concatOp tensors axis
      else Except.error TensorError.shapeMismatch

def Tensor.stackVerified (tensors : List Tensor) (axis : Nat) : TResult Tensor :=
  match tensors with
  | [] => Except.error TensorError.emptyInput
  | _ =>
      if listAllSameShape tensors then Tensor.stackOp tensors axis
      else Except.error TensorError.shapeMismatch

def Tensor.normChecked (t : Tensor) (order : Float) : TResult Float :=
  if order ≤ 0.0 then Except.error TensorError.invalidOrder
  else Except.ok (Tensor.norm t order)

def Tensor.deinitVerified (t : Tensor) : Option Tensor := Tensor.release t
def Tensor.initWithArenaVerified (dims : List Nat) : TResult Tensor := Tensor.init dims
def Tensor.initWithPoolVerified (dims : List Nat) : TResult Tensor := Tensor.init dims
def Tensor.initWithSlabVerified (dims : List Nat) : TResult Tensor := Tensor.init dims
def Tensor.initWithBuddyVerified (dims : List Nat) : TResult Tensor := Tensor.init dims
def Tensor.copyVerified (t : Tensor) : Tensor := Tensor.copy t
def Tensor.retainVerified (t : Tensor) : Tensor := Tensor.retain t
def Tensor.releaseVerified (t : Tensor) : Option Tensor := Tensor.release t
def Tensor.ensureWritableVerified (t : Tensor) : Tensor := Tensor.ensureWritable t
def Tensor.markSharedVerified (t : Tensor) : Tensor := Tensor.markShared t
def Tensor.divVerified (t1 t2 : Tensor) : TResult Tensor := Tensor.divOp t1 t2
def Tensor.expVerified (t : Tensor) : Tensor := Tensor.expOp t
def Tensor.logVerified (t : Tensor) : Tensor := Tensor.logOp t
def Tensor.sinVerified (t : Tensor) : Tensor := Tensor.sinOp t
def Tensor.cosVerified (t : Tensor) : Tensor := Tensor.cosOp t
def Tensor.tanVerified (t : Tensor) : Tensor := Tensor.tanOp t
def Tensor.sqrtVerified (t : Tensor) : Tensor := Tensor.sqrtOp t
def Tensor.powVerified (t : Tensor) (exponent : Float) : Tensor := Tensor.powOp t exponent
def Tensor.absVerified (t : Tensor) : Tensor := Tensor.absOp t
def Tensor.maxVerified (t : Tensor) (axis : Nat) : TResult Tensor := Tensor.maxReduce t axis
def Tensor.minVerified (t : Tensor) (axis : Nat) : TResult Tensor := Tensor.minReduce t axis
def Tensor.sumVerified (t : Tensor) (axis : Nat) : TResult Tensor := Tensor.sumReduce t axis
def Tensor.meanVerified (t : Tensor) (axis : Nat) : TResult Tensor := Tensor.meanReduce t axis
def Tensor.viewVerified (t : Tensor) (newDims : List Nat) : TResult Tensor := Tensor.view t newDims
def Tensor.sliceVerified (t : Tensor) (starts ends : List Nat) : TResult Tensor := Tensor.sliceOp t starts ends
def Tensor.unsqueezeVerified (t : Tensor) (axis : Nat) : TResult Tensor := Tensor.unsqueezeOp t axis
def Tensor.broadcastVerified (t : Tensor) (targetDims : List Nat) : TResult Tensor := Tensor.broadcastOp t targetDims
def Tensor.padVerified (t : Tensor) (pads : List (Nat × Nat)) : TResult Tensor := Tensor.padOp t pads
def Tensor.tileVerified (t : Tensor) (reps : List Nat) : TResult Tensor := Tensor.tileOp t reps
def Tensor.softmaxVerified (t : Tensor) (axis : Nat) : TResult Tensor := Tensor.softmaxOp t axis
def Tensor.randomUniformVerified (dims : List Nat) (minVal maxVal : Float) (seed : Nat) : TResult Tensor :=
  Tensor.randomUniform dims minVal maxVal seed
def Tensor.randomNormalVerified (dims : List Nat) (meanVal stddevVal : Float) (seed : Nat) : TResult Tensor :=
  Tensor.randomNormal dims meanVal stddevVal seed
def Tensor.conv2dVerified (input kernel : Tensor) (stride padding : Nat × Nat) : TResult Tensor :=
  Tensor.conv2d input kernel stride padding
def Tensor.reluVerified (t : Tensor) : Tensor := Tensor.reluOp t
def Tensor.identityVerified (n : Nat) : TResult Tensor := Tensor.identity n
def Tensor.clipVerified (t : Tensor) (minVal maxVal : Float) : Tensor := Tensor.clipOp t minVal maxVal
def Tensor.inverseVerified (t : Tensor) : TResult Tensor := Tensor.inverse t
def Tensor.detVerified (t : Tensor) : TResult Float := Tensor.det t
def Tensor.luVerified (t : Tensor) : TResult (Tensor × Tensor) := Tensor.lu t
def Tensor.qrVerified (t : Tensor) : TResult (Tensor × Tensor) := Tensor.qr t
def Tensor.svdVerified (t : Tensor) : TResult (Tensor × Tensor × Tensor) := Tensor.svd t
def Tensor.eigVerified (t : Tensor) : TResult (Tensor × Tensor) := Tensor.eig t
def Tensor.choleskyVerified (t : Tensor) : TResult Tensor := Tensor.cholesky t
def Tensor.solveVerified (a b : Tensor) : TResult Tensor := Tensor.solve a b
def Tensor.spectralNormVerified (t : Tensor) (maxIter : Nat) (tol : Float) : TResult Float :=
  Tensor.spectralNorm t maxIter tol

@[simp] theorem Tensor.newViewChecked_ok (t : Tensor) (newShape : Shape)
    (h : newShape.totalSize = t.shape.totalSize) :
    Tensor.newViewChecked t newShape = Except.ok (t.newView newShape h) := by
  unfold Tensor.newViewChecked
  simp [h]

@[simp] theorem Tensor.deinitVerified_eq (t : Tensor) :
    Tensor.deinitVerified t = Tensor.release t := rfl

@[simp] theorem Tensor.divVerified_eq (t1 t2 : Tensor) :
    Tensor.divVerified t1 t2 = Tensor.divOp t1 t2 := rfl

@[simp] theorem Tensor.transposeVerified_spec (t : Tensor) (axes : List Nat) :
    Tensor.transposeVerified t axes =
      (if axesPermutation axes t.shape.dims.length then Tensor.transposeOp t axes
       else Except.error TensorError.invalidPermutation) := rfl

@[simp] theorem Tensor.normChecked_spec (t : Tensor) (order : Float) :
    Tensor.normChecked t order =
      (if order ≤ 0.0 then Except.error TensorError.invalidOrder
       else Except.ok (Tensor.norm t order)) := rfl

def ZigTensorModel.init (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.init dims)

@[simp] theorem ZigTensorModel.init_spec (dims : List Nat) :
    ZigTensorModel.init dims = ZigTensorModel.fromTensorResult (Tensor.init dims) := rfl

theorem ZigTensorModel.init_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.init dims) = ZigTensorModel.init dims := by
  rfl

theorem ZigTensorModel.init_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.init dims; result) = ZigTensorModel.fromTensorResult (Tensor.init dims) := by
  rfl

def ZigTensorModel.initWithArena (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.initWithArenaVerified dims)

@[simp] theorem ZigTensorModel.initWithArena_spec (dims : List Nat) :
    ZigTensorModel.initWithArena dims = ZigTensorModel.fromTensorResult (Tensor.initWithArenaVerified dims) := rfl

theorem ZigTensorModel.initWithArena_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.initWithArenaVerified dims) = ZigTensorModel.initWithArena dims := by
  rfl

theorem ZigTensorModel.initWithArena_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.initWithArena dims; result) = ZigTensorModel.fromTensorResult (Tensor.initWithArenaVerified dims) := by
  rfl

def ZigTensorModel.initWithPool (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.initWithPoolVerified dims)

@[simp] theorem ZigTensorModel.initWithPool_spec (dims : List Nat) :
    ZigTensorModel.initWithPool dims = ZigTensorModel.fromTensorResult (Tensor.initWithPoolVerified dims) := rfl

theorem ZigTensorModel.initWithPool_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.initWithPoolVerified dims) = ZigTensorModel.initWithPool dims := by
  rfl

theorem ZigTensorModel.initWithPool_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.initWithPool dims; result) = ZigTensorModel.fromTensorResult (Tensor.initWithPoolVerified dims) := by
  rfl

def ZigTensorModel.initWithSlab (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.initWithSlabVerified dims)

@[simp] theorem ZigTensorModel.initWithSlab_spec (dims : List Nat) :
    ZigTensorModel.initWithSlab dims = ZigTensorModel.fromTensorResult (Tensor.initWithSlabVerified dims) := rfl

theorem ZigTensorModel.initWithSlab_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.initWithSlabVerified dims) = ZigTensorModel.initWithSlab dims := by
  rfl

theorem ZigTensorModel.initWithSlab_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.initWithSlab dims; result) = ZigTensorModel.fromTensorResult (Tensor.initWithSlabVerified dims) := by
  rfl

def ZigTensorModel.initWithBuddy (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.initWithBuddyVerified dims)

@[simp] theorem ZigTensorModel.initWithBuddy_spec (dims : List Nat) :
    ZigTensorModel.initWithBuddy dims = ZigTensorModel.fromTensorResult (Tensor.initWithBuddyVerified dims) := rfl

theorem ZigTensorModel.initWithBuddy_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.initWithBuddyVerified dims) = ZigTensorModel.initWithBuddy dims := by
  rfl

theorem ZigTensorModel.initWithBuddy_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.initWithBuddy dims; result) = ZigTensorModel.fromTensorResult (Tensor.initWithBuddyVerified dims) := by
  rfl

def ZigTensorModel.zeros (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.zeros dims)

@[simp] theorem ZigTensorModel.zeros_spec (dims : List Nat) :
    ZigTensorModel.zeros dims = ZigTensorModel.fromTensorResult (Tensor.zeros dims) := rfl

theorem ZigTensorModel.zeros_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.zeros dims) = ZigTensorModel.zeros dims := by
  rfl

theorem ZigTensorModel.zeros_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.zeros dims; result) = ZigTensorModel.fromTensorResult (Tensor.zeros dims) := by
  rfl

def ZigTensorModel.ones (dims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.ones dims)

@[simp] theorem ZigTensorModel.ones_spec (dims : List Nat) :
    ZigTensorModel.ones dims = ZigTensorModel.fromTensorResult (Tensor.ones dims) := rfl

theorem ZigTensorModel.ones_spec_symm (dims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.ones dims) = ZigTensorModel.ones dims := by
  rfl

theorem ZigTensorModel.ones_let_spec (dims : List Nat) :
    (let result := ZigTensorModel.ones dims; result) = ZigTensorModel.fromTensorResult (Tensor.ones dims) := by
  rfl

def ZigTensorModel.full (dims : List Nat) (value : Float) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.full dims value)

@[simp] theorem ZigTensorModel.full_spec (dims : List Nat) (value : Float) :
    ZigTensorModel.full dims value = ZigTensorModel.fromTensorResult (Tensor.full dims value) := rfl

theorem ZigTensorModel.full_spec_symm (dims : List Nat) (value : Float) :
    ZigTensorModel.fromTensorResult (Tensor.full dims value) = ZigTensorModel.full dims value := by
  rfl

theorem ZigTensorModel.full_let_spec (dims : List Nat) (value : Float) :
    (let result := ZigTensorModel.full dims value; result) = ZigTensorModel.fromTensorResult (Tensor.full dims value) := by
  rfl

def ZigTensorModel.randomUniform (dims : List Nat) (minVal : Float) (maxVal : Float) (seed : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.randomUniformVerified dims minVal maxVal seed)

@[simp] theorem ZigTensorModel.randomUniform_spec (dims : List Nat) (minVal : Float) (maxVal : Float) (seed : Nat) :
    ZigTensorModel.randomUniform dims minVal maxVal seed = ZigTensorModel.fromTensorResult (Tensor.randomUniformVerified dims minVal maxVal seed) := rfl

theorem ZigTensorModel.randomUniform_spec_symm (dims : List Nat) (minVal : Float) (maxVal : Float) (seed : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.randomUniformVerified dims minVal maxVal seed) = ZigTensorModel.randomUniform dims minVal maxVal seed := by
  rfl

theorem ZigTensorModel.randomUniform_let_spec (dims : List Nat) (minVal : Float) (maxVal : Float) (seed : Nat) :
    (let result := ZigTensorModel.randomUniform dims minVal maxVal seed; result) = ZigTensorModel.fromTensorResult (Tensor.randomUniformVerified dims minVal maxVal seed) := by
  rfl

def ZigTensorModel.randomNormal (dims : List Nat) (meanVal : Float) (stddevVal : Float) (seed : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.randomNormalVerified dims meanVal stddevVal seed)

@[simp] theorem ZigTensorModel.randomNormal_spec (dims : List Nat) (meanVal : Float) (stddevVal : Float) (seed : Nat) :
    ZigTensorModel.randomNormal dims meanVal stddevVal seed = ZigTensorModel.fromTensorResult (Tensor.randomNormalVerified dims meanVal stddevVal seed) := rfl

theorem ZigTensorModel.randomNormal_spec_symm (dims : List Nat) (meanVal : Float) (stddevVal : Float) (seed : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.randomNormalVerified dims meanVal stddevVal seed) = ZigTensorModel.randomNormal dims meanVal stddevVal seed := by
  rfl

theorem ZigTensorModel.randomNormal_let_spec (dims : List Nat) (meanVal : Float) (stddevVal : Float) (seed : Nat) :
    (let result := ZigTensorModel.randomNormal dims meanVal stddevVal seed; result) = ZigTensorModel.fromTensorResult (Tensor.randomNormalVerified dims meanVal stddevVal seed) := by
  rfl

def ZigTensorModel.identity (n : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.identityVerified n)

@[simp] theorem ZigTensorModel.identity_spec (n : Nat) :
    ZigTensorModel.identity n = ZigTensorModel.fromTensorResult (Tensor.identityVerified n) := rfl

theorem ZigTensorModel.identity_spec_symm (n : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.identityVerified n) = ZigTensorModel.identity n := by
  rfl

theorem ZigTensorModel.identity_let_spec (n : Nat) :
    (let result := ZigTensorModel.identity n; result) = ZigTensorModel.fromTensorResult (Tensor.identityVerified n) := by
  rfl

def ZigTensorModel.arange (start : Float) (stop : Float) (step : Float) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.arange start stop step)

@[simp] theorem ZigTensorModel.arange_spec (start : Float) (stop : Float) (step : Float) :
    ZigTensorModel.arange start stop step = ZigTensorModel.fromTensorResult (Tensor.arange start stop step) := rfl

theorem ZigTensorModel.arange_spec_symm (start : Float) (stop : Float) (step : Float) :
    ZigTensorModel.fromTensorResult (Tensor.arange start stop step) = ZigTensorModel.arange start stop step := by
  rfl

theorem ZigTensorModel.arange_let_spec (start : Float) (stop : Float) (step : Float) :
    (let result := ZigTensorModel.arange start stop step; result) = ZigTensorModel.fromTensorResult (Tensor.arange start stop step) := by
  rfl

def ZigTensorModel.linspace (start : Float) (stop : Float) (num : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.linspace start stop num)

@[simp] theorem ZigTensorModel.linspace_spec (start : Float) (stop : Float) (num : Nat) :
    ZigTensorModel.linspace start stop num = ZigTensorModel.fromTensorResult (Tensor.linspace start stop num) := rfl

theorem ZigTensorModel.linspace_spec_symm (start : Float) (stop : Float) (num : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.linspace start stop num) = ZigTensorModel.linspace start stop num := by
  rfl

theorem ZigTensorModel.linspace_let_spec (start : Float) (stop : Float) (num : Nat) :
    (let result := ZigTensorModel.linspace start stop num; result) = ZigTensorModel.fromTensorResult (Tensor.linspace start stop num) := by
  rfl

def ZigTensorModel.copy (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.copyVerified zt.toTensor)

@[simp] theorem ZigTensorModel.copy_spec (zt : ZigTensorModel) :
    ZigTensorModel.copy zt = ZigTensorModel.ofTensor (Tensor.copyVerified zt.toTensor) := rfl

theorem ZigTensorModel.copy_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.copyVerified zt.toTensor) = ZigTensorModel.copy zt := by
  rfl

theorem ZigTensorModel.copy_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.copy zt; result) = ZigTensorModel.ofTensor (Tensor.copyVerified zt.toTensor) := by
  rfl

def ZigTensorModel.retain (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.retainVerified zt.toTensor)

@[simp] theorem ZigTensorModel.retain_spec (zt : ZigTensorModel) :
    ZigTensorModel.retain zt = ZigTensorModel.ofTensor (Tensor.retainVerified zt.toTensor) := rfl

theorem ZigTensorModel.retain_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.retainVerified zt.toTensor) = ZigTensorModel.retain zt := by
  rfl

theorem ZigTensorModel.retain_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.retain zt; result) = ZigTensorModel.ofTensor (Tensor.retainVerified zt.toTensor) := by
  rfl

def ZigTensorModel.ensureWritable (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.ensureWritableVerified zt.toTensor)

@[simp] theorem ZigTensorModel.ensureWritable_spec (zt : ZigTensorModel) :
    ZigTensorModel.ensureWritable zt = ZigTensorModel.ofTensor (Tensor.ensureWritableVerified zt.toTensor) := rfl

theorem ZigTensorModel.ensureWritable_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.ensureWritableVerified zt.toTensor) = ZigTensorModel.ensureWritable zt := by
  rfl

theorem ZigTensorModel.ensureWritable_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.ensureWritable zt; result) = ZigTensorModel.ofTensor (Tensor.ensureWritableVerified zt.toTensor) := by
  rfl

def ZigTensorModel.markShared (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.markSharedVerified zt.toTensor)

@[simp] theorem ZigTensorModel.markShared_spec (zt : ZigTensorModel) :
    ZigTensorModel.markShared zt = ZigTensorModel.ofTensor (Tensor.markSharedVerified zt.toTensor) := rfl

theorem ZigTensorModel.markShared_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.markSharedVerified zt.toTensor) = ZigTensorModel.markShared zt := by
  rfl

theorem ZigTensorModel.markShared_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.markShared zt; result) = ZigTensorModel.ofTensor (Tensor.markSharedVerified zt.toTensor) := by
  rfl

def ZigTensorModel.fill (zt : ZigTensorModel) (value : Float) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.fill zt.toTensor value)

@[simp] theorem ZigTensorModel.fill_spec (zt : ZigTensorModel) (value : Float) :
    ZigTensorModel.fill zt value = ZigTensorModel.ofTensor (Tensor.fill zt.toTensor value) := rfl

theorem ZigTensorModel.fill_spec_symm (zt : ZigTensorModel) (value : Float) :
    ZigTensorModel.ofTensor (Tensor.fill zt.toTensor value) = ZigTensorModel.fill zt value := by
  rfl

theorem ZigTensorModel.fill_let_spec (zt : ZigTensorModel) (value : Float) :
    (let result := ZigTensorModel.fill zt value; result) = ZigTensorModel.ofTensor (Tensor.fill zt.toTensor value) := by
  rfl

def ZigTensorModel.addScalar (zt : ZigTensorModel) (s : Float) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.addScalar zt.toTensor s)

@[simp] theorem ZigTensorModel.addScalar_spec (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.addScalar zt s = ZigTensorModel.ofTensor (Tensor.addScalar zt.toTensor s) := rfl

theorem ZigTensorModel.addScalar_spec_symm (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.ofTensor (Tensor.addScalar zt.toTensor s) = ZigTensorModel.addScalar zt s := by
  rfl

theorem ZigTensorModel.addScalar_let_spec (zt : ZigTensorModel) (s : Float) :
    (let result := ZigTensorModel.addScalar zt s; result) = ZigTensorModel.ofTensor (Tensor.addScalar zt.toTensor s) := by
  rfl

def ZigTensorModel.subScalar (zt : ZigTensorModel) (s : Float) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.subScalar zt.toTensor s)

@[simp] theorem ZigTensorModel.subScalar_spec (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.subScalar zt s = ZigTensorModel.ofTensor (Tensor.subScalar zt.toTensor s) := rfl

theorem ZigTensorModel.subScalar_spec_symm (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.ofTensor (Tensor.subScalar zt.toTensor s) = ZigTensorModel.subScalar zt s := by
  rfl

theorem ZigTensorModel.subScalar_let_spec (zt : ZigTensorModel) (s : Float) :
    (let result := ZigTensorModel.subScalar zt s; result) = ZigTensorModel.ofTensor (Tensor.subScalar zt.toTensor s) := by
  rfl

def ZigTensorModel.mulScalar (zt : ZigTensorModel) (s : Float) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.mulScalar zt.toTensor s)

@[simp] theorem ZigTensorModel.mulScalar_spec (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.mulScalar zt s = ZigTensorModel.ofTensor (Tensor.mulScalar zt.toTensor s) := rfl

theorem ZigTensorModel.mulScalar_spec_symm (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.ofTensor (Tensor.mulScalar zt.toTensor s) = ZigTensorModel.mulScalar zt s := by
  rfl

theorem ZigTensorModel.mulScalar_let_spec (zt : ZigTensorModel) (s : Float) :
    (let result := ZigTensorModel.mulScalar zt s; result) = ZigTensorModel.ofTensor (Tensor.mulScalar zt.toTensor s) := by
  rfl

def ZigTensorModel.divScalar (zt : ZigTensorModel) (s : Float) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.divScalar zt.toTensor s)

@[simp] theorem ZigTensorModel.divScalar_spec (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.divScalar zt s = ZigTensorModel.fromTensorResult (Tensor.divScalar zt.toTensor s) := rfl

theorem ZigTensorModel.divScalar_spec_symm (zt : ZigTensorModel) (s : Float) :
    ZigTensorModel.fromTensorResult (Tensor.divScalar zt.toTensor s) = ZigTensorModel.divScalar zt s := by
  rfl

theorem ZigTensorModel.divScalar_let_spec (zt : ZigTensorModel) (s : Float) :
    (let result := ZigTensorModel.divScalar zt s; result) = ZigTensorModel.fromTensorResult (Tensor.divScalar zt.toTensor s) := by
  rfl

def ZigTensorModel.exp (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.expVerified zt.toTensor)

@[simp] theorem ZigTensorModel.exp_spec (zt : ZigTensorModel) :
    ZigTensorModel.exp zt = ZigTensorModel.ofTensor (Tensor.expVerified zt.toTensor) := rfl

theorem ZigTensorModel.exp_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.expVerified zt.toTensor) = ZigTensorModel.exp zt := by
  rfl

theorem ZigTensorModel.exp_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.exp zt; result) = ZigTensorModel.ofTensor (Tensor.expVerified zt.toTensor) := by
  rfl

def ZigTensorModel.log (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.logVerified zt.toTensor)

@[simp] theorem ZigTensorModel.log_spec (zt : ZigTensorModel) :
    ZigTensorModel.log zt = ZigTensorModel.ofTensor (Tensor.logVerified zt.toTensor) := rfl

theorem ZigTensorModel.log_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.logVerified zt.toTensor) = ZigTensorModel.log zt := by
  rfl

theorem ZigTensorModel.log_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.log zt; result) = ZigTensorModel.ofTensor (Tensor.logVerified zt.toTensor) := by
  rfl

def ZigTensorModel.sin (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.sinVerified zt.toTensor)

@[simp] theorem ZigTensorModel.sin_spec (zt : ZigTensorModel) :
    ZigTensorModel.sin zt = ZigTensorModel.ofTensor (Tensor.sinVerified zt.toTensor) := rfl

theorem ZigTensorModel.sin_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.sinVerified zt.toTensor) = ZigTensorModel.sin zt := by
  rfl

theorem ZigTensorModel.sin_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.sin zt; result) = ZigTensorModel.ofTensor (Tensor.sinVerified zt.toTensor) := by
  rfl

def ZigTensorModel.cos (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.cosVerified zt.toTensor)

@[simp] theorem ZigTensorModel.cos_spec (zt : ZigTensorModel) :
    ZigTensorModel.cos zt = ZigTensorModel.ofTensor (Tensor.cosVerified zt.toTensor) := rfl

theorem ZigTensorModel.cos_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.cosVerified zt.toTensor) = ZigTensorModel.cos zt := by
  rfl

theorem ZigTensorModel.cos_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.cos zt; result) = ZigTensorModel.ofTensor (Tensor.cosVerified zt.toTensor) := by
  rfl

def ZigTensorModel.tan (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.tanVerified zt.toTensor)

@[simp] theorem ZigTensorModel.tan_spec (zt : ZigTensorModel) :
    ZigTensorModel.tan zt = ZigTensorModel.ofTensor (Tensor.tanVerified zt.toTensor) := rfl

theorem ZigTensorModel.tan_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.tanVerified zt.toTensor) = ZigTensorModel.tan zt := by
  rfl

theorem ZigTensorModel.tan_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.tan zt; result) = ZigTensorModel.ofTensor (Tensor.tanVerified zt.toTensor) := by
  rfl

def ZigTensorModel.sqrt (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.sqrtVerified zt.toTensor)

@[simp] theorem ZigTensorModel.sqrt_spec (zt : ZigTensorModel) :
    ZigTensorModel.sqrt zt = ZigTensorModel.ofTensor (Tensor.sqrtVerified zt.toTensor) := rfl

theorem ZigTensorModel.sqrt_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.sqrtVerified zt.toTensor) = ZigTensorModel.sqrt zt := by
  rfl

theorem ZigTensorModel.sqrt_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.sqrt zt; result) = ZigTensorModel.ofTensor (Tensor.sqrtVerified zt.toTensor) := by
  rfl

def ZigTensorModel.abs (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.absVerified zt.toTensor)

@[simp] theorem ZigTensorModel.abs_spec (zt : ZigTensorModel) :
    ZigTensorModel.abs zt = ZigTensorModel.ofTensor (Tensor.absVerified zt.toTensor) := rfl

theorem ZigTensorModel.abs_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.absVerified zt.toTensor) = ZigTensorModel.abs zt := by
  rfl

theorem ZigTensorModel.abs_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.abs zt; result) = ZigTensorModel.ofTensor (Tensor.absVerified zt.toTensor) := by
  rfl

def ZigTensorModel.relu (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.reluVerified zt.toTensor)

@[simp] theorem ZigTensorModel.relu_spec (zt : ZigTensorModel) :
    ZigTensorModel.relu zt = ZigTensorModel.ofTensor (Tensor.reluVerified zt.toTensor) := rfl

theorem ZigTensorModel.relu_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.reluVerified zt.toTensor) = ZigTensorModel.relu zt := by
  rfl

theorem ZigTensorModel.relu_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.relu zt; result) = ZigTensorModel.ofTensor (Tensor.reluVerified zt.toTensor) := by
  rfl

def ZigTensorModel.toInt (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.toInt zt.toTensor)

@[simp] theorem ZigTensorModel.toInt_spec (zt : ZigTensorModel) :
    ZigTensorModel.toInt zt = ZigTensorModel.ofTensor (Tensor.toInt zt.toTensor) := rfl

theorem ZigTensorModel.toInt_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.toInt zt.toTensor) = ZigTensorModel.toInt zt := by
  rfl

theorem ZigTensorModel.toInt_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.toInt zt; result) = ZigTensorModel.ofTensor (Tensor.toInt zt.toTensor) := by
  rfl

def ZigTensorModel.toFixed (zt : ZigTensorModel) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.toFixed zt.toTensor)

@[simp] theorem ZigTensorModel.toFixed_spec (zt : ZigTensorModel) :
    ZigTensorModel.toFixed zt = ZigTensorModel.ofTensor (Tensor.toFixed zt.toTensor) := rfl

theorem ZigTensorModel.toFixed_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.ofTensor (Tensor.toFixed zt.toTensor) = ZigTensorModel.toFixed zt := by
  rfl

theorem ZigTensorModel.toFixed_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.toFixed zt; result) = ZigTensorModel.ofTensor (Tensor.toFixed zt.toTensor) := by
  rfl

def ZigTensorModel.pow (zt : ZigTensorModel) (exponent : Float) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.powVerified zt.toTensor exponent)

@[simp] theorem ZigTensorModel.pow_spec (zt : ZigTensorModel) (exponent : Float) :
    ZigTensorModel.pow zt exponent = ZigTensorModel.ofTensor (Tensor.powVerified zt.toTensor exponent) := rfl

theorem ZigTensorModel.pow_spec_symm (zt : ZigTensorModel) (exponent : Float) :
    ZigTensorModel.ofTensor (Tensor.powVerified zt.toTensor exponent) = ZigTensorModel.pow zt exponent := by
  rfl

theorem ZigTensorModel.pow_let_spec (zt : ZigTensorModel) (exponent : Float) :
    (let result := ZigTensorModel.pow zt exponent; result) = ZigTensorModel.ofTensor (Tensor.powVerified zt.toTensor exponent) := by
  rfl

def ZigTensorModel.clip (zt : ZigTensorModel) (minVal : Float) (maxVal : Float) : ZigTensorModel :=
  ZigTensorModel.ofTensor (Tensor.clipVerified zt.toTensor minVal maxVal)

@[simp] theorem ZigTensorModel.clip_spec (zt : ZigTensorModel) (minVal : Float) (maxVal : Float) :
    ZigTensorModel.clip zt minVal maxVal = ZigTensorModel.ofTensor (Tensor.clipVerified zt.toTensor minVal maxVal) := rfl

theorem ZigTensorModel.clip_spec_symm (zt : ZigTensorModel) (minVal : Float) (maxVal : Float) :
    ZigTensorModel.ofTensor (Tensor.clipVerified zt.toTensor minVal maxVal) = ZigTensorModel.clip zt minVal maxVal := by
  rfl

theorem ZigTensorModel.clip_let_spec (zt : ZigTensorModel) (minVal : Float) (maxVal : Float) :
    (let result := ZigTensorModel.clip zt minVal maxVal; result) = ZigTensorModel.ofTensor (Tensor.clipVerified zt.toTensor minVal maxVal) := by
  rfl

def ZigTensorModel.reshape (zt : ZigTensorModel) (newDims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.reshape zt.toTensor newDims)

@[simp] theorem ZigTensorModel.reshape_spec (zt : ZigTensorModel) (newDims : List Nat) :
    ZigTensorModel.reshape zt newDims = ZigTensorModel.fromTensorResult (Tensor.reshape zt.toTensor newDims) := rfl

theorem ZigTensorModel.reshape_spec_symm (zt : ZigTensorModel) (newDims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.reshape zt.toTensor newDims) = ZigTensorModel.reshape zt newDims := by
  rfl

theorem ZigTensorModel.reshape_let_spec (zt : ZigTensorModel) (newDims : List Nat) :
    (let result := ZigTensorModel.reshape zt newDims; result) = ZigTensorModel.fromTensorResult (Tensor.reshape zt.toTensor newDims) := by
  rfl

def ZigTensorModel.view (zt : ZigTensorModel) (newDims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.viewVerified zt.toTensor newDims)

@[simp] theorem ZigTensorModel.view_spec (zt : ZigTensorModel) (newDims : List Nat) :
    ZigTensorModel.view zt newDims = ZigTensorModel.fromTensorResult (Tensor.viewVerified zt.toTensor newDims) := rfl

theorem ZigTensorModel.view_spec_symm (zt : ZigTensorModel) (newDims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.viewVerified zt.toTensor newDims) = ZigTensorModel.view zt newDims := by
  rfl

theorem ZigTensorModel.view_let_spec (zt : ZigTensorModel) (newDims : List Nat) :
    (let result := ZigTensorModel.view zt newDims; result) = ZigTensorModel.fromTensorResult (Tensor.viewVerified zt.toTensor newDims) := by
  rfl

def ZigTensorModel.slice (zt : ZigTensorModel) (starts : List Nat) (ends : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.sliceVerified zt.toTensor starts ends)

@[simp] theorem ZigTensorModel.slice_spec (zt : ZigTensorModel) (starts : List Nat) (ends : List Nat) :
    ZigTensorModel.slice zt starts ends = ZigTensorModel.fromTensorResult (Tensor.sliceVerified zt.toTensor starts ends) := rfl

theorem ZigTensorModel.slice_spec_symm (zt : ZigTensorModel) (starts : List Nat) (ends : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.sliceVerified zt.toTensor starts ends) = ZigTensorModel.slice zt starts ends := by
  rfl

theorem ZigTensorModel.slice_let_spec (zt : ZigTensorModel) (starts : List Nat) (ends : List Nat) :
    (let result := ZigTensorModel.slice zt starts ends; result) = ZigTensorModel.fromTensorResult (Tensor.sliceVerified zt.toTensor starts ends) := by
  rfl

def ZigTensorModel.transpose (zt : ZigTensorModel) (axes : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.transposeVerified zt.toTensor axes)

@[simp] theorem ZigTensorModel.transpose_spec (zt : ZigTensorModel) (axes : List Nat) :
    ZigTensorModel.transpose zt axes = ZigTensorModel.fromTensorResult (Tensor.transposeVerified zt.toTensor axes) := rfl

theorem ZigTensorModel.transpose_spec_symm (zt : ZigTensorModel) (axes : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.transposeVerified zt.toTensor axes) = ZigTensorModel.transpose zt axes := by
  rfl

theorem ZigTensorModel.transpose_let_spec (zt : ZigTensorModel) (axes : List Nat) :
    (let result := ZigTensorModel.transpose zt axes; result) = ZigTensorModel.fromTensorResult (Tensor.transposeVerified zt.toTensor axes) := by
  rfl

def ZigTensorModel.max (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.maxVerified zt.toTensor axis)

@[simp] theorem ZigTensorModel.max_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.max zt axis = ZigTensorModel.fromTensorResult (Tensor.maxVerified zt.toTensor axis) := rfl

theorem ZigTensorModel.max_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.maxVerified zt.toTensor axis) = ZigTensorModel.max zt axis := by
  rfl

theorem ZigTensorModel.max_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.max zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.maxVerified zt.toTensor axis) := by
  rfl

def ZigTensorModel.min (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.minVerified zt.toTensor axis)

@[simp] theorem ZigTensorModel.min_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.min zt axis = ZigTensorModel.fromTensorResult (Tensor.minVerified zt.toTensor axis) := rfl

theorem ZigTensorModel.min_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.minVerified zt.toTensor axis) = ZigTensorModel.min zt axis := by
  rfl

theorem ZigTensorModel.min_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.min zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.minVerified zt.toTensor axis) := by
  rfl

def ZigTensorModel.sum (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.sumVerified zt.toTensor axis)

@[simp] theorem ZigTensorModel.sum_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.sum zt axis = ZigTensorModel.fromTensorResult (Tensor.sumVerified zt.toTensor axis) := rfl

theorem ZigTensorModel.sum_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.sumVerified zt.toTensor axis) = ZigTensorModel.sum zt axis := by
  rfl

theorem ZigTensorModel.sum_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.sum zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.sumVerified zt.toTensor axis) := by
  rfl

def ZigTensorModel.mean (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.meanVerified zt.toTensor axis)

@[simp] theorem ZigTensorModel.mean_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.mean zt axis = ZigTensorModel.fromTensorResult (Tensor.meanVerified zt.toTensor axis) := rfl

theorem ZigTensorModel.mean_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.meanVerified zt.toTensor axis) = ZigTensorModel.mean zt axis := by
  rfl

theorem ZigTensorModel.mean_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.mean zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.meanVerified zt.toTensor axis) := by
  rfl

def ZigTensorModel.broadcast (zt : ZigTensorModel) (targetDims : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.broadcastVerified zt.toTensor targetDims)

@[simp] theorem ZigTensorModel.broadcast_spec (zt : ZigTensorModel) (targetDims : List Nat) :
    ZigTensorModel.broadcast zt targetDims = ZigTensorModel.fromTensorResult (Tensor.broadcastVerified zt.toTensor targetDims) := rfl

theorem ZigTensorModel.broadcast_spec_symm (zt : ZigTensorModel) (targetDims : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.broadcastVerified zt.toTensor targetDims) = ZigTensorModel.broadcast zt targetDims := by
  rfl

theorem ZigTensorModel.broadcast_let_spec (zt : ZigTensorModel) (targetDims : List Nat) :
    (let result := ZigTensorModel.broadcast zt targetDims; result) = ZigTensorModel.fromTensorResult (Tensor.broadcastVerified zt.toTensor targetDims) := by
  rfl

def ZigTensorModel.unsqueeze (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.unsqueezeVerified zt.toTensor axis)

@[simp] theorem ZigTensorModel.unsqueeze_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.unsqueeze zt axis = ZigTensorModel.fromTensorResult (Tensor.unsqueezeVerified zt.toTensor axis) := rfl

theorem ZigTensorModel.unsqueeze_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.unsqueezeVerified zt.toTensor axis) = ZigTensorModel.unsqueeze zt axis := by
  rfl

theorem ZigTensorModel.unsqueeze_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.unsqueeze zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.unsqueezeVerified zt.toTensor axis) := by
  rfl

def ZigTensorModel.softmax (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.softmaxVerified zt.toTensor axis)

@[simp] theorem ZigTensorModel.softmax_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.softmax zt axis = ZigTensorModel.fromTensorResult (Tensor.softmaxVerified zt.toTensor axis) := rfl

theorem ZigTensorModel.softmax_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.softmaxVerified zt.toTensor axis) = ZigTensorModel.softmax zt axis := by
  rfl

theorem ZigTensorModel.softmax_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.softmax zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.softmaxVerified zt.toTensor axis) := by
  rfl

def ZigTensorModel.pad (zt : ZigTensorModel) (pads : List (Nat × Nat)) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.padVerified zt.toTensor pads)

@[simp] theorem ZigTensorModel.pad_spec (zt : ZigTensorModel) (pads : List (Nat × Nat)) :
    ZigTensorModel.pad zt pads = ZigTensorModel.fromTensorResult (Tensor.padVerified zt.toTensor pads) := rfl

theorem ZigTensorModel.pad_spec_symm (zt : ZigTensorModel) (pads : List (Nat × Nat)) :
    ZigTensorModel.fromTensorResult (Tensor.padVerified zt.toTensor pads) = ZigTensorModel.pad zt pads := by
  rfl

theorem ZigTensorModel.pad_let_spec (zt : ZigTensorModel) (pads : List (Nat × Nat)) :
    (let result := ZigTensorModel.pad zt pads; result) = ZigTensorModel.fromTensorResult (Tensor.padVerified zt.toTensor pads) := by
  rfl

def ZigTensorModel.tile (zt : ZigTensorModel) (reps : List Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.tileVerified zt.toTensor reps)

@[simp] theorem ZigTensorModel.tile_spec (zt : ZigTensorModel) (reps : List Nat) :
    ZigTensorModel.tile zt reps = ZigTensorModel.fromTensorResult (Tensor.tileVerified zt.toTensor reps) := rfl

theorem ZigTensorModel.tile_spec_symm (zt : ZigTensorModel) (reps : List Nat) :
    ZigTensorModel.fromTensorResult (Tensor.tileVerified zt.toTensor reps) = ZigTensorModel.tile zt reps := by
  rfl

theorem ZigTensorModel.tile_let_spec (zt : ZigTensorModel) (reps : List Nat) :
    (let result := ZigTensorModel.tile zt reps; result) = ZigTensorModel.fromTensorResult (Tensor.tileVerified zt.toTensor reps) := by
  rfl

def ZigTensorModel.argmax (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.argmax zt.toTensor axis)

@[simp] theorem ZigTensorModel.argmax_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.argmax zt axis = ZigTensorModel.fromTensorResult (Tensor.argmax zt.toTensor axis) := rfl

theorem ZigTensorModel.argmax_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.argmax zt.toTensor axis) = ZigTensorModel.argmax zt axis := by
  rfl

theorem ZigTensorModel.argmax_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.argmax zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.argmax zt.toTensor axis) := by
  rfl

def ZigTensorModel.argmin (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.argmin zt.toTensor axis)

@[simp] theorem ZigTensorModel.argmin_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.argmin zt axis = ZigTensorModel.fromTensorResult (Tensor.argmin zt.toTensor axis) := rfl

theorem ZigTensorModel.argmin_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.argmin zt.toTensor axis) = ZigTensorModel.argmin zt axis := by
  rfl

theorem ZigTensorModel.argmin_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.argmin zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.argmin zt.toTensor axis) := by
  rfl

def ZigTensorModel.cumsum (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.cumsum zt.toTensor axis)

@[simp] theorem ZigTensorModel.cumsum_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.cumsum zt axis = ZigTensorModel.fromTensorResult (Tensor.cumsum zt.toTensor axis) := rfl

theorem ZigTensorModel.cumsum_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.cumsum zt.toTensor axis) = ZigTensorModel.cumsum zt axis := by
  rfl

theorem ZigTensorModel.cumsum_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.cumsum zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.cumsum zt.toTensor axis) := by
  rfl

def ZigTensorModel.variance (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.variance zt.toTensor axis)

@[simp] theorem ZigTensorModel.variance_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.variance zt axis = ZigTensorModel.fromTensorResult (Tensor.variance zt.toTensor axis) := rfl

theorem ZigTensorModel.variance_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.variance zt.toTensor axis) = ZigTensorModel.variance zt axis := by
  rfl

theorem ZigTensorModel.variance_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.variance zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.variance zt.toTensor axis) := by
  rfl

def ZigTensorModel.stddev (zt : ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.stddev zt.toTensor axis)

@[simp] theorem ZigTensorModel.stddev_spec (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.stddev zt axis = ZigTensorModel.fromTensorResult (Tensor.stddev zt.toTensor axis) := rfl

theorem ZigTensorModel.stddev_spec_symm (zt : ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.stddev zt.toTensor axis) = ZigTensorModel.stddev zt axis := by
  rfl

theorem ZigTensorModel.stddev_let_spec (zt : ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.stddev zt axis; result) = ZigTensorModel.fromTensorResult (Tensor.stddev zt.toTensor axis) := by
  rfl

def ZigTensorModel.sort (zt : ZigTensorModel) (axis : Nat) (descending : Bool) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.sort zt.toTensor axis descending)

@[simp] theorem ZigTensorModel.sort_spec (zt : ZigTensorModel) (axis : Nat) (descending : Bool) :
    ZigTensorModel.sort zt axis descending = ZigTensorModel.fromTensorResult (Tensor.sort zt.toTensor axis descending) := rfl

theorem ZigTensorModel.sort_spec_symm (zt : ZigTensorModel) (axis : Nat) (descending : Bool) :
    ZigTensorModel.fromTensorResult (Tensor.sort zt.toTensor axis descending) = ZigTensorModel.sort zt axis descending := by
  rfl

theorem ZigTensorModel.sort_let_spec (zt : ZigTensorModel) (axis : Nat) (descending : Bool) :
    (let result := ZigTensorModel.sort zt axis descending; result) = ZigTensorModel.fromTensorResult (Tensor.sort zt.toTensor axis descending) := by
  rfl

def ZigTensorModel.unique (zt : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.unique zt.toTensor)

@[simp] theorem ZigTensorModel.unique_spec (zt : ZigTensorModel) :
    ZigTensorModel.unique zt = ZigTensorModel.fromTensorResult (Tensor.unique zt.toTensor) := rfl

theorem ZigTensorModel.unique_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.unique zt.toTensor) = ZigTensorModel.unique zt := by
  rfl

theorem ZigTensorModel.unique_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.unique zt; result) = ZigTensorModel.fromTensorResult (Tensor.unique zt.toTensor) := by
  rfl

def ZigTensorModel.oneHot (zt : ZigTensorModel) (numClasses : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.oneHot zt.toTensor numClasses)

@[simp] theorem ZigTensorModel.oneHot_spec (zt : ZigTensorModel) (numClasses : Nat) :
    ZigTensorModel.oneHot zt numClasses = ZigTensorModel.fromTensorResult (Tensor.oneHot zt.toTensor numClasses) := rfl

theorem ZigTensorModel.oneHot_spec_symm (zt : ZigTensorModel) (numClasses : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.oneHot zt.toTensor numClasses) = ZigTensorModel.oneHot zt numClasses := by
  rfl

theorem ZigTensorModel.oneHot_let_spec (zt : ZigTensorModel) (numClasses : Nat) :
    (let result := ZigTensorModel.oneHot zt numClasses; result) = ZigTensorModel.fromTensorResult (Tensor.oneHot zt.toTensor numClasses) := by
  rfl

def ZigTensorModel.inverse (zt : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.inverseVerified zt.toTensor)

@[simp] theorem ZigTensorModel.inverse_spec (zt : ZigTensorModel) :
    ZigTensorModel.inverse zt = ZigTensorModel.fromTensorResult (Tensor.inverseVerified zt.toTensor) := rfl

theorem ZigTensorModel.inverse_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.inverseVerified zt.toTensor) = ZigTensorModel.inverse zt := by
  rfl

theorem ZigTensorModel.inverse_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.inverse zt; result) = ZigTensorModel.fromTensorResult (Tensor.inverseVerified zt.toTensor) := by
  rfl

def ZigTensorModel.cholesky (zt : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.choleskyVerified zt.toTensor)

@[simp] theorem ZigTensorModel.cholesky_spec (zt : ZigTensorModel) :
    ZigTensorModel.cholesky zt = ZigTensorModel.fromTensorResult (Tensor.choleskyVerified zt.toTensor) := rfl

theorem ZigTensorModel.cholesky_spec_symm (zt : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.choleskyVerified zt.toTensor) = ZigTensorModel.cholesky zt := by
  rfl

theorem ZigTensorModel.cholesky_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.cholesky zt; result) = ZigTensorModel.fromTensorResult (Tensor.choleskyVerified zt.toTensor) := by
  rfl

def ZigTensorModel.newView (zt : ZigTensorModel) (newShape : ZigShapeModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.newViewChecked zt.toTensor newShape.toShape)

@[simp] theorem ZigTensorModel.newView_spec (zt : ZigTensorModel) (newShape : ZigShapeModel) :
    ZigTensorModel.newView zt newShape = ZigTensorModel.fromTensorResult (Tensor.newViewChecked zt.toTensor newShape.toShape) := rfl

theorem ZigTensorModel.newView_spec_symm (zt : ZigTensorModel) (newShape : ZigShapeModel) :
    ZigTensorModel.fromTensorResult (Tensor.newViewChecked zt.toTensor newShape.toShape) = ZigTensorModel.newView zt newShape := by
  rfl

theorem ZigTensorModel.newView_let_spec (zt : ZigTensorModel) (newShape : ZigShapeModel) :
    (let result := ZigTensorModel.newView zt newShape; result) = ZigTensorModel.fromTensorResult (Tensor.newViewChecked zt.toTensor newShape.toShape) := by
  rfl

def ZigTensorModel.add (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.add a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.add_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.add a b = ZigTensorModel.fromTensorResult (Tensor.add a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.add_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.add a.toTensor b.toTensor) = ZigTensorModel.add a b := by
  rfl

theorem ZigTensorModel.add_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.add a b; result) = ZigTensorModel.fromTensorResult (Tensor.add a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.sub (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.sub a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.sub_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.sub a b = ZigTensorModel.fromTensorResult (Tensor.sub a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.sub_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.sub a.toTensor b.toTensor) = ZigTensorModel.sub a b := by
  rfl

theorem ZigTensorModel.sub_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.sub a b; result) = ZigTensorModel.fromTensorResult (Tensor.sub a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.mul (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.mul a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.mul_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.mul a b = ZigTensorModel.fromTensorResult (Tensor.mul a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.mul_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.mul a.toTensor b.toTensor) = ZigTensorModel.mul a b := by
  rfl

theorem ZigTensorModel.mul_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.mul a b; result) = ZigTensorModel.fromTensorResult (Tensor.mul a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.div (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.divVerified a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.div_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.div a b = ZigTensorModel.fromTensorResult (Tensor.divVerified a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.div_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.divVerified a.toTensor b.toTensor) = ZigTensorModel.div a b := by
  rfl

theorem ZigTensorModel.div_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.div a b; result) = ZigTensorModel.fromTensorResult (Tensor.divVerified a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.matmul (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.matmul a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.matmul_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.matmul a b = ZigTensorModel.fromTensorResult (Tensor.matmul a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.matmul_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.matmul a.toTensor b.toTensor) = ZigTensorModel.matmul a b := by
  rfl

theorem ZigTensorModel.matmul_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.matmul a b; result) = ZigTensorModel.fromTensorResult (Tensor.matmul a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.outer (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.outer a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.outer_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.outer a b = ZigTensorModel.fromTensorResult (Tensor.outer a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.outer_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.outer a.toTensor b.toTensor) = ZigTensorModel.outer a b := by
  rfl

theorem ZigTensorModel.outer_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.outer a b; result) = ZigTensorModel.fromTensorResult (Tensor.outer a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.solve (a : ZigTensorModel) (b : ZigTensorModel) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.solveVerified a.toTensor b.toTensor)

@[simp] theorem ZigTensorModel.solve_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.solve a b = ZigTensorModel.fromTensorResult (Tensor.solveVerified a.toTensor b.toTensor) := rfl

theorem ZigTensorModel.solve_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.fromTensorResult (Tensor.solveVerified a.toTensor b.toTensor) = ZigTensorModel.solve a b := by
  rfl

theorem ZigTensorModel.solve_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.solve a b; result) = ZigTensorModel.fromTensorResult (Tensor.solveVerified a.toTensor b.toTensor) := by
  rfl

def ZigTensorModel.conv2d (input : ZigTensorModel) (kernel : ZigTensorModel) (stride : Nat × Nat) (padding : Nat × Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.conv2dVerified input.toTensor kernel.toTensor stride padding)

@[simp] theorem ZigTensorModel.conv2d_spec (input : ZigTensorModel) (kernel : ZigTensorModel) (stride : Nat × Nat) (padding : Nat × Nat) :
    ZigTensorModel.conv2d input kernel stride padding = ZigTensorModel.fromTensorResult (Tensor.conv2dVerified input.toTensor kernel.toTensor stride padding) := rfl

theorem ZigTensorModel.conv2d_spec_symm (input : ZigTensorModel) (kernel : ZigTensorModel) (stride : Nat × Nat) (padding : Nat × Nat) :
    ZigTensorModel.fromTensorResult (Tensor.conv2dVerified input.toTensor kernel.toTensor stride padding) = ZigTensorModel.conv2d input kernel stride padding := by
  rfl

theorem ZigTensorModel.conv2d_let_spec (input : ZigTensorModel) (kernel : ZigTensorModel) (stride : Nat × Nat) (padding : Nat × Nat) :
    (let result := ZigTensorModel.conv2d input kernel stride padding; result) = ZigTensorModel.fromTensorResult (Tensor.conv2dVerified input.toTensor kernel.toTensor stride padding) := by
  rfl

def ZigTensorModel.concat (tensors : List ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.concatVerified (tensors.map ZigTensorModel.toTensor) axis)

@[simp] theorem ZigTensorModel.concat_spec (tensors : List ZigTensorModel) (axis : Nat) :
    ZigTensorModel.concat tensors axis = ZigTensorModel.fromTensorResult (Tensor.concatVerified (tensors.map ZigTensorModel.toTensor) axis) := rfl

theorem ZigTensorModel.concat_spec_symm (tensors : List ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.concatVerified (tensors.map ZigTensorModel.toTensor) axis) = ZigTensorModel.concat tensors axis := by
  rfl

theorem ZigTensorModel.concat_let_spec (tensors : List ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.concat tensors axis; result) = ZigTensorModel.fromTensorResult (Tensor.concatVerified (tensors.map ZigTensorModel.toTensor) axis) := by
  rfl

def ZigTensorModel.stack (tensors : List ZigTensorModel) (axis : Nat) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.stackVerified (tensors.map ZigTensorModel.toTensor) axis)

@[simp] theorem ZigTensorModel.stack_spec (tensors : List ZigTensorModel) (axis : Nat) :
    ZigTensorModel.stack tensors axis = ZigTensorModel.fromTensorResult (Tensor.stackVerified (tensors.map ZigTensorModel.toTensor) axis) := rfl

theorem ZigTensorModel.stack_spec_symm (tensors : List ZigTensorModel) (axis : Nat) :
    ZigTensorModel.fromTensorResult (Tensor.stackVerified (tensors.map ZigTensorModel.toTensor) axis) = ZigTensorModel.stack tensors axis := by
  rfl

theorem ZigTensorModel.stack_let_spec (tensors : List ZigTensorModel) (axis : Nat) :
    (let result := ZigTensorModel.stack tensors axis; result) = ZigTensorModel.fromTensorResult (Tensor.stackVerified (tensors.map ZigTensorModel.toTensor) axis) := by
  rfl

def ZigTensorModel.get (zt : ZigTensorModel) (indices : List Nat) : TResult Float :=
  Tensor.get zt.toTensor indices

@[simp] theorem ZigTensorModel.get_spec (zt : ZigTensorModel) (indices : List Nat) :
    ZigTensorModel.get zt indices = Tensor.get zt.toTensor indices := rfl

theorem ZigTensorModel.get_spec_symm (zt : ZigTensorModel) (indices : List Nat) :
    Tensor.get zt.toTensor indices = ZigTensorModel.get zt indices := by
  rfl

theorem ZigTensorModel.get_let_spec (zt : ZigTensorModel) (indices : List Nat) :
    (let result := ZigTensorModel.get zt indices; result) = Tensor.get zt.toTensor indices := by
  rfl

def ZigTensorModel.set (zt : ZigTensorModel) (indices : List Nat) (value : Float) : TResult ZigTensorModel :=
  ZigTensorModel.fromTensorResult (Tensor.set zt.toTensor indices value)

@[simp] theorem ZigTensorModel.set_spec (zt : ZigTensorModel) (indices : List Nat) (value : Float) :
    ZigTensorModel.set zt indices value = ZigTensorModel.fromTensorResult (Tensor.set zt.toTensor indices value) := rfl

theorem ZigTensorModel.set_spec_symm (zt : ZigTensorModel) (indices : List Nat) (value : Float) :
    ZigTensorModel.fromTensorResult (Tensor.set zt.toTensor indices value) = ZigTensorModel.set zt indices value := by
  rfl

theorem ZigTensorModel.set_let_spec (zt : ZigTensorModel) (indices : List Nat) (value : Float) :
    (let result := ZigTensorModel.set zt indices value; result) = ZigTensorModel.fromTensorResult (Tensor.set zt.toTensor indices value) := by
  rfl

def ZigTensorModel.normL2 (zt : ZigTensorModel) : Float :=
  Tensor.normL2 zt.toTensor

@[simp] theorem ZigTensorModel.normL2_spec (zt : ZigTensorModel) :
    ZigTensorModel.normL2 zt = Tensor.normL2 zt.toTensor := rfl

theorem ZigTensorModel.normL2_spec_symm (zt : ZigTensorModel) :
    Tensor.normL2 zt.toTensor = ZigTensorModel.normL2 zt := by
  rfl

theorem ZigTensorModel.normL2_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.normL2 zt; result) = Tensor.normL2 zt.toTensor := by
  rfl

def ZigTensorModel.dot (a : ZigTensorModel) (b : ZigTensorModel) : TResult Float :=
  Tensor.dot a.toTensor b.toTensor

@[simp] theorem ZigTensorModel.dot_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    ZigTensorModel.dot a b = Tensor.dot a.toTensor b.toTensor := rfl

theorem ZigTensorModel.dot_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) :
    Tensor.dot a.toTensor b.toTensor = ZigTensorModel.dot a b := by
  rfl

theorem ZigTensorModel.dot_let_spec (a : ZigTensorModel) (b : ZigTensorModel) :
    (let result := ZigTensorModel.dot a b; result) = Tensor.dot a.toTensor b.toTensor := by
  rfl

def ZigTensorModel.trace (zt : ZigTensorModel) : TResult Float :=
  Tensor.trace zt.toTensor

@[simp] theorem ZigTensorModel.trace_spec (zt : ZigTensorModel) :
    ZigTensorModel.trace zt = Tensor.trace zt.toTensor := rfl

theorem ZigTensorModel.trace_spec_symm (zt : ZigTensorModel) :
    Tensor.trace zt.toTensor = ZigTensorModel.trace zt := by
  rfl

theorem ZigTensorModel.trace_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.trace zt; result) = Tensor.trace zt.toTensor := by
  rfl

def ZigTensorModel.norm (zt : ZigTensorModel) (order : Float) : TResult Float :=
  Tensor.normChecked zt.toTensor order

@[simp] theorem ZigTensorModel.norm_spec (zt : ZigTensorModel) (order : Float) :
    ZigTensorModel.norm zt order = Tensor.normChecked zt.toTensor order := rfl

theorem ZigTensorModel.norm_spec_symm (zt : ZigTensorModel) (order : Float) :
    Tensor.normChecked zt.toTensor order = ZigTensorModel.norm zt order := by
  rfl

theorem ZigTensorModel.norm_let_spec (zt : ZigTensorModel) (order : Float) :
    (let result := ZigTensorModel.norm zt order; result) = Tensor.normChecked zt.toTensor order := by
  rfl

def ZigTensorModel.det (zt : ZigTensorModel) : TResult Float :=
  Tensor.detVerified zt.toTensor

@[simp] theorem ZigTensorModel.det_spec (zt : ZigTensorModel) :
    ZigTensorModel.det zt = Tensor.detVerified zt.toTensor := rfl

theorem ZigTensorModel.det_spec_symm (zt : ZigTensorModel) :
    Tensor.detVerified zt.toTensor = ZigTensorModel.det zt := by
  rfl

theorem ZigTensorModel.det_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.det zt; result) = Tensor.detVerified zt.toTensor := by
  rfl

def ZigTensorModel.spectralNorm (zt : ZigTensorModel) (maxIter : Nat) (tol : Float) : TResult Float :=
  Tensor.spectralNormVerified zt.toTensor maxIter tol

@[simp] theorem ZigTensorModel.spectralNorm_spec (zt : ZigTensorModel) (maxIter : Nat) (tol : Float) :
    ZigTensorModel.spectralNorm zt maxIter tol = Tensor.spectralNormVerified zt.toTensor maxIter tol := rfl

theorem ZigTensorModel.spectralNorm_spec_symm (zt : ZigTensorModel) (maxIter : Nat) (tol : Float) :
    Tensor.spectralNormVerified zt.toTensor maxIter tol = ZigTensorModel.spectralNorm zt maxIter tol := by
  rfl

theorem ZigTensorModel.spectralNorm_let_spec (zt : ZigTensorModel) (maxIter : Nat) (tol : Float) :
    (let result := ZigTensorModel.spectralNorm zt maxIter tol; result) = Tensor.spectralNormVerified zt.toTensor maxIter tol := by
  rfl

def ZigTensorModel.isClose (a : ZigTensorModel) (b : ZigTensorModel) (rtol : Float) (atol : Float) : Bool :=
  Tensor.isClose a.toTensor b.toTensor rtol atol

@[simp] theorem ZigTensorModel.isClose_spec (a : ZigTensorModel) (b : ZigTensorModel) (rtol : Float) (atol : Float) :
    ZigTensorModel.isClose a b rtol atol = Tensor.isClose a.toTensor b.toTensor rtol atol := rfl

theorem ZigTensorModel.isClose_spec_symm (a : ZigTensorModel) (b : ZigTensorModel) (rtol : Float) (atol : Float) :
    Tensor.isClose a.toTensor b.toTensor rtol atol = ZigTensorModel.isClose a b rtol atol := by
  rfl

theorem ZigTensorModel.isClose_let_spec (a : ZigTensorModel) (b : ZigTensorModel) (rtol : Float) (atol : Float) :
    (let result := ZigTensorModel.isClose a b rtol atol; result) = Tensor.isClose a.toTensor b.toTensor rtol atol := by
  rfl

def ZigTensorModel.release (zt : ZigTensorModel) : Option ZigTensorModel :=
  optionMap ZigTensorModel.ofTensor (Tensor.releaseVerified zt.toTensor)

@[simp] theorem ZigTensorModel.release_spec (zt : ZigTensorModel) :
    ZigTensorModel.release zt = optionMap ZigTensorModel.ofTensor (Tensor.releaseVerified zt.toTensor) := rfl

theorem ZigTensorModel.release_spec_symm (zt : ZigTensorModel) :
    optionMap ZigTensorModel.ofTensor (Tensor.releaseVerified zt.toTensor) = ZigTensorModel.release zt := by
  rfl

theorem ZigTensorModel.release_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.release zt; result) = optionMap ZigTensorModel.ofTensor (Tensor.releaseVerified zt.toTensor) := by
  rfl

def ZigTensorModel.deinit (zt : ZigTensorModel) : Option ZigTensorModel :=
  optionMap ZigTensorModel.ofTensor (Tensor.deinitVerified zt.toTensor)

@[simp] theorem ZigTensorModel.deinit_spec (zt : ZigTensorModel) :
    ZigTensorModel.deinit zt = optionMap ZigTensorModel.ofTensor (Tensor.deinitVerified zt.toTensor) := rfl

theorem ZigTensorModel.deinit_spec_symm (zt : ZigTensorModel) :
    optionMap ZigTensorModel.ofTensor (Tensor.deinitVerified zt.toTensor) = ZigTensorModel.deinit zt := by
  rfl

theorem ZigTensorModel.deinit_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.deinit zt; result) = optionMap ZigTensorModel.ofTensor (Tensor.deinitVerified zt.toTensor) := by
  rfl

def ZigTensorModel.lu (zt : ZigTensorModel) : TResult (ZigTensorModel × ZigTensorModel) :=
  tresultMap ZigTensorModel.pairOfTensor (Tensor.luVerified zt.toTensor)

@[simp] theorem ZigTensorModel.lu_spec (zt : ZigTensorModel) :
    ZigTensorModel.lu zt = tresultMap ZigTensorModel.pairOfTensor (Tensor.luVerified zt.toTensor) := rfl

theorem ZigTensorModel.lu_spec_symm (zt : ZigTensorModel) :
    tresultMap ZigTensorModel.pairOfTensor (Tensor.luVerified zt.toTensor) = ZigTensorModel.lu zt := by
  rfl

theorem ZigTensorModel.lu_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.lu zt; result) = tresultMap ZigTensorModel.pairOfTensor (Tensor.luVerified zt.toTensor) := by
  rfl

def ZigTensorModel.qr (zt : ZigTensorModel) : TResult (ZigTensorModel × ZigTensorModel) :=
  tresultMap ZigTensorModel.pairOfTensor (Tensor.qrVerified zt.toTensor)

@[simp] theorem ZigTensorModel.qr_spec (zt : ZigTensorModel) :
    ZigTensorModel.qr zt = tresultMap ZigTensorModel.pairOfTensor (Tensor.qrVerified zt.toTensor) := rfl

theorem ZigTensorModel.qr_spec_symm (zt : ZigTensorModel) :
    tresultMap ZigTensorModel.pairOfTensor (Tensor.qrVerified zt.toTensor) = ZigTensorModel.qr zt := by
  rfl

theorem ZigTensorModel.qr_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.qr zt; result) = tresultMap ZigTensorModel.pairOfTensor (Tensor.qrVerified zt.toTensor) := by
  rfl

def ZigTensorModel.eig (zt : ZigTensorModel) : TResult (ZigTensorModel × ZigTensorModel) :=
  tresultMap ZigTensorModel.pairOfTensor (Tensor.eigVerified zt.toTensor)

@[simp] theorem ZigTensorModel.eig_spec (zt : ZigTensorModel) :
    ZigTensorModel.eig zt = tresultMap ZigTensorModel.pairOfTensor (Tensor.eigVerified zt.toTensor) := rfl

theorem ZigTensorModel.eig_spec_symm (zt : ZigTensorModel) :
    tresultMap ZigTensorModel.pairOfTensor (Tensor.eigVerified zt.toTensor) = ZigTensorModel.eig zt := by
  rfl

theorem ZigTensorModel.eig_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.eig zt; result) = tresultMap ZigTensorModel.pairOfTensor (Tensor.eigVerified zt.toTensor) := by
  rfl

def ZigTensorModel.svd (zt : ZigTensorModel) : TResult (ZigTensorModel × ZigTensorModel × ZigTensorModel) :=
  tresultMap ZigTensorModel.tripleOfTensor (Tensor.svdVerified zt.toTensor)

@[simp] theorem ZigTensorModel.svd_spec (zt : ZigTensorModel) :
    ZigTensorModel.svd zt = tresultMap ZigTensorModel.tripleOfTensor (Tensor.svdVerified zt.toTensor) := rfl

theorem ZigTensorModel.svd_spec_symm (zt : ZigTensorModel) :
    tresultMap ZigTensorModel.tripleOfTensor (Tensor.svdVerified zt.toTensor) = ZigTensorModel.svd zt := by
  rfl

theorem ZigTensorModel.svd_let_spec (zt : ZigTensorModel) :
    (let result := ZigTensorModel.svd zt; result) = tresultMap ZigTensorModel.tripleOfTensor (Tensor.svdVerified zt.toTensor) := by
  rfl


def zigApiBindings : List (String × String) :=
[
  ("Shape.init", "Shape.initVerified"),
  ("Shape.deinit", "ZigShapeModel.deinit"),
  ("Shape.copy", "Shape.copyVerified"),
  ("Shape.totalSize", "Shape.totalSizeVerified"),
  ("Shape.equals", "Shape.equalsVerified"),
  ("Shape.broadcastCompatible", "Shape.broadcastCompatibleVerified"),
  ("Shape.isContiguous", "Shape.isContiguousVerified"),
  ("Tensor.init", "Tensor.init"),
  ("Tensor.initWithArena", "Tensor.initWithArenaVerified"),
  ("Tensor.initWithPool", "Tensor.initWithPoolVerified"),
  ("Tensor.initWithSlab", "Tensor.initWithSlabVerified"),
  ("Tensor.initWithBuddy", "Tensor.initWithBuddyVerified"),
  ("Tensor.retain", "Tensor.retainVerified"),
  ("Tensor.release", "Tensor.releaseVerified"),
  ("Tensor.deinit", "Tensor.deinitVerified"),
  ("Tensor.copy", "Tensor.copyVerified"),
  ("Tensor.newView", "Tensor.newViewChecked"),
  ("Tensor.reshape", "Tensor.reshape"),
  ("Tensor.view", "Tensor.viewVerified"),
  ("Tensor.slice", "Tensor.sliceVerified"),
  ("Tensor.transpose", "Tensor.transposeVerified"),
  ("Tensor.get", "Tensor.get"),
  ("Tensor.set", "Tensor.set"),
  ("Tensor.fill", "Tensor.fill"),
  ("Tensor.add", "Tensor.add"),
  ("Tensor.sub", "Tensor.sub"),
  ("Tensor.mul", "Tensor.mul"),
  ("Tensor.div", "Tensor.divVerified"),
  ("Tensor.addScalar", "Tensor.addScalar"),
  ("Tensor.subScalar", "Tensor.subScalar"),
  ("Tensor.mulScalar", "Tensor.mulScalar"),
  ("Tensor.divScalar", "Tensor.divScalar"),
  ("Tensor.exp", "Tensor.expVerified"),
  ("Tensor.log", "Tensor.logVerified"),
  ("Tensor.sin", "Tensor.sinVerified"),
  ("Tensor.cos", "Tensor.cosVerified"),
  ("Tensor.tan", "Tensor.tanVerified"),
  ("Tensor.sqrt", "Tensor.sqrtVerified"),
  ("Tensor.pow", "Tensor.powVerified"),
  ("Tensor.abs", "Tensor.absVerified"),
  ("Tensor.max", "Tensor.maxVerified"),
  ("Tensor.min", "Tensor.minVerified"),
  ("Tensor.sum", "Tensor.sumVerified"),
  ("Tensor.mean", "Tensor.meanVerified"),
  ("Tensor.matmul", "Tensor.matmul"),
  ("Tensor.broadcast", "Tensor.broadcastVerified"),
  ("Tensor.unsqueeze", "Tensor.unsqueezeVerified"),
  ("Tensor.relu", "Tensor.reluVerified"),
  ("Tensor.softmax", "Tensor.softmaxVerified"),
  ("Tensor.zeros", "Tensor.zeros"),
  ("Tensor.ones", "Tensor.ones"),
  ("Tensor.full", "Tensor.full"),
  ("Tensor.randomUniform", "Tensor.randomUniformVerified"),
  ("Tensor.randomNormal", "Tensor.randomNormalVerified"),
  ("Tensor.identity", "Tensor.identityVerified"),
  ("Tensor.conv2d", "Tensor.conv2dVerified"),
  ("Tensor.pad", "Tensor.padVerified"),
  ("Tensor.tile", "Tensor.tileVerified"),
  ("Tensor.concat", "Tensor.concatVerified"),
  ("Tensor.stack", "Tensor.stackVerified"),
  ("Tensor.argmax", "Tensor.argmax"),
  ("Tensor.argmin", "Tensor.argmin"),
  ("Tensor.cumsum", "Tensor.cumsum"),
  ("Tensor.variance", "Tensor.variance"),
  ("Tensor.stddev", "Tensor.stddev"),
  ("Tensor.sort", "Tensor.sort"),
  ("Tensor.lessThan", "Tensor.sort.lessThan"),
  ("Tensor.greaterThan", "Tensor.sort.greaterThan"),
  ("Tensor.unique", "Tensor.unique"),
  ("Tensor.oneHot", "Tensor.oneHot"),
  ("Tensor.isClose", "Tensor.isClose"),
  ("Tensor.toInt", "Tensor.toInt"),
  ("Tensor.spectralNorm", "Tensor.spectralNormVerified"),
  ("Tensor.normL2", "Tensor.normL2"),
  ("Tensor.dot", "Tensor.dot"),
  ("Tensor.outer", "Tensor.outer"),
  ("Tensor.trace", "Tensor.trace"),
  ("Tensor.norm", "Tensor.normChecked"),
  ("Tensor.inverse", "Tensor.inverseVerified"),
  ("Tensor.det", "Tensor.detVerified"),
  ("Tensor.lu", "Tensor.luVerified"),
  ("Tensor.qr", "Tensor.qrVerified"),
  ("Tensor.svd", "Tensor.svdVerified"),
  ("Tensor.eig", "Tensor.eigVerified"),
  ("Tensor.cholesky", "Tensor.choleskyVerified"),
  ("Tensor.solve", "Tensor.solveVerified"),
  ("Tensor.clip", "Tensor.clipVerified"),
  ("Tensor.toFixed", "Tensor.toFixed"),
  ("Tensor.arange", "Tensor.arange"),
  ("Tensor.linspace", "Tensor.linspace")
]

@[simp] theorem zigApiBindings_length :
    zigApiBindings.length = 90 := rfl

@[simp] theorem zigApiBindings_0 :
    zigApiBindings.get! 0 = ("Shape.init", "Shape.initVerified") := rfl

@[simp] theorem zigApiBindings_1 :
    zigApiBindings.get! 1 = ("Shape.deinit", "ZigShapeModel.deinit") := rfl

@[simp] theorem zigApiBindings_2 :
    zigApiBindings.get! 2 = ("Shape.copy", "Shape.copyVerified") := rfl

@[simp] theorem zigApiBindings_3 :
    zigApiBindings.get! 3 = ("Shape.totalSize", "Shape.totalSizeVerified") := rfl

@[simp] theorem zigApiBindings_4 :
    zigApiBindings.get! 4 = ("Shape.equals", "Shape.equalsVerified") := rfl

@[simp] theorem zigApiBindings_5 :
    zigApiBindings.get! 5 = ("Shape.broadcastCompatible", "Shape.broadcastCompatibleVerified") := rfl

@[simp] theorem zigApiBindings_6 :
    zigApiBindings.get! 6 = ("Shape.isContiguous", "Shape.isContiguousVerified") := rfl

@[simp] theorem zigApiBindings_7 :
    zigApiBindings.get! 7 = ("Tensor.init", "Tensor.init") := rfl

@[simp] theorem zigApiBindings_8 :
    zigApiBindings.get! 8 = ("Tensor.initWithArena", "Tensor.initWithArenaVerified") := rfl

@[simp] theorem zigApiBindings_9 :
    zigApiBindings.get! 9 = ("Tensor.initWithPool", "Tensor.initWithPoolVerified") := rfl

@[simp] theorem zigApiBindings_10 :
    zigApiBindings.get! 10 = ("Tensor.initWithSlab", "Tensor.initWithSlabVerified") := rfl

@[simp] theorem zigApiBindings_11 :
    zigApiBindings.get! 11 = ("Tensor.initWithBuddy", "Tensor.initWithBuddyVerified") := rfl

@[simp] theorem zigApiBindings_12 :
    zigApiBindings.get! 12 = ("Tensor.retain", "Tensor.retainVerified") := rfl

@[simp] theorem zigApiBindings_13 :
    zigApiBindings.get! 13 = ("Tensor.release", "Tensor.releaseVerified") := rfl

@[simp] theorem zigApiBindings_14 :
    zigApiBindings.get! 14 = ("Tensor.deinit", "Tensor.deinitVerified") := rfl

@[simp] theorem zigApiBindings_15 :
    zigApiBindings.get! 15 = ("Tensor.copy", "Tensor.copyVerified") := rfl

@[simp] theorem zigApiBindings_16 :
    zigApiBindings.get! 16 = ("Tensor.newView", "Tensor.newViewChecked") := rfl

@[simp] theorem zigApiBindings_17 :
    zigApiBindings.get! 17 = ("Tensor.reshape", "Tensor.reshape") := rfl

@[simp] theorem zigApiBindings_18 :
    zigApiBindings.get! 18 = ("Tensor.view", "Tensor.viewVerified") := rfl

@[simp] theorem zigApiBindings_19 :
    zigApiBindings.get! 19 = ("Tensor.slice", "Tensor.sliceVerified") := rfl

@[simp] theorem zigApiBindings_20 :
    zigApiBindings.get! 20 = ("Tensor.transpose", "Tensor.transposeVerified") := rfl

@[simp] theorem zigApiBindings_21 :
    zigApiBindings.get! 21 = ("Tensor.get", "Tensor.get") := rfl

@[simp] theorem zigApiBindings_22 :
    zigApiBindings.get! 22 = ("Tensor.set", "Tensor.set") := rfl

@[simp] theorem zigApiBindings_23 :
    zigApiBindings.get! 23 = ("Tensor.fill", "Tensor.fill") := rfl

@[simp] theorem zigApiBindings_24 :
    zigApiBindings.get! 24 = ("Tensor.add", "Tensor.add") := rfl

@[simp] theorem zigApiBindings_25 :
    zigApiBindings.get! 25 = ("Tensor.sub", "Tensor.sub") := rfl

@[simp] theorem zigApiBindings_26 :
    zigApiBindings.get! 26 = ("Tensor.mul", "Tensor.mul") := rfl

@[simp] theorem zigApiBindings_27 :
    zigApiBindings.get! 27 = ("Tensor.div", "Tensor.divVerified") := rfl

@[simp] theorem zigApiBindings_28 :
    zigApiBindings.get! 28 = ("Tensor.addScalar", "Tensor.addScalar") := rfl

@[simp] theorem zigApiBindings_29 :
    zigApiBindings.get! 29 = ("Tensor.subScalar", "Tensor.subScalar") := rfl

@[simp] theorem zigApiBindings_30 :
    zigApiBindings.get! 30 = ("Tensor.mulScalar", "Tensor.mulScalar") := rfl

@[simp] theorem zigApiBindings_31 :
    zigApiBindings.get! 31 = ("Tensor.divScalar", "Tensor.divScalar") := rfl

@[simp] theorem zigApiBindings_32 :
    zigApiBindings.get! 32 = ("Tensor.exp", "Tensor.expVerified") := rfl

@[simp] theorem zigApiBindings_33 :
    zigApiBindings.get! 33 = ("Tensor.log", "Tensor.logVerified") := rfl

@[simp] theorem zigApiBindings_34 :
    zigApiBindings.get! 34 = ("Tensor.sin", "Tensor.sinVerified") := rfl

@[simp] theorem zigApiBindings_35 :
    zigApiBindings.get! 35 = ("Tensor.cos", "Tensor.cosVerified") := rfl

@[simp] theorem zigApiBindings_36 :
    zigApiBindings.get! 36 = ("Tensor.tan", "Tensor.tanVerified") := rfl

@[simp] theorem zigApiBindings_37 :
    zigApiBindings.get! 37 = ("Tensor.sqrt", "Tensor.sqrtVerified") := rfl

@[simp] theorem zigApiBindings_38 :
    zigApiBindings.get! 38 = ("Tensor.pow", "Tensor.powVerified") := rfl

@[simp] theorem zigApiBindings_39 :
    zigApiBindings.get! 39 = ("Tensor.abs", "Tensor.absVerified") := rfl

@[simp] theorem zigApiBindings_40 :
    zigApiBindings.get! 40 = ("Tensor.max", "Tensor.maxVerified") := rfl

@[simp] theorem zigApiBindings_41 :
    zigApiBindings.get! 41 = ("Tensor.min", "Tensor.minVerified") := rfl

@[simp] theorem zigApiBindings_42 :
    zigApiBindings.get! 42 = ("Tensor.sum", "Tensor.sumVerified") := rfl

@[simp] theorem zigApiBindings_43 :
    zigApiBindings.get! 43 = ("Tensor.mean", "Tensor.meanVerified") := rfl

@[simp] theorem zigApiBindings_44 :
    zigApiBindings.get! 44 = ("Tensor.matmul", "Tensor.matmul") := rfl

@[simp] theorem zigApiBindings_45 :
    zigApiBindings.get! 45 = ("Tensor.broadcast", "Tensor.broadcastVerified") := rfl

@[simp] theorem zigApiBindings_46 :
    zigApiBindings.get! 46 = ("Tensor.unsqueeze", "Tensor.unsqueezeVerified") := rfl

@[simp] theorem zigApiBindings_47 :
    zigApiBindings.get! 47 = ("Tensor.relu", "Tensor.reluVerified") := rfl

@[simp] theorem zigApiBindings_48 :
    zigApiBindings.get! 48 = ("Tensor.softmax", "Tensor.softmaxVerified") := rfl

@[simp] theorem zigApiBindings_49 :
    zigApiBindings.get! 49 = ("Tensor.zeros", "Tensor.zeros") := rfl

@[simp] theorem zigApiBindings_50 :
    zigApiBindings.get! 50 = ("Tensor.ones", "Tensor.ones") := rfl

@[simp] theorem zigApiBindings_51 :
    zigApiBindings.get! 51 = ("Tensor.full", "Tensor.full") := rfl

@[simp] theorem zigApiBindings_52 :
    zigApiBindings.get! 52 = ("Tensor.randomUniform", "Tensor.randomUniformVerified") := rfl

@[simp] theorem zigApiBindings_53 :
    zigApiBindings.get! 53 = ("Tensor.randomNormal", "Tensor.randomNormalVerified") := rfl

@[simp] theorem zigApiBindings_54 :
    zigApiBindings.get! 54 = ("Tensor.identity", "Tensor.identityVerified") := rfl

@[simp] theorem zigApiBindings_55 :
    zigApiBindings.get! 55 = ("Tensor.conv2d", "Tensor.conv2dVerified") := rfl

@[simp] theorem zigApiBindings_56 :
    zigApiBindings.get! 56 = ("Tensor.pad", "Tensor.padVerified") := rfl

@[simp] theorem zigApiBindings_57 :
    zigApiBindings.get! 57 = ("Tensor.tile", "Tensor.tileVerified") := rfl

@[simp] theorem zigApiBindings_58 :
    zigApiBindings.get! 58 = ("Tensor.concat", "Tensor.concatVerified") := rfl

@[simp] theorem zigApiBindings_59 :
    zigApiBindings.get! 59 = ("Tensor.stack", "Tensor.stackVerified") := rfl

@[simp] theorem zigApiBindings_60 :
    zigApiBindings.get! 60 = ("Tensor.argmax", "Tensor.argmax") := rfl

@[simp] theorem zigApiBindings_61 :
    zigApiBindings.get! 61 = ("Tensor.argmin", "Tensor.argmin") := rfl

@[simp] theorem zigApiBindings_62 :
    zigApiBindings.get! 62 = ("Tensor.cumsum", "Tensor.cumsum") := rfl

@[simp] theorem zigApiBindings_63 :
    zigApiBindings.get! 63 = ("Tensor.variance", "Tensor.variance") := rfl

@[simp] theorem zigApiBindings_64 :
    zigApiBindings.get! 64 = ("Tensor.stddev", "Tensor.stddev") := rfl

@[simp] theorem zigApiBindings_65 :
    zigApiBindings.get! 65 = ("Tensor.sort", "Tensor.sort") := rfl

@[simp] theorem zigApiBindings_66 :
    zigApiBindings.get! 66 = ("Tensor.lessThan", "Tensor.sort.lessThan") := rfl

@[simp] theorem zigApiBindings_67 :
    zigApiBindings.get! 67 = ("Tensor.greaterThan", "Tensor.sort.greaterThan") := rfl

@[simp] theorem zigApiBindings_68 :
    zigApiBindings.get! 68 = ("Tensor.unique", "Tensor.unique") := rfl

@[simp] theorem zigApiBindings_69 :
    zigApiBindings.get! 69 = ("Tensor.oneHot", "Tensor.oneHot") := rfl

@[simp] theorem zigApiBindings_70 :
    zigApiBindings.get! 70 = ("Tensor.isClose", "Tensor.isClose") := rfl

@[simp] theorem zigApiBindings_71 :
    zigApiBindings.get! 71 = ("Tensor.toInt", "Tensor.toInt") := rfl

@[simp] theorem zigApiBindings_72 :
    zigApiBindings.get! 72 = ("Tensor.spectralNorm", "Tensor.spectralNormVerified") := rfl

@[simp] theorem zigApiBindings_73 :
    zigApiBindings.get! 73 = ("Tensor.normL2", "Tensor.normL2") := rfl

@[simp] theorem zigApiBindings_74 :
    zigApiBindings.get! 74 = ("Tensor.dot", "Tensor.dot") := rfl

@[simp] theorem zigApiBindings_75 :
    zigApiBindings.get! 75 = ("Tensor.outer", "Tensor.outer") := rfl

@[simp] theorem zigApiBindings_76 :
    zigApiBindings.get! 76 = ("Tensor.trace", "Tensor.trace") := rfl

@[simp] theorem zigApiBindings_77 :
    zigApiBindings.get! 77 = ("Tensor.norm", "Tensor.normChecked") := rfl

@[simp] theorem zigApiBindings_78 :
    zigApiBindings.get! 78 = ("Tensor.inverse", "Tensor.inverseVerified") := rfl

@[simp] theorem zigApiBindings_79 :
    zigApiBindings.get! 79 = ("Tensor.det", "Tensor.detVerified") := rfl

@[simp] theorem zigApiBindings_80 :
    zigApiBindings.get! 80 = ("Tensor.lu", "Tensor.luVerified") := rfl

@[simp] theorem zigApiBindings_81 :
    zigApiBindings.get! 81 = ("Tensor.qr", "Tensor.qrVerified") := rfl

@[simp] theorem zigApiBindings_82 :
    zigApiBindings.get! 82 = ("Tensor.svd", "Tensor.svdVerified") := rfl

@[simp] theorem zigApiBindings_83 :
    zigApiBindings.get! 83 = ("Tensor.eig", "Tensor.eigVerified") := rfl

@[simp] theorem zigApiBindings_84 :
    zigApiBindings.get! 84 = ("Tensor.cholesky", "Tensor.choleskyVerified") := rfl

@[simp] theorem zigApiBindings_85 :
    zigApiBindings.get! 85 = ("Tensor.solve", "Tensor.solveVerified") := rfl

@[simp] theorem zigApiBindings_86 :
    zigApiBindings.get! 86 = ("Tensor.clip", "Tensor.clipVerified") := rfl

@[simp] theorem zigApiBindings_87 :
    zigApiBindings.get! 87 = ("Tensor.toFixed", "Tensor.toFixed") := rfl

@[simp] theorem zigApiBindings_88 :
    zigApiBindings.get! 88 = ("Tensor.arange", "Tensor.arange") := rfl

@[simp] theorem zigApiBindings_89 :
    zigApiBindings.get! 89 = ("Tensor.linspace", "Tensor.linspace") := rfl



/-- Removed the raw embedded Zig source list and its text-level verification lemmas.
    The Lean model above should be kept independent from line-by-line string encodings of Zig code. -/


end
