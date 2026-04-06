namespace SFDVerified

inductive Precision where
  | fp4 : Precision
  | fp8 : Precision
  | fp16 : Precision
  | fp32 : Precision
  | fp64 : Precision
deriving Repr, Inhabited

namespace Precision

def toNat : Precision → Nat
  | .fp4 => 0
  | .fp8 => 1
  | .fp16 => 2
  | .fp32 => 3
  | .fp64 => 4

def fromNat : Nat → Option Precision
  | 0 => some .fp4
  | 1 => some .fp8
  | 2 => some .fp16
  | 3 => some .fp32
  | 4 => some .fp64
  | _ => none

def bitWidth : Precision → Nat
  | .fp4 => 4
  | .fp8 => 8
  | .fp16 => 16
  | .fp32 => 32
  | .fp64 => 64

def maxRangeNat : Precision → Nat
  | .fp4 => 8
  | .fp8 => 448
  | .fp16 => 65504
  | .fp32 => 3402823466
  | .fp64 => 0

def precisionLe : Precision → Precision → Bool
  | .fp4, .fp4 => true
  | .fp4, .fp8 => true
  | .fp4, .fp16 => true
  | .fp4, .fp32 => true
  | .fp4, .fp64 => true
  | .fp8, .fp8 => true
  | .fp8, .fp16 => true
  | .fp8, .fp32 => true
  | .fp8, .fp64 => true
  | .fp16, .fp16 => true
  | .fp16, .fp32 => true
  | .fp16, .fp64 => true
  | .fp32, .fp32 => true
  | .fp32, .fp64 => true
  | .fp64, .fp64 => true
  | _, _ => false

def precisionLt : Precision → Precision → Bool
  | .fp4, .fp8 => true
  | .fp4, .fp16 => true
  | .fp4, .fp32 => true
  | .fp4, .fp64 => true
  | .fp8, .fp16 => true
  | .fp8, .fp32 => true
  | .fp8, .fp64 => true
  | .fp16, .fp32 => true
  | .fp16, .fp64 => true
  | .fp32, .fp64 => true
  | _, _ => false

def beq : Precision → Precision → Bool
  | .fp4, .fp4 => true
  | .fp8, .fp8 => true
  | .fp16, .fp16 => true
  | .fp32, .fp32 => true
  | .fp64, .fp64 => true
  | _, _ => false

instance : BEq Precision := ⟨beq⟩

theorem roundtrip_fp4 :
    fromNat (toNat Precision.fp4) = some Precision.fp4 :=
  rfl

theorem roundtrip_fp8 :
    fromNat (toNat Precision.fp8) = some Precision.fp8 :=
  rfl

theorem roundtrip_fp16 :
    fromNat (toNat Precision.fp16) = some Precision.fp16 :=
  rfl

theorem roundtrip_fp32 :
    fromNat (toNat Precision.fp32) = some Precision.fp32 :=
  rfl

theorem roundtrip_fp64 :
    fromNat (toNat Precision.fp64) = some Precision.fp64 :=
  rfl

theorem roundtrip_all (p : Precision) :
    fromNat (toNat p) = some p :=
  match p with
  | .fp4 => rfl
  | .fp8 => rfl
  | .fp16 => rfl
  | .fp32 => rfl
  | .fp64 => rfl

theorem bitWidth_pos (p : Precision) :
    0 < bitWidth p :=
  match p with
  | .fp4 => Nat.zero_lt_succ 3
  | .fp8 => Nat.zero_lt_succ 7
  | .fp16 => Nat.zero_lt_succ 15
  | .fp32 => Nat.zero_lt_succ 31
  | .fp64 => Nat.zero_lt_succ 63

theorem toNat_lt_five (p : Precision) :
    toNat p < 5 :=
  match p with
  | .fp4 => Nat.zero_lt_succ 4
  | .fp8 => Nat.succ_lt_succ (Nat.zero_lt_succ 3)
  | .fp16 => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 2))
  | .fp32 => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1)))
  | .fp64 => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0))))

theorem fp4_ne_fp8 : Precision.fp4 ≠ Precision.fp8 :=
  fun h =>
    have h' : toNat Precision.fp4 = toNat Precision.fp8 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp4_ne_fp16 : Precision.fp4 ≠ Precision.fp16 :=
  fun h =>
    have h' : toNat Precision.fp4 = toNat Precision.fp16 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp4_ne_fp32 : Precision.fp4 ≠ Precision.fp32 :=
  fun h =>
    have h' : toNat Precision.fp4 = toNat Precision.fp32 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp4_ne_fp64 : Precision.fp4 ≠ Precision.fp64 :=
  fun h =>
    have h' : toNat Precision.fp4 = toNat Precision.fp64 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp8_ne_fp4 : Precision.fp8 ≠ Precision.fp4 :=
  fun h =>
    have h' : toNat Precision.fp8 = toNat Precision.fp4 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp8_ne_fp16 : Precision.fp8 ≠ Precision.fp16 :=
  fun h =>
    have h' : toNat Precision.fp8 = toNat Precision.fp16 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' => Nat.noConfusion h'')

theorem fp8_ne_fp32 : Precision.fp8 ≠ Precision.fp32 :=
  fun h =>
    have h' : toNat Precision.fp8 = toNat Precision.fp32 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' => Nat.noConfusion h'')

theorem fp8_ne_fp64 : Precision.fp8 ≠ Precision.fp64 :=
  fun h =>
    have h' : toNat Precision.fp8 = toNat Precision.fp64 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' => Nat.noConfusion h'')

theorem fp16_ne_fp4 : Precision.fp16 ≠ Precision.fp4 :=
  fun h =>
    have h' : toNat Precision.fp16 = toNat Precision.fp4 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp16_ne_fp8 : Precision.fp16 ≠ Precision.fp8 :=
  fun h =>
    have h' : toNat Precision.fp16 = toNat Precision.fp8 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' => Nat.noConfusion h'')

theorem fp16_ne_fp32 : Precision.fp16 ≠ Precision.fp32 :=
  fun h =>
    have h' : toNat Precision.fp16 = toNat Precision.fp32 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' =>
      Nat.noConfusion h'' (fun h''' => Nat.noConfusion h'''))

theorem fp16_ne_fp64 : Precision.fp16 ≠ Precision.fp64 :=
  fun h =>
    have h' : toNat Precision.fp16 = toNat Precision.fp64 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' =>
      Nat.noConfusion h'' (fun h''' => Nat.noConfusion h'''))

theorem fp32_ne_fp4 : Precision.fp32 ≠ Precision.fp4 :=
  fun h =>
    have h' : toNat Precision.fp32 = toNat Precision.fp4 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp32_ne_fp8 : Precision.fp32 ≠ Precision.fp8 :=
  fun h =>
    have h' : toNat Precision.fp32 = toNat Precision.fp8 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' => Nat.noConfusion h'')

theorem fp32_ne_fp16 : Precision.fp32 ≠ Precision.fp16 :=
  fun h =>
    have h' : toNat Precision.fp32 = toNat Precision.fp16 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' =>
      Nat.noConfusion h'' (fun h''' => Nat.noConfusion h'''))

theorem fp32_ne_fp64 : Precision.fp32 ≠ Precision.fp64 :=
  fun h =>
    have h' : toNat Precision.fp32 = toNat Precision.fp64 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' =>
      Nat.noConfusion h'' (fun h''' =>
        Nat.noConfusion h''' (fun h'''' => Nat.noConfusion h'''')))

theorem fp64_ne_fp4 : Precision.fp64 ≠ Precision.fp4 :=
  fun h =>
    have h' : toNat Precision.fp64 = toNat Precision.fp4 :=
      congrArg toNat h
    Nat.noConfusion h'

theorem fp64_ne_fp8 : Precision.fp64 ≠ Precision.fp8 :=
  fun h =>
    have h' : toNat Precision.fp64 = toNat Precision.fp8 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' => Nat.noConfusion h'')

theorem fp64_ne_fp16 : Precision.fp64 ≠ Precision.fp16 :=
  fun h =>
    have h' : toNat Precision.fp64 = toNat Precision.fp16 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' =>
      Nat.noConfusion h'' (fun h''' => Nat.noConfusion h'''))

theorem fp64_ne_fp32 : Precision.fp64 ≠ Precision.fp32 :=
  fun h =>
    have h' : toNat Precision.fp64 = toNat Precision.fp32 :=
      congrArg toNat h
    Nat.noConfusion h' (fun h'' =>
      Nat.noConfusion h'' (fun h''' =>
        Nat.noConfusion h''' (fun h'''' => Nat.noConfusion h'''')))

theorem toNat_injective (a b : Precision) (h : toNat a = toNat b) : a = b :=
  match a, b with
  | .fp4, .fp4 => rfl
  | .fp4, .fp8 => Nat.noConfusion h
  | .fp4, .fp16 => Nat.noConfusion h
  | .fp4, .fp32 => Nat.noConfusion h
  | .fp4, .fp64 => Nat.noConfusion h
  | .fp8, .fp4 => Nat.noConfusion h
  | .fp8, .fp8 => rfl
  | .fp8, .fp16 => Nat.noConfusion h (fun h' => Nat.noConfusion h')
  | .fp8, .fp32 => Nat.noConfusion h (fun h' => Nat.noConfusion h')
  | .fp8, .fp64 => Nat.noConfusion h (fun h' => Nat.noConfusion h')
  | .fp16, .fp4 => Nat.noConfusion h
  | .fp16, .fp8 => Nat.noConfusion h (fun h' => Nat.noConfusion h')
  | .fp16, .fp16 => rfl
  | .fp16, .fp32 => Nat.noConfusion h (fun h' => Nat.noConfusion h' (fun h'' => Nat.noConfusion h''))
  | .fp16, .fp64 => Nat.noConfusion h (fun h' => Nat.noConfusion h' (fun h'' => Nat.noConfusion h''))
  | .fp32, .fp4 => Nat.noConfusion h
  | .fp32, .fp8 => Nat.noConfusion h (fun h' => Nat.noConfusion h')
  | .fp32, .fp16 => Nat.noConfusion h (fun h' => Nat.noConfusion h' (fun h'' => Nat.noConfusion h''))
  | .fp32, .fp32 => rfl
  | .fp32, .fp64 => Nat.noConfusion h (fun h' => Nat.noConfusion h' (fun h'' => Nat.noConfusion h'' (fun h''' => Nat.noConfusion h''')))
  | .fp64, .fp4 => Nat.noConfusion h
  | .fp64, .fp8 => Nat.noConfusion h (fun h' => Nat.noConfusion h')
  | .fp64, .fp16 => Nat.noConfusion h (fun h' => Nat.noConfusion h' (fun h'' => Nat.noConfusion h''))
  | .fp64, .fp32 => Nat.noConfusion h (fun h' => Nat.noConfusion h' (fun h'' => Nat.noConfusion h'' (fun h''' => Nat.noConfusion h''')))
  | .fp64, .fp64 => rfl

theorem beq_refl (p : Precision) : beq p p = true :=
  match p with
  | .fp4 => rfl
  | .fp8 => rfl
  | .fp16 => rfl
  | .fp32 => rfl
  | .fp64 => rfl

theorem beq_symm (a b : Precision) : beq a b = beq b a :=
  match a, b with
  | .fp4, .fp4 => rfl
  | .fp4, .fp8 => rfl
  | .fp4, .fp16 => rfl
  | .fp4, .fp32 => rfl
  | .fp4, .fp64 => rfl
  | .fp8, .fp4 => rfl
  | .fp8, .fp8 => rfl
  | .fp8, .fp16 => rfl
  | .fp8, .fp32 => rfl
  | .fp8, .fp64 => rfl
  | .fp16, .fp4 => rfl
  | .fp16, .fp8 => rfl
  | .fp16, .fp16 => rfl
  | .fp16, .fp32 => rfl
  | .fp16, .fp64 => rfl
  | .fp32, .fp4 => rfl
  | .fp32, .fp8 => rfl
  | .fp32, .fp16 => rfl
  | .fp32, .fp32 => rfl
  | .fp32, .fp64 => rfl
  | .fp64, .fp4 => rfl
  | .fp64, .fp8 => rfl
  | .fp64, .fp16 => rfl
  | .fp64, .fp32 => rfl
  | .fp64, .fp64 => rfl

theorem precisionLe_refl (p : Precision) : precisionLe p p = true :=
  match p with
  | .fp4 => rfl
  | .fp8 => rfl
  | .fp16 => rfl
  | .fp32 => rfl
  | .fp64 => rfl

theorem precisionLt_irrefl (p : Precision) : precisionLt p p = false :=
  match p with
  | .fp4 => rfl
  | .fp8 => rfl
  | .fp16 => rfl
  | .fp32 => rfl
  | .fp64 => rfl

theorem precisionLt_implies_le (a b : Precision) (h : precisionLt a b = true) :
    precisionLe a b = true :=
  match a, b with
  | .fp4, .fp4 => absurd h (fun h' => Bool.noConfusion h')
  | .fp4, .fp8 => rfl
  | .fp4, .fp16 => rfl
  | .fp4, .fp32 => rfl
  | .fp4, .fp64 => rfl
  | .fp8, .fp4 => absurd h (fun h' => Bool.noConfusion h')
  | .fp8, .fp8 => absurd h (fun h' => Bool.noConfusion h')
  | .fp8, .fp16 => rfl
  | .fp8, .fp32 => rfl
  | .fp8, .fp64 => rfl
  | .fp16, .fp4 => absurd h (fun h' => Bool.noConfusion h')
  | .fp16, .fp8 => absurd h (fun h' => Bool.noConfusion h')
  | .fp16, .fp16 => absurd h (fun h' => Bool.noConfusion h')
  | .fp16, .fp32 => rfl
  | .fp16, .fp64 => rfl
  | .fp32, .fp4 => absurd h (fun h' => Bool.noConfusion h')
  | .fp32, .fp8 => absurd h (fun h' => Bool.noConfusion h')
  | .fp32, .fp16 => absurd h (fun h' => Bool.noConfusion h')
  | .fp32, .fp32 => absurd h (fun h' => Bool.noConfusion h')
  | .fp32, .fp64 => rfl
  | .fp64, .fp4 => absurd h (fun h' => Bool.noConfusion h')
  | .fp64, .fp8 => absurd h (fun h' => Bool.noConfusion h')
  | .fp64, .fp16 => absurd h (fun h' => Bool.noConfusion h')
  | .fp64, .fp32 => absurd h (fun h' => Bool.noConfusion h')
  | .fp64, .fp64 => absurd h (fun h' => Bool.noConfusion h')

theorem bitWidth_fp4_lt_fp8 : bitWidth .fp4 < bitWidth .fp8 :=
  Nat.lt.step (Nat.lt.step (Nat.lt.step (Nat.lt.step (Nat.lt.base 3))))

theorem bitWidth_fp8_lt_fp16 : bitWidth .fp8 < bitWidth .fp16 :=
  Nat.lt.step (Nat.lt.step (Nat.lt.step (Nat.lt.step
    (Nat.lt.step (Nat.lt.step (Nat.lt.step (Nat.lt.base 7)))))))

theorem bitWidth_monotone_fp4_fp16 : bitWidth .fp4 < bitWidth .fp16 :=
  Nat.lt_trans bitWidth_fp4_lt_fp8 bitWidth_fp8_lt_fp16

theorem bitWidth_le_64 (p : Precision) : bitWidth p ≤ 64 :=
  match p with
  | .fp4 => Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
        (Nat.le_refl 4)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  | .fp8 => Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
        (Nat.le_refl 8))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  | .fp16 => Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
        (Nat.le_refl 16)))))))))))))))))))))))))))))))))))))))))))))))))
  | .fp32 => Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
      (Nat.le_step (Nat.le_step (Nat.le_step (Nat.le_step
        (Nat.le_refl 32)))))))))))))))))))))))))))))))))
  | .fp64 => Nat.le_refl 64

end Precision

structure FxP where
  raw : Int
deriving Repr, Inhabited, DecidableEq

namespace FxP

def scale : Nat := 10000

def zero : FxP := ⟨0⟩
def one : FxP := ⟨(scale : Int)⟩
def negOne : FxP := ⟨-(scale : Int)⟩
def half : FxP := ⟨(scale / 2 : Int)⟩

def fromInt (n : Int) : FxP := ⟨n * (scale : Int)⟩
def fromNat (n : Nat) : FxP := ⟨(n : Int) * (scale : Int)⟩

def add (a b : FxP) : FxP := ⟨a.raw + b.raw⟩
def sub (a b : FxP) : FxP := ⟨a.raw - b.raw⟩
def mul (a b : FxP) : FxP := ⟨(a.raw * b.raw) / (scale : Int)⟩
def neg (a : FxP) : FxP := ⟨-a.raw⟩
def abs (a : FxP) : FxP := ⟨Int.natAbs a.raw⟩

def le (a b : FxP) : Prop := a.raw ≤ b.raw
def lt (a b : FxP) : Prop := a.raw < b.raw
def ge (a b : FxP) : Prop := a.raw ≥ b.raw
def gt (a b : FxP) : Prop := a.raw > b.raw

def max (a b : FxP) : FxP :=
  if a.raw ≥ b.raw then a else b

def min (a b : FxP) : FxP :=
  if a.raw ≤ b.raw then a else b

def clamp (v lo hi : FxP) : FxP :=
  max lo (min v hi)

def isNonNeg (a : FxP) : Bool :=
  a.raw ≥ 0

def isPositive (a : FxP) : Bool :=
  a.raw > 0

theorem add_raw (a b : FxP) :
    (add a b).raw = a.raw + b.raw :=
  rfl

theorem sub_raw (a b : FxP) :
    (sub a b).raw = a.raw - b.raw :=
  rfl

theorem neg_raw (a : FxP) :
    (neg a).raw = -a.raw :=
  rfl

theorem mul_raw (a b : FxP) :
    (mul a b).raw = (a.raw * b.raw) / (scale : Int) :=
  rfl

theorem add_comm (a b : FxP) :
    add a b = add b a :=
  show FxP.mk (a.raw + b.raw) = FxP.mk (b.raw + a.raw) from
    congrArg FxP.mk (Int.add_comm a.raw b.raw)

theorem add_assoc (a b c : FxP) :
    add (add a b) c = add a (add b c) :=
  show FxP.mk ((a.raw + b.raw) + c.raw) = FxP.mk (a.raw + (b.raw + c.raw)) from
    congrArg FxP.mk (Int.add_assoc a.raw b.raw c.raw)

theorem add_zero_right (a : FxP) :
    add a zero = a :=
  show FxP.mk (a.raw + 0) = a from
    have h : a.raw + 0 = a.raw := Int.add_zero a.raw
    Eq.trans (congrArg FxP.mk h) rfl

theorem add_zero_left (a : FxP) :
    add zero a = a :=
  show FxP.mk (0 + a.raw) = a from
    have h : (0 : Int) + a.raw = a.raw := Int.zero_add a.raw
    Eq.trans (congrArg FxP.mk h) rfl

theorem sub_self (a : FxP) :
    sub a a = zero :=
  show FxP.mk (a.raw - a.raw) = FxP.mk 0 from
    congrArg FxP.mk (Int.sub_self a.raw)

theorem neg_neg (a : FxP) :
    neg (neg a) = a :=
  show FxP.mk (- -a.raw) = a from
    have h : - -a.raw = a.raw := Int.neg_neg a.raw
    Eq.trans (congrArg FxP.mk h) rfl

theorem add_neg_cancel (a : FxP) :
    add a (neg a) = zero :=
  show FxP.mk (a.raw + -a.raw) = FxP.mk 0 from
    congrArg FxP.mk (Int.add_right_neg a.raw)

theorem neg_add_cancel (a : FxP) :
    add (neg a) a = zero :=
  Eq.trans (add_comm (neg a) a) (add_neg_cancel a)

theorem sub_eq_add_neg (a b : FxP) :
    sub a b = add a (neg b) :=
  show FxP.mk (a.raw - b.raw) = FxP.mk (a.raw + -b.raw) from
    congrArg FxP.mk (Int.sub_eq_add_neg a.raw b.raw)

theorem mul_comm (a b : FxP) :
    mul a b = mul b a :=
  show FxP.mk ((a.raw * b.raw) / (scale : Int)) =
       FxP.mk ((b.raw * a.raw) / (scale : Int)) from
    congrArg FxP.mk (congrArg (· / (scale : Int)) (Int.mul_comm a.raw b.raw))

theorem mul_zero_right (a : FxP) :
    mul a zero = zero :=
  show FxP.mk ((a.raw * 0) / (scale : Int)) = FxP.mk 0 from
    have h1 : a.raw * 0 = 0 := Int.mul_zero a.raw
    have h2 : (0 : Int) / (scale : Int) = 0 := Int.zero_div (scale : Int)
    congrArg FxP.mk (Eq.trans (congrArg (· / (scale : Int)) h1) h2)

theorem mul_zero_left (a : FxP) :
    mul zero a = zero :=
  Eq.trans (mul_comm zero a) (mul_zero_right a)

theorem neg_zero : neg zero = zero :=
  show FxP.mk (-(0 : Int)) = FxP.mk 0 from
    congrArg FxP.mk (Int.neg_zero)

theorem fromInt_zero : fromInt 0 = zero :=
  show FxP.mk ((0 : Int) * (scale : Int)) = FxP.mk 0 from
    congrArg FxP.mk (Int.zero_mul (scale : Int))

theorem add_sub_cancel (a b : FxP) :
    sub (add a b) b = a :=
  show FxP.mk ((a.raw + b.raw) - b.raw) = a from
    have h : (a.raw + b.raw) - b.raw = a.raw := Int.add_sub_cancel a.raw b.raw
    Eq.trans (congrArg FxP.mk h) rfl

theorem sub_add_cancel (a b : FxP) :
    add (sub a b) b = a :=
  show FxP.mk ((a.raw - b.raw) + b.raw) = a from
    have h : (a.raw - b.raw) + b.raw = a.raw := Int.sub_add_cancel a.raw b.raw
    Eq.trans (congrArg FxP.mk h) rfl

theorem max_comm (a b : FxP) :
    max a b = max b a :=
  show (if a.raw ≥ b.raw then a else b) = (if b.raw ≥ a.raw then b else a) from
    if h1 : a.raw ≥ b.raw then
      if h2 : b.raw ≥ a.raw then
        have hab : a.raw = b.raw := Int.le_antisymm h1 h2
        have : a = b := congrArg FxP.mk (Eq.symm hab)
        Eq.trans (dite_cond_eq_true (a.raw ≥ b.raw) h1 (fun _ => a) (fun _ => b))
          (Eq.trans this (Eq.symm (dite_cond_eq_true (b.raw ≥ a.raw) h2 (fun _ => b) (fun _ => a))))
      else
        dite_cond_eq_true (a.raw ≥ b.raw) h1 (fun _ => a) (fun _ => b)
    else
      dite_cond_eq_false (a.raw ≥ b.raw) h1 (fun _ => a) (fun _ => b)

end FxP

structure TensorFlags where
  in_tensor_memory : Bool := false
  requires_grad : Bool := true
  is_compressed : Bool := false
deriving Repr, Inhabited, DecidableEq

namespace TensorFlags

def default : TensorFlags := ⟨false, true, false⟩

def toBits (f : TensorFlags) : Nat :=
  (if f.in_tensor_memory then 1 else 0) +
  (if f.requires_grad then 2 else 0) +
  (if f.is_compressed then 4 else 0)

def fromBits (n : Nat) : TensorFlags :=
  { in_tensor_memory := n % 2 == 1
  , requires_grad := (n / 2) % 2 == 1
  , is_compressed := (n / 4) % 2 == 1 }

def flagsBeq (a b : TensorFlags) : Bool :=
  a.in_tensor_memory == b.in_tensor_memory &&
  a.requires_grad == b.requires_grad &&
  a.is_compressed == b.is_compressed

theorem default_in_tensor_memory : default.in_tensor_memory = false := rfl
theorem default_requires_grad : default.requires_grad = true := rfl
theorem default_is_compressed : default.is_compressed = false := rfl

theorem toBits_default : toBits default = 2 := rfl

theorem toBits_all_false :
    toBits ⟨false, false, false⟩ = 0 := rfl

theorem toBits_all_true :
    toBits ⟨true, true, true⟩ = 7 := rfl

theorem toBits_only_memory :
    toBits ⟨true, false, false⟩ = 1 := rfl

theorem toBits_only_grad :
    toBits ⟨false, true, false⟩ = 2 := rfl

theorem toBits_only_compressed :
    toBits ⟨false, false, true⟩ = 4 := rfl

theorem toBits_memory_grad :
    toBits ⟨true, true, false⟩ = 3 := rfl

theorem toBits_memory_compressed :
    toBits ⟨true, false, true⟩ = 5 := rfl

theorem toBits_grad_compressed :
    toBits ⟨false, true, true⟩ = 6 := rfl

theorem toBits_bounded (f : TensorFlags) :
    toBits f < 8 :=
  match f.in_tensor_memory, f.requires_grad, f.is_compressed with
  | false, false, false => Nat.zero_lt_succ 7
  | true, false, false => Nat.succ_lt_succ (Nat.zero_lt_succ 6)
  | false, true, false => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 5))
  | true, true, false => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 4)))
  | false, false, true => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 3))))
  | true, false, true => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 2)))))
  | false, true, true => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1))))))
  | true, true, true => Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0)))))))

theorem roundtrip_all_false :
    fromBits (toBits ⟨false, false, false⟩) = ⟨false, false, false⟩ := rfl

theorem roundtrip_all_true :
    fromBits (toBits ⟨true, true, true⟩) = ⟨true, true, true⟩ := rfl

theorem roundtrip_only_memory :
    fromBits (toBits ⟨true, false, false⟩) = ⟨true, false, false⟩ := rfl

theorem roundtrip_only_grad :
    fromBits (toBits ⟨false, true, false⟩) = ⟨false, true, false⟩ := rfl

theorem roundtrip_only_compressed :
    fromBits (toBits ⟨false, false, true⟩) = ⟨false, false, true⟩ := rfl

theorem roundtrip_memory_grad :
    fromBits (toBits ⟨true, true, false⟩) = ⟨true, true, false⟩ := rfl

theorem roundtrip_memory_compressed :
    fromBits (toBits ⟨true, false, true⟩) = ⟨true, false, true⟩ := rfl

theorem roundtrip_grad_compressed :
    fromBits (toBits ⟨false, true, true⟩) = ⟨false, true, true⟩ := rfl

theorem flagsBeq_refl (f : TensorFlags) : flagsBeq f f = true :=
  match f.in_tensor_memory, f.requires_grad, f.is_compressed with
  | false, false, false => rfl
  | true, false, false => rfl
  | false, true, false => rfl
  | true, true, false => rfl
  | false, false, true => rfl
  | true, false, true => rfl
  | false, true, true => rfl
  | true, true, true => rfl

theorem flagsBeq_symm (a b : TensorFlags) :
    flagsBeq a b = flagsBeq b a :=
  match a.in_tensor_memory, a.requires_grad, a.is_compressed,
        b.in_tensor_memory, b.requires_grad, b.is_compressed with
  | false, false, false, false, false, false => rfl
  | false, false, false, true, false, false => rfl
  | false, false, false, false, true, false => rfl
  | false, false, false, true, true, false => rfl
  | false, false, false, false, false, true => rfl
  | false, false, false, true, false, true => rfl
  | false, false, false, false, true, true => rfl
  | false, false, false, true, true, true => rfl
  | true, false, false, false, false, false => rfl
  | true, false, false, true, false, false => rfl
  | true, false, false, false, true, false => rfl
  | true, false, false, true, true, false => rfl
  | true, false, false, false, false, true => rfl
  | true, false, false, true, false, true => rfl
  | true, false, false, false, true, true => rfl
  | true, false, false, true, true, true => rfl
  | false, true, false, false, false, false => rfl
  | false, true, false, true, false, false => rfl
  | false, true, false, false, true, false => rfl
  | false, true, false, true, true, false => rfl
  | false, true, false, false, false, true => rfl
  | false, true, false, true, false, true => rfl
  | false, true, false, false, true, true => rfl
  | false, true, false, true, true, true => rfl
  | true, true, false, false, false, false => rfl
  | true, true, false, true, false, false => rfl
  | true, true, false, false, true, false => rfl
  | true, true, false, true, true, false => rfl
  | true, true, false, false, false, true => rfl
  | true, true, false, true, false, true => rfl
  | true, true, false, false, true, true => rfl
  | true, true, false, true, true, true => rfl
  | false, false, true, false, false, false => rfl
  | false, false, true, true, false, false => rfl
  | false, false, true, false, true, false => rfl
  | false, false, true, true, true, false => rfl
  | false, false, true, false, false, true => rfl
  | false, false, true, true, false, true => rfl
  | false, false, true, false, true, true => rfl
  | false, false, true, true, true, true => rfl
  | true, false, true, false, false, false => rfl
  | true, false, true, true, false, false => rfl
  | true, false, true, false, true, false => rfl
  | true, false, true, true, true, false => rfl
  | true, false, true, false, false, true => rfl
  | true, false, true, true, false, true => rfl
  | true, false, true, false, true, true => rfl
  | true, false, true, true, true, true => rfl
  | false, true, true, false, false, false => rfl
  | false, true, true, true, false, false => rfl
  | false, true, true, false, true, false => rfl
  | false, true, true, true, true, false => rfl
  | false, true, true, false, false, true => rfl
  | false, true, true, true, false, true => rfl
  | false, true, true, false, true, true => rfl
  | false, true, true, true, true, true => rfl
  | true, true, true, false, false, false => rfl
  | true, true, true, true, false, false => rfl
  | true, true, true, false, true, false => rfl
  | true, true, true, true, true, false => rfl
  | true, true, true, false, false, true => rfl
  | true, true, true, true, false, true => rfl
  | true, true, true, false, true, true => rfl
  | true, true, true, true, true, true => rfl

end TensorFlags

structure Shape where
  dims : List Nat
deriving Repr, Inhabited, DecidableEq

namespace Shape

def totalSize (s : Shape) : Nat :=
  List.foldl (· * ·) 1 s.dims

def ndim (s : Shape) : Nat := s.dims.length

def empty : Shape := ⟨[]⟩

def scalar : Shape := ⟨[]⟩

def vector (n : Nat) : Shape := ⟨[n]⟩

def matrix (m n : Nat) : Shape := ⟨[m, n]⟩

def tensor3d (a b c : Nat) : Shape := ⟨[a, b, c]⟩

def shapesEqual (a b : Shape) : Bool :=
  a.dims == b.dims

def isScalar (s : Shape) : Bool :=
  s.dims.length == 0

def isVector (s : Shape) : Bool :=
  s.dims.length == 1

def isMatrix (s : Shape) : Bool :=
  s.dims.length == 2

theorem totalSize_empty : totalSize empty = 1 := rfl

theorem totalSize_scalar : totalSize scalar = 1 := rfl

theorem ndim_empty : ndim empty = 0 := rfl

theorem ndim_scalar : ndim scalar = 0 := rfl

theorem ndim_vector (n : Nat) : ndim (vector n) = 1 := rfl

theorem ndim_matrix (m n : Nat) : ndim (matrix m n) = 2 := rfl

theorem ndim_tensor3d (a b c : Nat) : ndim (tensor3d a b c) = 3 := rfl

theorem isScalar_empty : isScalar empty = true := rfl
theorem isScalar_scalar : isScalar scalar = true := rfl
theorem isVector_vector (n : Nat) : isVector (vector n) = true := rfl
theorem isMatrix_matrix (m n : Nat) : isMatrix (matrix m n) = true := rfl

theorem shapesEqual_refl (s : Shape) : shapesEqual s s = true :=
  show (s.dims == s.dims) = true from
    List.beq_self_eq_true s.dims

def foldlMulAppend :
    ∀ (xs ys : List Nat) (acc : Nat),
    List.foldl (· * ·) acc (xs ++ ys) =
    List.foldl (· * ·) (List.foldl (· * ·) acc xs) ys :=
  fun xs =>
    @List.recOn Nat
      (fun xs => ∀ (ys : List Nat) (acc : Nat),
        List.foldl (· * ·) acc (xs ++ ys) =
        List.foldl (· * ·) (List.foldl (· * ·) acc xs) ys)
      xs
      (fun ys acc => rfl)
      (fun x xs' ih ys acc => ih ys (acc * x))

def foldlMulScaleAcc :
    ∀ (xs : List Nat) (a b : Nat),
    List.foldl (· * ·) (a * b) xs =
    a * List.foldl (· * ·) b xs :=
  fun xs =>
    @List.recOn Nat
      (fun xs => ∀ (a b : Nat),
        List.foldl (· * ·) (a * b) xs =
        a * List.foldl (· * ·) b xs)
      xs
      (fun a b => rfl)
      (fun x xs' ih a b =>
        have step1 : List.foldl (· * ·) (a * b * x) xs' =
                     List.foldl (· * ·) (a * (b * x)) xs' :=
          congrArg (fun v => List.foldl (· * ·) v xs') (Nat.mul_assoc a b x)
        have step2 : List.foldl (· * ·) (a * (b * x)) xs' =
                     a * List.foldl (· * ·) (b * x) xs' :=
          ih a (b * x)
        Eq.trans step1 step2)

theorem totalSize_append (s1 s2 : Shape) :
    totalSize ⟨s1.dims ++ s2.dims⟩ = totalSize s1 * totalSize s2 :=
  show List.foldl (· * ·) 1 (s1.dims ++ s2.dims) =
       List.foldl (· * ·) 1 s1.dims * List.foldl (· * ·) 1 s2.dims from
    have h1 : List.foldl (· * ·) 1 (s1.dims ++ s2.dims) =
              List.foldl (· * ·) (List.foldl (· * ·) 1 s1.dims) s2.dims :=
      foldlMulAppend s1.dims s2.dims 1
    have h2 : List.foldl (· * ·) (List.foldl (· * ·) 1 s1.dims) s2.dims =
              List.foldl (· * ·) (List.foldl (· * ·) 1 s1.dims * 1) s2.dims :=
      congrArg (fun v => List.foldl (· * ·) v s2.dims)
        (Eq.symm (Nat.mul_one (List.foldl (· * ·) 1 s1.dims)))
    have h3 : List.foldl (· * ·) (List.foldl (· * ·) 1 s1.dims * 1) s2.dims =
              List.foldl (· * ·) 1 s1.dims * List.foldl (· * ·) 1 s2.dims :=
      foldlMulScaleAcc s2.dims (List.foldl (· * ·) 1 s1.dims) 1
    Eq.trans h1 (Eq.trans h2 h3)

def allPositive : List Nat → Prop
  | [] => True
  | x :: xs => 0 < x ∧ allPositive xs

theorem allPositive_nil : allPositive [] = True := rfl

theorem allPositive_cons (x : Nat) (xs : List Nat) :
    allPositive (x :: xs) = (0 < x ∧ allPositive xs) := rfl

def foldlMulPos :
    ∀ (xs : List Nat) (acc : Nat),
    0 < acc → allPositive xs → 0 < List.foldl (· * ·) acc xs :=
  fun xs =>
    @List.recOn Nat
      (fun xs => ∀ (acc : Nat),
        0 < acc → allPositive xs → 0 < List.foldl (· * ·) acc xs)
      xs
      (fun acc hacc _ => hacc)
      (fun x xs' ih acc hacc hap =>
        have hx : 0 < x := hap.1
        have hxs' : allPositive xs' := hap.2
        have haccx : 0 < acc * x := Nat.mul_pos hacc hx
        ih (acc * x) haccx hxs')

theorem totalSize_pos (s : Shape) (h : allPositive s.dims) :
    0 < totalSize s :=
  foldlMulPos s.dims 1 (Nat.zero_lt_succ 0) h

theorem totalSize_pos_nonempty (n : Nat) (ns : List Nat) (hn : 0 < n) (hns : allPositive ns) :
    0 < totalSize ⟨n :: ns⟩ :=
  totalSize_pos ⟨n :: ns⟩ (And.intro hn hns)

theorem vector_totalSize (n : Nat) :
    totalSize (vector n) = 1 * n * 1 := rfl

theorem matrix_totalSize (m n : Nat) :
    totalSize (matrix m n) = 1 * m * n * 1 := rfl

end Shape

structure TensorData where
  values : List Int
  shape : Shape
  precision : Precision
  flags : TensorFlags
deriving Repr

namespace TensorData

def size (t : TensorData) : Nat := t.values.length

def empty : TensorData :=
  { values := []
  , shape := Shape.empty
  , precision := .fp32
  , flags := TensorFlags.default }

def zeros (dims : List Nat) : TensorData :=
  { values := List.replicate (Shape.totalSize ⟨dims⟩) 0
  , shape := ⟨dims⟩
  , precision := .fp32
  , flags := TensorFlags.default }

def ones (dims : List Nat) : TensorData :=
  { values := List.replicate (Shape.totalSize ⟨dims⟩) (FxP.scale : Int)
  , shape := ⟨dims⟩
  , precision := .fp32
  , flags := TensorFlags.default }

def fill (dims : List Nat) (v : Int) : TensorData :=
  { values := List.replicate (Shape.totalSize ⟨dims⟩) v
  , shape := ⟨dims⟩
  , precision := .fp32
  , flags := TensorFlags.default }

def fromList (dims : List Nat) (vs : List Int) : TensorData :=
  { values := vs
  , shape := ⟨dims⟩
  , precision := .fp32
  , flags := TensorFlags.default }

def addElementwise (a b : TensorData) : Option TensorData :=
  if a.values.length == b.values.length then
    some { values := List.zipWith (· + ·) a.values b.values
         , shape := a.shape
         , precision := a.precision
         , flags := a.flags }
  else none

def subElementwise (a b : TensorData) : Option TensorData :=
  if a.values.length == b.values.length then
    some { values := List.zipWith (· - ·) a.values b.values
         , shape := a.shape
         , precision := a.precision
         , flags := a.flags }
  else none

def scaleElementwise (t : TensorData) (s : Int) : TensorData :=
  { t with values := List.map (· * s / (FxP.scale : Int)) t.values }

def normSqInt (t : TensorData) : Int :=
  List.foldl (fun acc v => acc + v * v) 0 t.values

def shapeValid (t : TensorData) : Prop :=
  t.values.length = Shape.totalSize t.shape

def clone (t : TensorData) : TensorData :=
  { values := t.values
  , shape := t.shape
  , precision := t.precision
  , flags := t.flags }

theorem clone_eq (t : TensorData) : clone t = t := rfl

theorem clone_values (t : TensorData) : (clone t).values = t.values := rfl

theorem clone_shape (t : TensorData) : (clone t).shape = t.shape := rfl

theorem clone_precision (t : TensorData) : (clone t).precision = t.precision := rfl

theorem clone_flags (t : TensorData) : (clone t).flags = t.flags := rfl

theorem zeros_length (dims : List Nat) :
    (zeros dims).values.length = Shape.totalSize ⟨dims⟩ :=
  List.length_replicate (Shape.totalSize ⟨dims⟩) 0

theorem ones_length (dims : List Nat) :
    (ones dims).values.length = Shape.totalSize ⟨dims⟩ :=
  List.length_replicate (Shape.totalSize ⟨dims⟩) (FxP.scale : Int)

theorem fill_length (dims : List Nat) (v : Int) :
    (fill dims v).values.length = Shape.totalSize ⟨dims⟩ :=
  List.length_replicate (Shape.totalSize ⟨dims⟩) v

theorem zeros_shapeValid (dims : List Nat) :
    shapeValid (zeros dims) :=
  zeros_length dims

theorem ones_shapeValid (dims : List Nat) :
    shapeValid (ones dims) :=
  ones_length dims

theorem fill_shapeValid (dims : List Nat) (v : Int) :
    shapeValid (fill dims v) :=
  fill_length dims v

theorem empty_size : size empty = 0 := rfl

theorem zeros_shape (dims : List Nat) : (zeros dims).shape = ⟨dims⟩ := rfl

theorem ones_shape (dims : List Nat) : (ones dims).shape = ⟨dims⟩ := rfl

theorem fill_shape (dims : List Nat) (v : Int) : (fill dims v).shape = ⟨dims⟩ := rfl

theorem zeros_precision (dims : List Nat) : (zeros dims).precision = .fp32 := rfl

theorem ones_precision (dims : List Nat) : (ones dims).precision = .fp32 := rfl

theorem scaleElementwise_shape (t : TensorData) (s : Int) :
    (scaleElementwise t s).shape = t.shape := rfl

theorem scaleElementwise_precision (t : TensorData) (s : Int) :
    (scaleElementwise t s).precision = t.precision := rfl

theorem scaleElementwise_flags (t : TensorData) (s : Int) :
    (scaleElementwise t s).flags = t.flags := rfl

def zipWithLength {α β γ : Type} (f : α → β → γ) :
    ∀ (xs : List α) (ys : List β),
    (List.zipWith f xs ys).length = Nat.min xs.length ys.length :=
  fun xs =>
    @List.recOn α
      (fun xs => ∀ (ys : List β),
        (List.zipWith f xs ys).length = Nat.min xs.length ys.length)
      xs
      (fun ys => rfl)
      (fun x xs' ih ys =>
        match ys with
        | [] => rfl
        | y :: ys' =>
          show (f x y :: List.zipWith f xs' ys').length =
               Nat.min (xs'.length + 1) (ys'.length + 1) from
            have step : (f x y :: List.zipWith f xs' ys').length =
                        (List.zipWith f xs' ys').length + 1 := rfl
            have ih_step : (List.zipWith f xs' ys').length =
                           Nat.min xs'.length ys'.length := ih ys'
            have min_succ : Nat.min (xs'.length + 1) (ys'.length + 1) =
                            Nat.min xs'.length ys'.length + 1 := Nat.succ_min_succ xs'.length ys'.length
            Eq.trans step (Eq.trans (congrArg (· + 1) ih_step) (Eq.symm min_succ)))

def normSqNonNeg :
    ∀ (vs : List Int) (acc : Int),
    0 ≤ acc → 0 ≤ List.foldl (fun a v => a + v * v) acc vs :=
  fun vs =>
    @List.recOn Int
      (fun vs => ∀ (acc : Int),
        0 ≤ acc → 0 ≤ List.foldl (fun a v => a + v * v) acc vs)
      vs
      (fun acc hacc => hacc)
      (fun v vs' ih acc hacc =>
        have hsq : 0 ≤ v * v := Int.mul_self_nonneg v
        have hsum : 0 ≤ acc + v * v := Int.add_nonneg hacc hsq
        ih (acc + v * v) hsum)

theorem normSqInt_nonneg (t : TensorData) :
    0 ≤ normSqInt t :=
  normSqNonNeg t.values 0 (Int.le_refl 0)

end TensorData

inductive LRScheduleType where
  | cosine_annealing : LRScheduleType
  | cosine_annealing_with_warmup : LRScheduleType| polynomial_decay : LRScheduleType
  | exponential_decay : LRScheduleType
  | one_cycle : LRScheduleType
  | sophia_style : LRScheduleType
deriving Repr, Inhabited, DecidableEq

namespace LRScheduleType

def beq : LRScheduleType → LRScheduleType → Bool
  | cosine_annealing, cosine_annealing => true
  | cosine_annealing_with_warmup, cosine_annealing_with_warmup => true
  | polynomial_decay, polynomial_decay => true
  | exponential_decay, exponential_decay => true
  | one_cycle, one_cycle => true
  | sophia_style, sophia_style => true
  | _, _ => false

instance : BEq LRScheduleType := ⟨beq⟩

theorem beq_refl (x : LRScheduleType) : beq x x = true :=
  match x with
  | cosine_annealing => rfl
  | cosine_annealing_with_warmup => rfl
  | polynomial_decay => rfl
  | exponential_decay => rfl
  | one_cycle => rfl
  | sophia_style => rfl

theorem beq_symm (x y : LRScheduleType) : beq x y = beq y x :=
  match x, y with
  | cosine_annealing, cosine_annealing => rfl
  | cosine_annealing, cosine_annealing_with_warmup => rfl
  | cosine_annealing, polynomial_decay => rfl
  | cosine_annealing, exponential_decay => rfl
  | cosine_annealing, one_cycle => rfl
  | cosine_annealing, sophia_style => rfl
  | cosine_annealing_with_warmup, cosine_annealing => rfl
  | cosine_annealing_with_warmup, cosine_annealing_with_warmup => rfl
  | cosine_annealing_with_warmup, polynomial_decay => rfl
  | cosine_annealing_with_warmup, exponential_decay => rfl
  | cosine_annealing_with_warmup, one_cycle => rfl
  | cosine_annealing_with_warmup, sophia_style => rfl
  | polynomial_decay, cosine_annealing => rfl
  | polynomial_decay, cosine_annealing_with_warmup => rfl
  | polynomial_decay, polynomial_decay => rfl
  | polynomial_decay, exponential_decay => rfl
  | polynomial_decay, one_cycle => rfl
  | polynomial_decay, sophia_style => rfl
  | exponential_decay, cosine_annealing => rfl
  | exponential_decay, cosine_annealing_with_warmup => rfl
  | exponential_decay, polynomial_decay => rfl
  | exponential_decay, exponential_decay => rfl
  | exponential_decay, one_cycle => rfl
  | exponential_decay, sophia_style => rfl
  | one_cycle, cosine_annealing => rfl
  | one_cycle, cosine_annealing_with_warmup => rfl
  | one_cycle, polynomial_decay => rfl
  | one_cycle, exponential_decay => rfl
  | one_cycle, one_cycle => rfl
  | one_cycle, sophia_style => rfl
  | sophia_style, cosine_annealing => rfl
  | sophia_style, cosine_annealing_with_warmup => rfl
  | sophia_style, polynomial_decay => rfl
  | sophia_style, exponential_decay => rfl
  | sophia_style, one_cycle => rfl
  | sophia_style, sophia_style => rfl

theorem cosine_annealing_ne_cosine_annealing_with_warmup : cosine_annealing ≠ cosine_annealing_with_warmup := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_ne_polynomial_decay : cosine_annealing ≠ polynomial_decay := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_ne_exponential_decay : cosine_annealing ≠ exponential_decay := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_ne_one_cycle : cosine_annealing ≠ one_cycle := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_ne_sophia_style : cosine_annealing ≠ sophia_style := fun h => LRScheduleType.noConfusion h

theorem cosine_annealing_with_warmup_ne_cosine_annealing : cosine_annealing_with_warmup ≠ cosine_annealing := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_with_warmup_ne_polynomial_decay : cosine_annealing_with_warmup ≠ polynomial_decay := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_with_warmup_ne_exponential_decay : cosine_annealing_with_warmup ≠ exponential_decay := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_with_warmup_ne_one_cycle : cosine_annealing_with_warmup ≠ one_cycle := fun h => LRScheduleType.noConfusion h
theorem cosine_annealing_with_warmup_ne_sophia_style : cosine_annealing_with_warmup ≠ sophia_style := fun h => LRScheduleType.noConfusion h

theorem polynomial_decay_ne_cosine_annealing : polynomial_decay ≠ cosine_annealing := fun h => LRScheduleType.noConfusion h
theorem polynomial_decay_ne_cosine_annealing_with_warmup : polynomial_decay ≠ cosine_annealing_with_warmup := fun h => LRScheduleType.noConfusion h
theorem polynomial_decay_ne_exponential_decay : polynomial_decay ≠ exponential_decay := fun h => LRScheduleType.noConfusion h
theorem polynomial_decay_ne_one_cycle : polynomial_decay ≠ one_cycle := fun h => LRScheduleType.noConfusion h
theorem polynomial_decay_ne_sophia_style : polynomial_decay ≠ sophia_style := fun h => LRScheduleType.noConfusion h

theorem exponential_decay_ne_cosine_annealing : exponential_decay ≠ cosine_annealing := fun h => LRScheduleType.noConfusion h
theorem exponential_decay_ne_cosine_annealing_with_warmup : exponential_decay ≠ cosine_annealing_with_warmup := fun h => LRScheduleType.noConfusion h
theorem exponential_decay_ne_polynomial_decay : exponential_decay ≠ polynomial_decay := fun h => LRScheduleType.noConfusion h
theorem exponential_decay_ne_one_cycle : exponential_decay ≠ one_cycle := fun h => LRScheduleType.noConfusion h
theorem exponential_decay_ne_sophia_style : exponential_decay ≠ sophia_style := fun h => LRScheduleType.noConfusion h

theorem one_cycle_ne_cosine_annealing : one_cycle ≠ cosine_annealing := fun h => LRScheduleType.noConfusion h
theorem one_cycle_ne_cosine_annealing_with_warmup : one_cycle ≠ cosine_annealing_with_warmup := fun h => LRScheduleType.noConfusion h
theorem one_cycle_ne_polynomial_decay : one_cycle ≠ polynomial_decay := fun h => LRScheduleType.noConfusion h
theorem one_cycle_ne_exponential_decay : one_cycle ≠ exponential_decay := fun h => LRScheduleType.noConfusion h
theorem one_cycle_ne_sophia_style : one_cycle ≠ sophia_style := fun h => LRScheduleType.noConfusion h

theorem sophia_style_ne_cosine_annealing : sophia_style ≠ cosine_annealing := fun h => LRScheduleType.noConfusion h
theorem sophia_style_ne_cosine_annealing_with_warmup : sophia_style ≠ cosine_annealing_with_warmup := fun h => LRScheduleType.noConfusion h
theorem sophia_style_ne_polynomial_decay : sophia_style ≠ polynomial_decay := fun h => LRScheduleType.noConfusion h
theorem sophia_style_ne_exponential_decay : sophia_style ≠ exponential_decay := fun h => LRScheduleType.noConfusion h
theorem sophia_style_ne_one_cycle : sophia_style ≠ one_cycle := fun h => LRScheduleType.noConfusion h

end LRScheduleType

structure LRScheduler where
  schedule_type : LRScheduleType
  base_lr : FxP
  min_lr : FxP
  max_lr : FxP
  warmup_steps : Nat
  total_steps : Nat
  current_step : Nat
deriving Repr

namespace LRScheduler

def Invariant (sched : LRScheduler) : Prop :=
  sched.current_step ≤ sched.total_steps

structure Valid where
  sched : LRScheduler
  valid : Invariant sched

def init (schedule_type : LRScheduleType) (base_lr : FxP) (warmup_steps total_steps : Nat) : LRScheduler :=
  { schedule_type := schedule_type
  , base_lr := base_lr
  , min_lr := FxP.zero
  , max_lr := FxP.add base_lr base_lr
  , warmup_steps := warmup_steps
  , total_steps := total_steps
  , current_step := 0 }

theorem init_valid (schedule_type : LRScheduleType) (base_lr : FxP) (warmup_steps total_steps : Nat) :
    Invariant (init schedule_type base_lr warmup_steps total_steps) :=
  Nat.zero_le total_steps

def initValid (schedule_type : LRScheduleType) (base_lr : FxP) (warmup_steps total_steps : Nat) : Valid :=
  ⟨init schedule_type base_lr warmup_steps total_steps, init_valid schedule_type base_lr warmup_steps total_steps⟩

def step (sched : LRScheduler) : LRScheduler :=
  if sched.current_step < sched.total_steps then
    { sched with current_step := Nat.succ sched.current_step }
  else
    sched

theorem step_valid (v : Valid) : Invariant (step v.sched) :=
  if h : v.sched.current_step < v.sched.total_steps then
    have h_step : step v.sched = { v.sched with current_step := Nat.succ v.sched.current_step } := dif_pos h
    have h_inv : Invariant { v.sched with current_step := Nat.succ v.sched.current_step } := h
    Eq.subst (Eq.symm h_step) h_inv
  else
    have h_step : step v.sched = v.sched := dif_neg h
    Eq.subst (Eq.symm h_step) v.valid

def stepValid (v : Valid) : Valid :=
  ⟨step v.sched, step_valid v⟩

theorem step_mono (v : Valid) (h : v.sched.current_step < v.sched.total_steps) :
    v.sched.current_step < (step v.sched).current_step :=
  have h_step : step v.sched = { v.sched with current_step := Nat.succ v.sched.current_step } := dif_pos h
  have h_curr : (step v.sched).current_step = Nat.succ v.sched.current_step := congrArg current_step h_step
  have h_lt : v.sched.current_step < Nat.succ v.sched.current_step := Nat.lt_succ_self v.sched.current_step
  Eq.subst (Eq.symm h_curr) h_lt

theorem step_idempotent_at_end (v : Valid) (h : v.sched.current_step = v.sched.total_steps) :
    step v.sched = v.sched :=
  have h_not_lt : ¬(v.sched.current_step < v.sched.total_steps) :=
    fun hlt => Nat.ne_of_lt hlt h
  dif_neg h_not_lt

def clone (sched : LRScheduler) : LRScheduler :=
  { schedule_type := sched.schedule_type
  , base_lr := sched.base_lr
  , min_lr := sched.min_lr
  , max_lr := sched.max_lr
  , warmup_steps := sched.warmup_steps
  , total_steps := sched.total_steps
  , current_step := sched.current_step }

theorem clone_eq (sched : LRScheduler) : clone sched = sched := rfl

end LRScheduler

structure DynamicLossScaler where
  scale : FxP
  growth_factor : FxP
  backoff_factor : FxP
  growth_interval : Nat
  steps_since_last_overflow : Nat
deriving Repr

namespace DynamicLossScaler

def Invariant (scaler : DynamicLossScaler) : Prop :=
  scaler.steps_since_last_overflow ≤ scaler.growth_interval

structure Valid where
  scaler : DynamicLossScaler
  valid : Invariant scaler

def init (initial_scale : FxP) : DynamicLossScaler :=
  { scale := initial_scale
  , growth_factor := FxP.add FxP.one FxP.one
  , backoff_factor := FxP.half
  , growth_interval := 2000
  , steps_since_last_overflow := 0 }

theorem init_valid (initial_scale : FxP) : Invariant (init initial_scale) :=
  Nat.zero_le 2000

def initValid (initial_scale : FxP) : Valid :=
  ⟨init initial_scale, init_valid initial_scale⟩

def update (scaler : DynamicLossScaler) (has_overflow : Bool) : DynamicLossScaler :=
  match has_overflow with
  | true =>
    { scaler with scale := FxP.mul scaler.scale scaler.backoff_factor
                , steps_since_last_overflow := 0 }
  | false =>
    if scaler.steps_since_last_overflow + 1 ≥ scaler.growth_interval then
      { scaler with scale := FxP.mul scaler.scale scaler.growth_factor
                  , steps_since_last_overflow := 0 }
    else
      { scaler with steps_since_last_overflow := Nat.succ scaler.steps_since_last_overflow }

theorem update_valid (v : Valid) (has_overflow : Bool) : Invariant (update v.scaler has_overflow) :=
  match has_overflow with
  | true =>
    have h_update : update v.scaler true = { v.scaler with scale := FxP.mul v.scaler.scale v.scaler.backoff_factor, steps_since_last_overflow := 0 } := rfl
    have h_inv : Invariant { v.scaler with scale := FxP.mul v.scaler.scale v.scaler.backoff_factor, steps_since_last_overflow := 0 } := Nat.zero_le v.scaler.growth_interval
    Eq.subst (Eq.symm h_update) h_inv
  | false =>
    if h_ge : v.scaler.steps_since_last_overflow + 1 ≥ v.scaler.growth_interval then
      have h_update : update v.scaler false = { v.scaler with scale := FxP.mul v.scaler.scale v.scaler.growth_factor, steps_since_last_overflow := 0 } := dif_pos h_ge
      have h_inv : Invariant { v.scaler with scale := FxP.mul v.scaler.scale v.scaler.growth_factor, steps_since_last_overflow := 0 } := Nat.zero_le v.scaler.growth_interval
      Eq.subst (Eq.symm h_update) h_inv
    else
      have h_update : update v.scaler false = { v.scaler with steps_since_last_overflow := Nat.succ v.scaler.steps_since_last_overflow } := dif_neg h_ge
      have h_lt : v.scaler.steps_since_last_overflow + 1 < v.scaler.growth_interval := Nat.lt_of_not_ge h_ge
      have h_inv : Invariant { v.scaler with steps_since_last_overflow := Nat.succ v.scaler.steps_since_last_overflow } := h_lt
      Eq.subst (Eq.symm h_update) h_inv

def updateValid (v : Valid) (has_overflow : Bool) : Valid :=
  ⟨update v.scaler has_overflow, update_valid v has_overflow⟩

def clone (scaler : DynamicLossScaler) : DynamicLossScaler :=
  { scale := scaler.scale
  , growth_factor := scaler.growth_factor
  , backoff_factor := scaler.backoff_factor
  , growth_interval := scaler.growth_interval
  , steps_since_last_overflow := scaler.steps_since_last_overflow }

theorem clone_eq (scaler : DynamicLossScaler) : clone scaler = scaler := rfl

end DynamicLossScaler

structure MixedPrecisionConfig where
  use_fp4 : Bool
  use_fp8 : Bool
  use_fp16 : Bool
  master_weights_precision : Precision
  gradient_accumulation_steps : Nat
  loss_scale : FxP
  dynamic_loss_scaling : Bool
deriving Repr

namespace MixedPrecisionConfig

def default : MixedPrecisionConfig :=
  { use_fp4 := true
  , use_fp8 := true
  , use_fp16 := true
  , master_weights_precision := .fp32
  , gradient_accumulation_steps := 4
  , loss_scale := FxP.fromNat 1024
  , dynamic_loss_scaling := true }

def clone (config : MixedPrecisionConfig) : MixedPrecisionConfig :=
  { use_fp4 := config.use_fp4
  , use_fp8 := config.use_fp8
  , use_fp16 := config.use_fp16
  , master_weights_precision := config.master_weights_precision
  , gradient_accumulation_steps := config.gradient_accumulation_steps
  , loss_scale := config.loss_scale
  , dynamic_loss_scaling := config.dynamic_loss_scaling }

theorem clone_eq (config : MixedPrecisionConfig) : clone config = config := rfl

end MixedPrecisionConfig

def list_map_length {α β : Type} (f : α → β) : ∀ (xs : List α), (List.map f xs).length = xs.length :=
  fun xs => @List.recOn α (fun xs => (List.map f xs).length = xs.length) xs rfl (fun x xs' ih => congrArg Nat.succ ih)

structure MixedPrecisionTrainer where
  config : MixedPrecisionConfig
  master_weights : List TensorData
  working_weights : List TensorData
  accumulated_gradients : List TensorData
  accumulation_counter : Nat
  loss_scaler : DynamicLossScaler
deriving Repr

namespace MixedPrecisionTrainer

def Invariant (trainer : MixedPrecisionTrainer) : Prop :=
  trainer.master_weights.length = trainer.working_weights.length ∧
  trainer.working_weights.length = trainer.accumulated_gradients.length ∧
  DynamicLossScaler.Invariant trainer.loss_scaler

structure Valid where
  trainer : MixedPrecisionTrainer
  valid : Invariant trainer

def init (config : MixedPrecisionConfig) (shapes : List Shape) : MixedPrecisionTrainer :=
  { config := config
  , master_weights := List.map (fun s => TensorData.zeros s.dims) shapes
  , working_weights := List.map (fun s => TensorData.zeros s.dims) shapes
  , accumulated_gradients := List.map (fun s => TensorData.zeros s.dims) shapes
  , accumulation_counter := 0
  , loss_scaler := DynamicLossScaler.init config.loss_scale }

theorem init_valid (config : MixedPrecisionConfig) (shapes : List Shape) :
    Invariant (init config shapes) :=
  have h_master : (List.map (fun s => TensorData.zeros s.dims) shapes).length = shapes.length := list_map_length (fun s => TensorData.zeros s.dims) shapes
  have h_working : (List.map (fun s => TensorData.zeros s.dims) shapes).length = shapes.length := list_map_length (fun s => TensorData.zeros s.dims) shapes
  have h_accum : (List.map (fun s => TensorData.zeros s.dims) shapes).length = shapes.length := list_map_length (fun s => TensorData.zeros s.dims) shapes
  have h_mw : (List.map (fun s => TensorData.zeros s.dims) shapes).length = (List.map (fun s => TensorData.zeros s.dims) shapes).length := Eq.trans h_master (Eq.symm h_working)
  have h_wa : (List.map (fun s => TensorData.zeros s.dims) shapes).length = (List.map (fun s => TensorData.zeros s.dims) shapes).length := Eq.trans h_working (Eq.symm h_accum)
  have h_scaler : DynamicLossScaler.Invariant (DynamicLossScaler.init config.loss_scale) := DynamicLossScaler.init_valid config.loss_scale
  And.intro h_mw (And.intro h_wa h_scaler)

def initValid (config : MixedPrecisionConfig) (shapes : List Shape) : Valid :=
  ⟨init config shapes, init_valid config shapes⟩

def updateWeights (trainer : MixedPrecisionTrainer) (lr : FxP) : MixedPrecisionTrainer :=
  { trainer with accumulation_counter := 0 }

theorem updateWeights_valid (v : Valid) (lr : FxP) :
    Invariant (updateWeights v.trainer lr) :=
  have h_update : updateWeights v.trainer lr = { v.trainer with accumulation_counter := 0 } := rfl
  have h_inv : Invariant { v.trainer with accumulation_counter := 0 } := v.valid
  Eq.subst (Eq.symm h_update) h_inv

def updateWeightsValid (v : Valid) (lr : FxP) : Valid :=
  ⟨updateWeights v.trainer lr, updateWeights_valid v lr⟩

def clone (trainer : MixedPrecisionTrainer) : MixedPrecisionTrainer :=
  { config := trainer.config
  , master_weights := trainer.master_weights
  , working_weights := trainer.working_weights
  , accumulated_gradients := trainer.accumulated_gradients
  , accumulation_counter := trainer.accumulation_counter
  , loss_scaler := trainer.loss_scaler }

theorem clone_eq (trainer : MixedPrecisionTrainer) : clone trainer = trainer := rfl

end MixedPrecisionTrainer

structure B200OptimizationConfig where
  use_fp4_tensor_cores : Bool
  use_tensor_memory : Bool
  nvlink_bandwidth_tbps : FxP
  hbm_bandwidth_tbps : FxP
  decompression_engine : Bool
  multi_instance_gpu : Bool
  l2_cache_size_mb : Nat
  tmem_size_mb : Nat
deriving Repr

namespace B200OptimizationConfig

def default : B200OptimizationConfig :=
  { use_fp4_tensor_cores := true
  , use_tensor_memory := true
  , nvlink_bandwidth_tbps := FxP.fromNat 1
  , hbm_bandwidth_tbps := FxP.fromNat 8
  , decompression_engine := true
  , multi_instance_gpu := false
  , l2_cache_size_mb := 50
  , tmem_size_mb := 32 }

def clone (config : B200OptimizationConfig) : B200OptimizationConfig :=
  { use_fp4_tensor_cores := config.use_fp4_tensor_cores
  , use_tensor_memory := config.use_tensor_memory
  , nvlink_bandwidth_tbps := config.nvlink_bandwidth_tbps
  , hbm_bandwidth_tbps := config.hbm_bandwidth_tbps
  , decompression_engine := config.decompression_engine
  , multi_instance_gpu := config.multi_instance_gpu
  , l2_cache_size_mb := config.l2_cache_size_mb
  , tmem_size_mb := config.tmem_size_mb }

theorem clone_eq (config : B200OptimizationConfig) : clone config = config := rfl

end B200OptimizationConfig

structure B200MemoryManager where
  config : B200OptimizationConfig
  tensor_memory_used : Nat
  prefetch_queue : List Nat
deriving Repr

namespace B200MemoryManager

def Invariant (manager : B200MemoryManager) : Prop :=
  manager.tensor_memory_used ≤ manager.config.tmem_size_mb * 1024 * 1024

structure Valid where
  manager : B200MemoryManager
  valid : Invariant manager

def init (config : B200OptimizationConfig) : B200MemoryManager :=
  { config := config
  , tensor_memory_used := 0
  , prefetch_queue :=[] }

theorem init_valid (config : B200OptimizationConfig) :
    Invariant (init config) :=
  Nat.zero_le (config.tmem_size_mb * 1024 * 1024)

def initValid (config : B200OptimizationConfig) : Valid :=
  ⟨init config, init_valid config⟩

def allocate (manager : B200MemoryManager) (size : Nat) : B200MemoryManager :=
  if manager.tensor_memory_used + size ≤ manager.config.tmem_size_mb * 1024 * 1024 then
    { manager with tensor_memory_used := manager.tensor_memory_used + size }
  else
    manager

theorem allocate_valid (v : Valid) (size : Nat) :
    Invariant (allocate v.manager size) :=
  if h : v.manager.tensor_memory_used + size ≤ v.manager.config.tmem_size_mb * 1024 * 1024 then
    have h_alloc : allocate v.manager size = { v.manager with tensor_memory_used := v.manager.tensor_memory_used + size } := dif_pos h
    have h_inv : Invariant { v.manager with tensor_memory_used := v.manager.tensor_memory_used + size } := h
    Eq.subst (Eq.symm h_alloc) h_inv
  else
    have h_alloc : allocate v.manager size = v.manager := dif_neg h
    Eq.subst (Eq.symm h_alloc) v.valid

def allocateValid (v : Valid) (size : Nat) : Valid :=
  ⟨allocate v.manager size, allocate_valid v size⟩

def free (manager : B200MemoryManager) (size : Nat) : B200MemoryManager :=
  if size ≤ manager.tensor_memory_used then
    { manager with tensor_memory_used := manager.tensor_memory_used - size }
  else
    { manager with tensor_memory_used := 0 }

theorem free_valid (v : Valid) (size : Nat) :
    Invariant (free v.manager size) :=
  if h : size ≤ v.manager.tensor_memory_used then
    have h_free : free v.manager size = { v.manager with tensor_memory_used := v.manager.tensor_memory_used - size } := dif_pos h
    have h_sub_le : v.manager.tensor_memory_used - size ≤ v.manager.tensor_memory_used := Nat.sub_le v.manager.tensor_memory_used size
    have h_inv : Invariant { v.manager with tensor_memory_used := v.manager.tensor_memory_used - size } := Nat.le_trans h_sub_le v.valid
    Eq.subst (Eq.symm h_free) h_inv
  else
    have h_free : free v.manager size = { v.manager with tensor_memory_used := 0 } := dif_neg h
    have h_inv : Invariant { v.manager with tensor_memory_used := 0 } := Nat.zero_le (v.manager.config.tmem_size_mb * 1024 * 1024)
    Eq.subst (Eq.symm h_free) h_inv

def freeValid (v : Valid) (size : Nat) : Valid :=
  ⟨free v.manager size, free_valid v size⟩

def clone (manager : B200MemoryManager) : B200MemoryManager :=
  { config := manager.config
  , tensor_memory_used := manager.tensor_memory_used
  , prefetch_queue := manager.prefetch_queue }

theorem clone_eq (manager : B200MemoryManager) : clone manager = manager := rfl

end B200MemoryManager

inductive OpType where
  | matmul : OpType
  | add : OpType
  | activation : OpType
  | fused_gemm_bias_act : OpType
deriving Repr, Inhabited, DecidableEq

namespace OpType

def beq : OpType → OpType → Bool
  | matmul, matmul => true
  | add, add => true
  | activation, activation => true
  | fused_gemm_bias_act, fused_gemm_bias_act => true
  | _, _ => false

instance : BEq OpType := ⟨beq⟩

theorem beq_refl (x : OpType) : beq x x = true :=
  match x with
  | matmul => rfl
  | add => rfl
  | activation => rfl
  | fused_gemm_bias_act => rfl

theorem beq_symm (x y : OpType) : beq x y = beq y x :=
  match x, y with
  | matmul, matmul => rfl
  | matmul, add => rfl
  | matmul, activation => rfl
  | matmul, fused_gemm_bias_act => rfl
  | add, matmul => rfl
  | add, add => rfl
  | add, activation => rfl
  | add, fused_gemm_bias_act => rfl
  | activation, matmul => rfl
  | activation, add => rfl
  | activation, activation => rfl
  | activation, fused_gemm_bias_act => rfl
  | fused_gemm_bias_act, matmul => rfl
  | fused_gemm_bias_act, add => rfl
  | fused_gemm_bias_act, activation => rfl
  | fused_gemm_bias_act, fused_gemm_bias_act => rfl

theorem matmul_ne_add : matmul ≠ add := fun h => OpType.noConfusion h
theorem matmul_ne_activation : matmul ≠ activation := fun h => OpType.noConfusion h
theorem matmul_ne_fused_gemm_bias_act : matmul ≠ fused_gemm_bias_act := fun h => OpType.noConfusion h

theorem add_ne_matmul : add ≠ matmul := fun h => OpType.noConfusion h
theorem add_ne_activation : add ≠ activation := fun h => OpType.noConfusion h
theorem add_ne_fused_gemm_bias_act : add ≠ fused_gemm_bias_act := fun h => OpType.noConfusion h

theorem activation_ne_matmul : activation ≠ matmul := fun h => OpType.noConfusion h
theorem activation_ne_add : activation ≠ add := fun h => OpType.noConfusion h
theorem activation_ne_fused_gemm_bias_act : activation ≠ fused_gemm_bias_act := fun h => OpType.noConfusion h

theorem fused_gemm_bias_act_ne_matmul : fused_gemm_bias_act ≠ matmul := fun h => OpType.noConfusion h
theorem fused_gemm_bias_act_ne_add : fused_gemm_bias_act ≠ add := fun h => OpType.noConfusion h
theorem fused_gemm_bias_act_ne_activation : fused_gemm_bias_act ≠ activation := fun h => OpType.noConfusion h

end OpType

structure FusedKernel where
  operations : List OpType
  use_fp4 : Bool
deriving Repr

namespace FusedKernel

def clone (kernel : FusedKernel) : FusedKernel :=
  { operations := kernel.operations
  , use_fp4 := kernel.use_fp4 }

theorem clone_eq (kernel : FusedKernel) : clone kernel = kernel := rfl

end FusedKernel

structure B200KernelOptimizer where
  config : B200OptimizationConfig
deriving Repr

namespace B200KernelOptimizer

def fuseOperationsList : List OpType → List OpType
  | [] =>[]
  | OpType.matmul :: OpType.add :: OpType.activation :: rest =>
      OpType.fused_gemm_bias_act :: fuseOperationsList rest
  | op :: rest => op :: fuseOperationsList rest

theorem fuseOperationsList_length_le : ∀ (ops : List OpType),
    (fuseOperationsList ops).length ≤ ops.length :=
  fun ops =>
    @List.recOn OpType
      (fun ops => (fuseOperationsList ops).length ≤ ops.length)
      ops
      (Nat.le_refl 0)
      (fun op1 ops1 ih1 =>
        match op1, ops1 with
        | OpType.matmul, OpType.add :: OpType.activation :: rest =>
          have step : fuseOperationsList (OpType.matmul :: OpType.add :: OpType.activation :: rest) =
                      OpType.fused_gemm_bias_act :: fuseOperationsList rest := rfl
          have len_step : (fuseOperationsList (OpType.matmul :: OpType.add :: OpType.activation :: rest)).length =
                          (fuseOperationsList rest).length + 1 := congrArg List.length step
          have ih_rest : (fuseOperationsList rest).length ≤ rest.length := fuseOperationsList_length_le rest
          have h_le1 : (fuseOperationsList rest).length + 1 ≤ rest.length + 1 := Nat.succ_le_succ ih_rest
          have h_le2 : rest.length + 1 ≤ rest.length + 3 := Nat.le_trans (Nat.le_succ (rest.length + 1)) (Nat.le_succ (rest.length + 2))
          have h_len_orig : (OpType.matmul :: OpType.add :: OpType.activation :: rest).length = rest.length + 3 := rfl
          Eq.subst (Eq.symm h_len_orig) (Eq.subst (Eq.symm len_step) (Nat.le_trans h_le1 h_le2))
        | op, rest =>
          have step : fuseOperationsList (op :: rest) = op :: fuseOperationsList rest :=
            match op, rest with
            | OpType.matmul,[] => rfl
            | OpType.matmul, OpType.matmul :: _ => rfl
            | OpType.matmul, OpType.activation :: _ => rfl
            | OpType.matmul, OpType.fused_gemm_bias_act :: _ => rfl
            | OpType.matmul, OpType.add ::[] => rfl
            | OpType.matmul, OpType.add :: OpType.matmul :: _ => rfl
            | OpType.matmul, OpType.add :: OpType.add :: _ => rfl
            | OpType.matmul, OpType.add :: OpType.fused_gemm_bias_act :: _ => rfl
            | OpType.add, _ => rfl
            | OpType.activation, _ => rfl
            | OpType.fused_gemm_bias_act, _ => rfl
          have len_step : (fuseOperationsList (op :: rest)).length = (fuseOperationsList rest).length + 1 := congrArg List.length step
          have ih_rest : (fuseOperationsList rest).length ≤ rest.length := fuseOperationsList_length_le rest
          have h_le : (fuseOperationsList rest).length + 1 ≤ rest.length + 1 := Nat.succ_le_succ ih_rest
          have h_len_orig : (op :: rest).length = rest.length + 1 := rfl
          Eq.subst (Eq.symm h_len_orig) (Eq.subst (Eq.symm len_step) h_le))

def fuseOperations (optimizer : B200KernelOptimizer) (operations : List OpType) : FusedKernel :=
  { operations := fuseOperationsList operations
  , use_fp4 := optimizer.config.use_fp4_tensor_cores }

def selectOptimalPrecision (optimizer : B200KernelOptimizer) (operation : OpType) (tensor_size : Nat) : Precision :=
  if optimizer.config.use_fp4_tensor_cores && OpType.beq operation OpType.matmul then
    if tensor_size > 1000000 then
      .fp4
    else if tensor_size > 100000 then
      .fp8
    else
      .fp16
  else
    if tensor_size > 100000 then
      .fp8
    else
      .fp16

def clone (optimizer : B200KernelOptimizer) : B200KernelOptimizer :=
  { config := optimizer.config }

theorem clone_eq (optimizer : B200KernelOptimizer) : clone optimizer = optimizer := rfl

end B200KernelOptimizer

structure HyperparameterSpace where
  lr_min : FxP
  lr_max : FxP
  beta1_min : FxP
  beta1_max : FxP
  beta2_min : FxP
  beta2_max : FxP
  weight_decay_min : FxP
  weight_decay_max : FxP
deriving Repr

namespace HyperparameterSpace

def default : HyperparameterSpace :=
  { lr_min := FxP.zero
  , lr_max := FxP.one
  , beta1_min := FxP.half
  , beta1_max := FxP.one
  , beta2_min := FxP.half
  , beta2_max := FxP.one
  , weight_decay_min := FxP.zero
  , weight_decay_max := FxP.half }

def clone (space : HyperparameterSpace) : HyperparameterSpace :=
  { lr_min := space.lr_min
  , lr_max := space.lr_max
  , beta1_min := space.beta1_min
  , beta1_max := space.beta1_max
  , beta2_min := space.beta2_min
  , beta2_max := space.beta2_max
  , weight_decay_min := space.weight_decay_min
  , weight_decay_max := space.weight_decay_max }

theorem clone_eq (space : HyperparameterSpace) : clone space = space := rfl

end HyperparameterSpace

structure HyperparamConfig where
  lr : FxP
  beta1 : FxP
  beta2 : FxP
  weight_decay : FxP
deriving Repr

namespace HyperparamConfig

def default : HyperparamConfig :=
  { lr := FxP.half
  , beta1 := FxP.half
  , beta2 := FxP.half
  , weight_decay := FxP.zero }

def clone (config : HyperparamConfig) : HyperparamConfig :=
  { lr := config.lr
  , beta1 := config.beta1
  , beta2 := config.beta2
  , weight_decay := config.weight_decay }

theorem clone_eq (config : HyperparamConfig) : clone config = config := rfl

end HyperparamConfig

structure Observation where
  params : HyperparamConfig
  score : FxP
deriving Repr

namespace Observation

def clone (obs : Observation) : Observation :=
  { params := obs.params
  , score := obs.score }

theorem clone_eq (obs : Observation) : clone obs = obs := rfl

end Observation

structure Prediction where
  mean : FxP
  variance : FxP
deriving Repr

namespace Prediction

def clone (pred : Prediction) : Prediction :=
  { mean := pred.mean
  , variance := pred.variance }

theorem clone_eq (pred : Prediction) : clone pred = pred := rfl

end Prediction

structure GaussianProcess where
  observations : List Observation
  kernel_variance : FxP
  length_scale : FxP
  noise_variance : FxP
deriving Repr

namespace GaussianProcess

def init (obs : List Observation) : GaussianProcess :=
  { observations := obs
  , kernel_variance := FxP.one
  , length_scale := FxP.half
  , noise_variance := FxP.zero }

def clone (gp : GaussianProcess) : GaussianProcess :=
  { observations := gp.observations
  , kernel_variance := gp.kernel_variance
  , length_scale := gp.length_scale
  , noise_variance := gp.noise_variance }

theorem clone_eq (gp : GaussianProcess) : clone gp = gp := rfl

end GaussianProcess

structure BayesianOptimizer where
  space : HyperparameterSpace
  observations : List Observation
  best_params : HyperparamConfig
  best_score : FxP
deriving Repr

namespace BayesianOptimizer

def init (space : HyperparameterSpace) : BayesianOptimizer :=
  { space := space
  , observations :=[]
  , best_params := HyperparamConfig.default
  , best_score := FxP.fromNat 1000000 }

def observe (opt : BayesianOptimizer) (params : HyperparamConfig) (score : FxP) : BayesianOptimizer :=
  if score.raw < opt.best_score.raw then
    { opt with observations := { params := params, score := score } :: opt.observations
             , best_params := params
             , best_score := score }
  else
    { opt with observations := { params := params, score := score } :: opt.observations }

def clone (opt : BayesianOptimizer) : BayesianOptimizer :=
  { space := opt.space
  , observations := opt.observations
  , best_params := opt.best_params
  , best_score := opt.best_score }

theorem clone_eq (opt : BayesianOptimizer) : clone opt = opt := rfl

end BayesianOptimizer

structure GPUMetrics where
  utilization_percent : FxP
  memory_used_gb : FxP
  tensor_core_util : FxP
  nvlink_bandwidth_util : FxP
deriving Repr

namespace GPUMetrics

def default : GPUMetrics :=
  { utilization_percent := FxP.zero
  , memory_used_gb := FxP.zero
  , tensor_core_util := FxP.zero
  , nvlink_bandwidth_util := FxP.zero }

def clone (metrics : GPUMetrics) : GPUMetrics :=
  { utilization_percent := metrics.utilization_percent
  , memory_used_gb := metrics.memory_used_gb
  , tensor_core_util := metrics.tensor_core_util
  , nvlink_bandwidth_util := metrics.nvlink_bandwidth_util }

theorem clone_eq (metrics : GPUMetrics) : clone metrics = metrics := rfl

end GPUMetrics

structure MetricsStore where
  training_losses : List FxP
  validation_losses : List FxP
  learning_rates : List FxP
  gradient_norms : List FxP
  parameter_norms : List FxP
  step_times_ms : List FxP
  gpu_utilization : List FxP
  memory_usage_gb : List FxP
  tensor_core_utilization : List FxP
  nvlink_bandwidth_utilization : List FxP
deriving Repr

namespace MetricsStore

def init : MetricsStore :=
  { training_losses :=[]
  , validation_losses := []
  , learning_rates := []
  , gradient_norms :=[]
  , parameter_norms := []
  , step_times_ms := []
  , gpu_utilization :=[]
  , memory_usage_gb := []
  , tensor_core_utilization :=[]
  , nvlink_bandwidth_utilization :=[] }

def clone (store : MetricsStore) : MetricsStore :=
  { training_losses := store.training_losses
  , validation_losses := store.validation_losses
  , learning_rates := store.learning_rates
  , gradient_norms := store.gradient_norms
  , parameter_norms := store.parameter_norms
  , step_times_ms := store.step_times_ms
  , gpu_utilization := store.gpu_utilization
  , memory_usage_gb := store.memory_usage_gb
  , tensor_core_utilization := store.tensor_core_utilization
  , nvlink_bandwidth_utilization := store.nvlink_bandwidth_utilization }

theorem clone_eq (store : MetricsStore) : clone store = store := rfl

end MetricsStore

structure Report where
  average_loss : FxP
  average_step_time_ms : FxP
  throughput_steps_per_sec : FxP
  average_gpu_utilization : FxP
  average_memory_usage_gb : FxP
  average_tensor_core_utilization : FxP
  average_nvlink_utilization : FxP
  total_steps : Nat
deriving Repr

namespace Report

def default : Report :=
  { average_loss := FxP.zero
  , average_step_time_ms := FxP.zero
  , throughput_steps_per_sec := FxP.zero
  , average_gpu_utilization := FxP.zero
  , average_memory_usage_gb := FxP.zero
  , average_tensor_core_utilization := FxP.zero
  , average_nvlink_utilization := FxP.zero
  , total_steps := 0 }

def clone (report : Report) : Report :=
  { average_loss := report.average_loss
  , average_step_time_ms := report.average_step_time_ms
  , throughput_steps_per_sec := report.throughput_steps_per_sec
  , average_gpu_utilization := report.average_gpu_utilization
  , average_memory_usage_gb := report.average_memory_usage_gb
  , average_tensor_core_utilization := report.average_tensor_core_utilization
  , average_nvlink_utilization := report.average_nvlink_utilization
  , total_steps := report.total_steps }

theorem clone_eq (report : Report) : clone report = report := rfl

end Report

structure PerformanceMonitor where
  metrics : MetricsStore
  telemetry_enabled : Bool
deriving Repr

namespace PerformanceMonitor

def init (enable_telemetry : Bool) : PerformanceMonitor :=
  { metrics := MetricsStore.init
  , telemetry_enabled := enable_telemetry }

def recordStep (monitor : PerformanceMonitor) (loss lr grad_norm param_norm step_time_ms : FxP) (gpu_metrics : GPUMetrics) : PerformanceMonitor :=
  if monitor.telemetry_enabled then
    { monitor with metrics :=
      { monitor.metrics with
        training_losses := loss :: monitor.metrics.training_losses
      , learning_rates := lr :: monitor.metrics.learning_rates
      , gradient_norms := grad_norm :: monitor.metrics.gradient_norms
      , parameter_norms := param_norm :: monitor.metrics.parameter_norms
      , step_times_ms := step_time_ms :: monitor.metrics.step_times_ms
      , gpu_utilization := gpu_metrics.utilization_percent :: monitor.metrics.gpu_utilization
      , memory_usage_gb := gpu_metrics.memory_used_gb :: monitor.metrics.memory_usage_gb
      , tensor_core_utilization := gpu_metrics.tensor_core_util :: monitor.metrics.tensor_core_utilization
      , nvlink_bandwidth_utilization := gpu_metrics.nvlink_bandwidth_util :: monitor.metrics.nvlink_bandwidth_utilization } }
  else
    monitor

def clone (monitor : PerformanceMonitor) : PerformanceMonitor :=
  { metrics := monitor.metrics
  , telemetry_enabled := monitor.telemetry_enabled }

theorem clone_eq (monitor : PerformanceMonitor) : clone monitor = monitor := rfl

end PerformanceMonitor

structure SFDConfig where
  beta1 : FxP
  beta2 : FxP
  eps : FxP
  clip_threshold : FxP
  fisher_max : FxP
  warmup_steps : Nat
deriving Repr

namespace SFDConfig

def default : SFDConfig :=
  { beta1 := FxP.fromNat 1
  , beta2 := FxP.fromNat 1
  , eps := FxP.zero
  , clip_threshold := FxP.one
  , fisher_max := FxP.fromNat 1000000
  , warmup_steps := 10 }

def clone (config : SFDConfig) : SFDConfig :=
  { beta1 := config.beta1
  , beta2 := config.beta2
  , eps := config.eps
  , clip_threshold := config.clip_threshold
  , fisher_max := config.fisher_max
  , warmup_steps := config.warmup_steps }

theorem clone_eq (config : SFDConfig) : clone config = config := rfl

end SFDConfig

structure SFD where
  fisher_diag : TensorData
  momentum_buffer : TensorData
  velocity_buffer : TensorData
  config : SFDConfig
  step_count : Nat
  param_size : Nat
  initialized : Bool
deriving Repr

namespace SFD

def Invariant (sfd : SFD) : Prop :=
  sfd.fisher_diag.values.length = sfd.param_size ∧
  sfd.momentum_buffer.values.length = sfd.param_size ∧
  sfd.velocity_buffer.values.length = sfd.param_size ∧
  Shape.totalSize sfd.fisher_diag.shape = sfd.param_size ∧
  Shape.totalSize sfd.momentum_buffer.shape = sfd.param_size ∧
  Shape.totalSize sfd.velocity_buffer.shape = sfd.param_size

structure Valid where
  sfd : SFD
  valid : Invariant sfd

def init (param_size : Nat) (config : SFDConfig) : SFD :=
  let shape := Shape.vector param_size
  { fisher_diag := TensorData.ones [param_size]
  , momentum_buffer := TensorData.zeros [param_size]
  , velocity_buffer := TensorData.zeros [param_size]
  , config := config
  , step_count := 0
  , param_size := param_size
  , initialized := true }

theorem init_valid (param_size : Nat) (config : SFDConfig) :
    Invariant (init param_size config) :=
  have h_shape : Shape.totalSize (Shape.vector param_size) = param_size := Shape.vector_totalSize param_size
  have h_fisher_len : (TensorData.ones [param_size]).values.length = param_size :=
    have h_len : (TensorData.ones [param_size]).values.length = Shape.totalSize ⟨[param_size]⟩ := TensorData.ones_length [param_size]
    Eq.trans h_len h_shape
  have h_momentum_len : (TensorData.zeros [param_size]).values.length = param_size :=
    have h_len : (TensorData.zeros [param_size]).values.length = Shape.totalSize ⟨[param_size]⟩ := TensorData.zeros_length [param_size]
    Eq.trans h_len h_shape
  have h_velocity_len : (TensorData.zeros[param_size]).values.length = param_size :=
    have h_len : (TensorData.zeros[param_size]).values.length = Shape.totalSize ⟨[param_size]⟩ := TensorData.zeros_length [param_size]
    Eq.trans h_len h_shape
  And.intro h_fisher_len (And.intro h_momentum_len (And.intro h_velocity_len (And.intro h_shape (And.intro h_shape h_shape))))

def initValid (param_size : Nat) (config : SFDConfig) : Valid :=
  ⟨init param_size config, init_valid param_size config⟩

def update_momentum (momentum : List Int) (grad : List Int) (beta1 : FxP) : List Int :=
  List.zipWith (fun m g => (beta1.raw * m + (FxP.scale - beta1.raw) * g) / FxP.scale) momentum grad

theorem update_momentum_length (momentum grad : List Int) (beta1 : FxP) (h_len : momentum.length = grad.length) :
    (update_momentum momentum grad beta1).length = momentum.length :=
  have h_zip : (List.zipWith (fun m g => (beta1.raw * m + (FxP.scale - beta1.raw) * g) / FxP.scale) momentum grad).length = Nat.min momentum.length grad.length :=
    TensorData.zipWithLength (fun m g => (beta1.raw * m + (FxP.scale - beta1.raw) * g) / FxP.scale) momentum grad
  have h_min : Nat.min momentum.length grad.length = momentum.length :=
    Eq.subst (Eq.symm h_len) (Nat.min_self momentum.length)
  Eq.trans h_zip h_min

def update_velocity (velocity : List Int) (grad : List Int) (beta2 : FxP) : List Int :=
  List.zipWith (fun v g => (beta2.raw * v + (FxP.scale - beta2.raw) * (g * g / FxP.scale)) / FxP.scale) velocity grad

theorem update_velocity_length (velocity grad : List Int) (beta2 : FxP) (h_len : velocity.length = grad.length) :
    (update_velocity velocity grad beta2).length = velocity.length :=
  have h_zip : (List.zipWith (fun v g => (beta2.raw * v + (FxP.scale - beta2.raw) * (g * g / FxP.scale)) / FxP.scale) velocity grad).length = Nat.min velocity.length grad.length :=
    TensorData.zipWithLength (fun v g => (beta2.raw * v + (FxP.scale - beta2.raw) * (g * g / FxP.scale)) / FxP.scale) velocity grad
  have h_min : Nat.min velocity.length grad.length = velocity.length :=
    Eq.subst (Eq.symm h_len) (Nat.min_self velocity.length)
  Eq.trans h_zip h_min

def update_fisher (fisher : List Int) (grad : List Int) (beta2 fisher_max : FxP) : List Int :=
  List.zipWith (fun f g =>
    let updated := (beta2.raw * f + (FxP.scale - beta2.raw) * (g * g / FxP.scale)) / FxP.scale
    if updated > fisher_max.raw then fisher_max.raw else updated) fisher grad

theorem update_fisher_length (fisher grad : List Int) (beta2 fisher_max : FxP) (h_len : fisher.length = grad.length) :
    (update_fisher fisher grad beta2 fisher_max).length = fisher.length :=
  have h_zip : (List.zipWith (fun f g => let updated := (beta2.raw * f + (FxP.scale - beta2.raw) * (g * g / FxP.scale)) / FxP.scale; if updated > fisher_max.raw then fisher_max.raw else updated) fisher grad).length = Nat.min fisher.length grad.length :=
    TensorData.zipWithLength (fun f g => let updated := (beta2.raw * f + (FxP.scale - beta2.raw) * (g * g / FxP.scale)) / FxP.scale; if updated > fisher_max.raw then fisher_max.raw else updated) fisher grad
  have h_min : Nat.min fisher.length grad.length = fisher.length :=
    Eq.subst (Eq.symm h_len) (Nat.min_self fisher.length)
  Eq.trans h_zip h_min

def update (sfd : SFD) (grad : TensorData) : SFD :=
  if sfd.param_size == grad.values.length then
    { fisher_diag := { sfd.fisher_diag with values := update_fisher sfd.fisher_diag.values grad.values sfd.config.beta2 sfd.config.fisher_max }
    , momentum_buffer := { sfd.momentum_buffer with values := update_momentum sfd.momentum_buffer.values grad.values sfd.config.beta1 }
    , velocity_buffer := { sfd.velocity_buffer with values := update_velocity sfd.velocity_buffer.values grad.values sfd.config.beta2 }
    , config := sfd.config
    , step_count := Nat.succ sfd.step_count
    , param_size := sfd.param_size
    , initialized := sfd.initialized }
  else
    sfd

theorem update_valid (v : Valid) (grad : TensorData) :
    Invariant (update v.sfd grad) :=
  if h_eq : v.sfd.param_size == grad.values.length then
    have h_eq_prop : v.sfd.param_size = grad.values.length := eq_of_beq h_eq
    have h_update : update v.sfd grad = { fisher_diag := { v.sfd.fisher_diag with values := update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max }, momentum_buffer := { v.sfd.momentum_buffer with values := update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 }, velocity_buffer := { v.sfd.velocity_buffer with values := update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 }, config := v.sfd.config, step_count := Nat.succ v.sfd.step_count, param_size := v.sfd.param_size, initialized := v.sfd.initialized } :=
      have h_cond : (v.sfd.param_size == grad.values.length) = true := h_eq
      dite_cond_eq_true (v.sfd.param_size == grad.values.length) h_cond (fun _ => { fisher_diag := { v.sfd.fisher_diag with values := update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max }, momentum_buffer := { v.sfd.momentum_buffer with values := update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 }, velocity_buffer := { v.sfd.velocity_buffer with values := update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 }, config := v.sfd.config, step_count := Nat.succ v.sfd.step_count, param_size := v.sfd.param_size, initialized := v.sfd.initialized }) (fun _ => v.sfd)
    have h_fisher_len_orig : v.sfd.fisher_diag.values.length = v.sfd.param_size := v.valid.left
    have h_momentum_len_orig : v.sfd.momentum_buffer.values.length = v.sfd.param_size := v.valid.right.left
    have h_velocity_len_orig : v.sfd.velocity_buffer.values.length = v.sfd.param_size := v.valid.right.right.left
    have h_fisher_grad_len : v.sfd.fisher_diag.values.length = grad.values.length := Eq.trans h_fisher_len_orig h_eq_prop
    have h_momentum_grad_len : v.sfd.momentum_buffer.values.length = grad.values.length := Eq.trans h_momentum_len_orig h_eq_prop
    have h_velocity_grad_len : v.sfd.velocity_buffer.values.length = grad.values.length := Eq.trans h_velocity_len_orig h_eq_prop
    have h_fisher_len_new : (update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max).length = v.sfd.param_size := Eq.trans (update_fisher_length v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max h_fisher_grad_len) h_fisher_len_orig
    have h_momentum_len_new : (update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1).length = v.sfd.param_size := Eq.trans (update_momentum_length v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 h_momentum_grad_len) h_momentum_len_orig
    have h_velocity_len_new : (update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2).length = v.sfd.param_size := Eq.trans (update_velocity_length v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 h_velocity_grad_len) h_velocity_len_orig
    have h_fisher_shape : Shape.totalSize v.sfd.fisher_diag.shape = v.sfd.param_size := v.valid.right.right.right.left
    have h_momentum_shape : Shape.totalSize v.sfd.momentum_buffer.shape = v.sfd.param_size := v.valid.right.right.right.right.left
    have h_velocity_shape : Shape.totalSize v.sfd.velocity_buffer.shape = v.sfd.param_size := v.valid.right.right.right.right.right
    have h_inv : Invariant { fisher_diag := { v.sfd.fisher_diag with values := update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max }, momentum_buffer := { v.sfd.momentum_buffer with values := update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 }, velocity_buffer := { v.sfd.velocity_buffer with values := update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 }, config := v.sfd.config, step_count := Nat.succ v.sfd.step_count, param_size := v.sfd.param_size, initialized := v.sfd.initialized } := And.intro h_fisher_len_new (And.intro h_momentum_len_new (And.intro h_velocity_len_new (And.intro h_fisher_shape (And.intro h_momentum_shape h_velocity_shape))))
    Eq.subst (Eq.symm h_update) h_inv
  else
    have h_cond : (v.sfd.param_size == grad.values.length) = false := h_eq
    have h_update : update v.sfd grad = v.sfd :=
      dite_cond_eq_false (v.sfd.param_size == grad.values.length) h_cond (fun _ => { fisher_diag := { v.sfd.fisher_diag with values := update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max }, momentum_buffer := { v.sfd.momentum_buffer with values := update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 }, velocity_buffer := { v.sfd.velocity_buffer with values := update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 }, config := v.sfd.config, step_count := Nat.succ v.sfd.step_count, param_size := v.sfd.param_size, initialized := v.sfd.initialized }) (fun _ => v.sfd)
    Eq.subst (Eq.symm h_update) v.valid

def updateValid (v : Valid) (grad : TensorData) : Valid :=
  ⟨update v.sfd grad, update_valid v grad⟩

theorem step_count_mono (v : Valid) (grad : TensorData) (h_eq : v.sfd.param_size == grad.values.length) :
    v.sfd.step_count < (update v.sfd grad).step_count :=
  have h_cond : (v.sfd.param_size == grad.values.length) = true := h_eq
  have h_update : update v.sfd grad = { fisher_diag := { v.sfd.fisher_diag with values := update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max }, momentum_buffer := { v.sfd.momentum_buffer with values := update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 }, velocity_buffer := { v.sfd.velocity_buffer with values := update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 }, config := v.sfd.config, step_count := Nat.succ v.sfd.step_count, param_size := v.sfd.param_size, initialized := v.sfd.initialized } :=
    dite_cond_eq_true (v.sfd.param_size == grad.values.length) h_cond (fun _ => { fisher_diag := { v.sfd.fisher_diag with values := update_fisher v.sfd.fisher_diag.values grad.values v.sfd.config.beta2 v.sfd.config.fisher_max }, momentum_buffer := { v.sfd.momentum_buffer with values := update_momentum v.sfd.momentum_buffer.values grad.values v.sfd.config.beta1 }, velocity_buffer := { v.sfd.velocity_buffer with values := update_velocity v.sfd.velocity_buffer.values grad.values v.sfd.config.beta2 }, config := v.sfd.config, step_count := Nat.succ v.sfd.step_count, param_size := v.sfd.param_size, initialized := v.sfd.initialized }) (fun _ => v.sfd)
  have h_step : (update v.sfd grad).step_count = Nat.succ v.sfd.step_count := congrArg step_count h_update
  have h_lt : v.sfd.step_count < Nat.succ v.sfd.step_count := Nat.lt_succ_self v.sfd.step_count
  Eq.subst (Eq.symm h_step) h_lt

def resetFisher (sfd : SFD) : SFD :=
  { sfd with fisher_diag := { sfd.fisher_diag with values := List.replicate sfd.param_size FxP.scale } }

theorem resetFisher_valid (v : Valid) :
    Invariant (resetFisher v.sfd) :=
  have h_reset : resetFisher v.sfd = { v.sfd with fisher_diag := { v.sfd.fisher_diag with values := List.replicate v.sfd.param_size FxP.scale } } := rfl
  have h_fisher_len : (List.replicate v.sfd.param_size FxP.scale).length = v.sfd.param_size := List.length_replicate v.sfd.param_size FxP.scale
  have h_momentum_len : v.sfd.momentum_buffer.values.length = v.sfd.param_size := v.valid.right.left
  have h_velocity_len : v.sfd.velocity_buffer.values.length = v.sfd.param_size := v.valid.right.right.left
  have h_fisher_shape : Shape.totalSize v.sfd.fisher_diag.shape = v.sfd.param_size := v.valid.right.right.right.left
  have h_momentum_shape : Shape.totalSize v.sfd.momentum_buffer.shape = v.sfd.param_size := v.valid.right.right.right.right.left
  have h_velocity_shape : Shape.totalSize v.sfd.velocity_buffer.shape = v.sfd.param_size := v.valid.right.right.right.right.right
  have h_inv : Invariant { v.sfd with fisher_diag := { v.sfd.fisher_diag with values := List.replicate v.sfd.param_size FxP.scale } } := And.intro h_fisher_len (And.intro h_momentum_len (And.intro h_velocity_len (And.intro h_fisher_shape (And.intro h_momentum_shape h_velocity_shape))))
  Eq.subst (Eq.symm h_reset) h_inv

def resetFisherValid (v : Valid) : Valid :=
  ⟨resetFisher v.sfd, resetFisher_valid v⟩

def clone (sfd : SFD) : SFD :=
  { fisher_diag := sfd.fisher_diag
  , momentum_buffer := sfd.momentum_buffer
  , velocity_buffer := sfd.velocity_buffer
  , config := sfd.config
  , step_count := sfd.step_count
  , param_size := sfd.param_size
  , initialized := sfd.initialized }

theorem clone_eq (sfd : SFD) : clone sfd = sfd := rfl

end SFD

structure SophiaSOAPConfig where
  rho : FxP
  gamma : FxP
  hessian_update_freq : Nat
  use_gauss_newton : Bool
deriving Repr

namespace SophiaSOAPConfig

def default : SophiaSOAPConfig :=
  { rho := FxP.fromNat 4 / 100
  , gamma := FxP.fromNat 1 / 100
  , hessian_update_freq := 10
  , use_gauss_newton := true }

def clone (config : SophiaSOAPConfig) : SophiaSOAPConfig :=
  { rho := config.rho
  , gamma := config.gamma
  , hessian_update_freq := config.hessian_update_freq
  , use_gauss_newton := config.use_gauss_newton }

theorem clone_eq (config : SophiaSOAPConfig) : clone config = config := rfl

end SophiaSOAPConfig

structure SophiaSOAPOptimizer where
  sfd : SFD
  hessian_diag : TensorData
  hutchinson_vector : TensorData
  config : SophiaSOAPConfig
  param_size : Nat
deriving Repr

namespace SophiaSOAPOptimizer

def Invariant (opt : SophiaSOAPOptimizer) : Prop :=
  SFD.Invariant opt.sfd ∧
  opt.hessian_diag.values.length = opt.param_size ∧
  opt.hutchinson_vector.values.length = opt.param_size ∧
  Shape.totalSize opt.hessian_diag.shape = opt.param_size ∧
  Shape.totalSize opt.hutchinson_vector.shape = opt.param_size ∧
  opt.sfd.param_size = opt.param_size

structure Valid where
  opt : SophiaSOAPOptimizer
  valid : Invariant opt

def init (param_size : Nat) (config : SophiaSOAPConfig) (sfd_config : SFDConfig) : SophiaSOAPOptimizer :=
  let shape := Shape.vector param_size
  { sfd := SFD.init param_size sfd_config
  , hessian_diag := TensorData.ones [param_size]
  , hutchinson_vector := TensorData.ones[param_size]
  , config := config
  , param_size := param_size }

theorem init_valid (param_size : Nat) (config : SophiaSOAPConfig) (sfd_config : SFDConfig) :
    Invariant (init param_size config sfd_config) :=
  have h_sfd_inv : SFD.Invariant (SFD.init param_size sfd_config) := SFD.init_valid param_size sfd_config
  have h_shape : Shape.totalSize (Shape.vector param_size) = param_size := Shape.vector_totalSize param_size
  have h_hessian_len : (TensorData.ones [param_size]).values.length = param_size :=
    have h_len : (TensorData.ones [param_size]).values.length = Shape.totalSize ⟨[param_size]⟩ := TensorData.ones_length [param_size]
    Eq.trans h_len h_shape
  have h_hutch_len : (TensorData.ones[param_size]).values.length = param_size :=
    have h_len : (TensorData.ones [param_size]).values.length = Shape.totalSize ⟨[param_size]⟩ := TensorData.ones_length [param_size]
    Eq.trans h_len h_shape
  have h_sfd_param : (SFD.init param_size sfd_config).param_size = param_size := rfl
  And.intro h_sfd_inv (And.intro h_hessian_len (And.intro h_hutch_len (And.intro h_shape (And.intro h_shape h_sfd_param))))

def initValid (param_size : Nat) (config : SophiaSOAPConfig) (sfd_config : SFDConfig) : Valid :=
  ⟨init param_size config sfd_config, init_valid param_size config sfd_config⟩

def clone (opt : SophiaSOAPOptimizer) : SophiaSOAPOptimizer :=
  { sfd := opt.sfd
  , hessian_diag := opt.hessian_diag
  , hutchinson_vector := opt.hutchinson_vector
  , config := opt.config
  , param_size := opt.param_size }

theorem clone_eq (opt : SophiaSOAPOptimizer) : clone opt = opt := rfl

end SophiaSOAPOptimizer

namespace TraceVerification

def SFDTrace := List SFD

def validTrace : SFDTrace → Prop
  |[] => True
  | s :: ss => SFD.Invariant s ∧ validTrace ss

theorem validTrace_nil : validTrace[] = True := rfl

theorem validTrace_cons (s : SFD) (ss : SFDTrace) :
    validTrace (s :: ss) = (SFD.Invariant s ∧ validTrace ss) := rfl

def traceMonotonic : SFDTrace → Prop
  |[] => True
  | [s] => True
  | s1 :: s2 :: ss => s1.step_count ≤ s2.step_count ∧ traceMonotonic (s2 :: ss)

theorem traceMonotonic_nil : traceMonotonic[] = True := rfl

theorem traceMonotonic_singleton (s : SFD) : traceMonotonic [s] = True := rfl

theorem traceMonotonic_cons2 (s1 s2 : SFD) (ss : SFDTrace) :
    traceMonotonic (s1 :: s2 :: ss) = (s1.step_count ≤ s2.step_count ∧ traceMonotonic (s2 :: ss)) := rfl

def generateTrace (start : SFD.Valid) (grads : List TensorData) : SFDTrace :=
  @List.recOn TensorData (fun _ => SFDTrace) grads
    [start.sfd]
    (fun g gs ih =>
      match ih with
      | [] =>
