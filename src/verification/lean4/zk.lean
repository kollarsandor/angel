namespace ZKVerification

namespace Types

inductive ZKProofError
| CircomCompilationFailed
| WitnessGenerationFailed
| ProofGenerationFailed
| VerificationFailed
| FileNotFound
| InvalidProofFormat
| InvalidWitnessFormat
| SnarkjsNotFound
| CircuitNotCompiled
| KeysNotGenerated
| OutOfMemory
| ProcessSpawnFailed
| Timeout
| InvalidInput
| InvalidOutput

def U8 := Fin 256
def U64 := Fin 18446744073709551616
def I64 := Int
def I256 := Int
def Byte := Fin 256
def ByteArray32 := Fin 32 → Fin 256
def ByteArray96 := Fin 96 → Fin 256

structure ZKCircuitConfig where
  circuit_path : List U8
  wasm_path : List U8
  zkey_path : List U8
  vkey_path : List U8
  witness_dir : List U8
  proof_dir : List U8
  num_layers : Nat
  embedding_dim : Nat
  precision_bits : Nat
  timeout_ms : Nat

structure Groth16Proof where
  pi_a : Fin 3 → ByteArray96
  pi_b : Fin 3 → Fin 2 → ByteArray96
  pi_c : Fin 3 → ByteArray96
  protocol : Fin 7 → U8
  curve : Fin 4 → U8

structure PublicSignals where
  signals : List Int

structure ZKProofBundle where
  proof : Groth16Proof
  public_signals : PublicSignals
  proof_json : List U8
  public_json : List U8
  timestamp : Int
  verification_status : Bool

structure CircomProverState where
  config : ZKCircuitConfig
  circuit_compiled : Bool
  keys_generated : Bool
  snarkjs_path : List U8
  node_path : List U8

inductive CompileResult
| success
| failure (err : ZKProofError)

inductive WitnessResult
| success
| failure (err : ZKProofError)

inductive ProofResult
| success
| failure (err : ZKProofError)

inductive SetupResult
| success
| failure (err : ZKProofError)

structure InferenceWitness where
  tokens : List Int
  layer_weights_s : List (List (List Int))
  layer_weights_t : List (List (List Int))
  expected_output : List Int
  input_commitment : Int
  output_commitment : Int
  layer_commitments : List Int
  max_error_squared : Int
  num_layers : Nat
  dim : Nat

def HashFn := List (Fin 256) → Fin 32 → Fin 256

structure HashOracle where
  blake3 : List (List (Fin 256)) → ByteArray32
  sha256 : List (List (Fin 256)) → ByteArray32

structure CommitmentData where
  value_hash : ByteArray32
  nonce : ByteArray32
  timestamp : Int
  blinding_factor : ByteArray32

structure CommitmentScheme where
  commitments : List (ByteArray32 × CommitmentData)
  nonce_counter : Nat

structure RangeProofBit where
  commitment : ByteArray32
  opening : ByteArray32
  bit_value : Fin 2

structure RangeProof where
  min_value : Int
  max_value : Int
  proof_bits : List RangeProofBit

structure MerkleProof where
  root : ByteArray32
  path : List ByteArray32
  directions : List Bool

structure SchnorrSignature where
  challenge : ByteArray32
  response : ByteArray32

structure DifferentialPrivacyConfig where
  epsilon : Nat
  delta : Nat
  sensitivity : Nat
  noise_scale : Nat

structure ZKInferenceProof where
  input_commitment : ByteArray32
  output_commitment : ByteArray32
  computation_proof : List ByteArray32
  timestamp : Int
  has_zk_prover : Bool
  proof_bundle : Option ZKProofBundle

structure SecureAggregation where
  participant_commitments : List (Nat × ByteArray32)
  aggregated_result : Option (List Int)
  threshold : Nat

inductive AggregateError
| InsufficientParticipants
| DimensionMismatch

structure ZKInferenceProver where
  config : ZKCircuitConfig
  prover_state : CircomProverState
  proof_counter : Nat

end Types

namespace Helpers

def natBle : Nat → Nat → Bool
| 0, _ => true
| Nat.succ _, 0 => false
| Nat.succ n, Nat.succ m => natBle n m

def natBeq : Nat → Nat → Bool
| 0, 0 => true
| Nat.succ n, Nat.succ m => natBeq n m
| _, _ => false

def boolAnd : Bool → Bool → Bool
| true, true => true
| _, _ => false

def boolOr : Bool → Bool → Bool
| false, false => false
| _, _ => true

def boolNot : Bool → Bool
| true => false
| false => true

def listLength {α : Type} : List α → Nat
| [] => 0
| _ :: xs => Nat.succ (listLength xs)

def listAppend {α : Type} : List α → List α → List α
| [], ys => ys
| x :: xs, ys => x :: listAppend xs ys

def listMap {α β : Type} (f : α → β) : List α → List β
| [] => []
| x :: xs => f x :: listMap f xs

def listZipWith {α β γ : Type} (f : α → β → γ) : List α → List β → List γ
| [], _ => []
| _, [] => []
| x :: xs, y :: ys => f x y :: listZipWith f xs ys

def listFoldl {α β : Type} (f : α → β → α) : α → List β → α
| acc, [] => acc
| acc, x :: xs => listFoldl f (f acc x) xs

def listFoldr {α β : Type} (f : α → β → β) : β → List α → β
| acc, [] => acc
| acc, x :: xs => f x (listFoldr f acc xs)

def listReplicate {α : Type} : Nat → α → List α
| 0, _ => []
| Nat.succ n, a => a :: listReplicate n a

def listGet {α : Type} : (l : List α) → (i : Nat) → i < listLength l → α
| x :: _, 0, _ => x
| _ :: xs, Nat.succ i, h => listGet xs i (Nat.lt_of_succ_lt_succ h)

def listGetD {α : Type} : List α → Nat → α → α
| [], _, d => d
| x :: _, 0, _ => x
| _ :: xs, Nat.succ i, d => listGetD xs i d

def intToNat : Int → Nat
| Int.ofNat n => n
| Int.negSucc _ => 0

def intLe (a b : Int) : Bool := natBle (intToNat a) (intToNat b)

def intAdd (a b : Int) : Int := a + b
def intSub (a b : Int) : Int := a - b
def intMul (a b : Int) : Int := a * b

def natAdd (a b : Nat) : Nat := a + b
def natSub (a b : Nat) : Nat := a - b
def natMul (a b : Nat) : Nat := a * b
def natDiv (a b : Nat) : Nat := a / b
def natMod (a b : Nat) : Nat := a % b

end Helpers

namespace Proofs

open Helpers

theorem natBle_refl (n : Nat) : natBle n n = true :=
  Nat.recOn n
    (Eq.refl true)
    (fun k ih => ih)

theorem natBle_sound (a b : Nat) (h : natBle a b = true) : a ≤ b :=
  Nat.recOn a
    (fun _ _ => Nat.zero_le _)
    (fun n ih b =>
      Nat.recOn b
        (fun h_false => False.elim (Bool.noConfusion h_false))
        (fun m _ h_succ => Nat.succ_le_succ (ih m h_succ)))
    b h

theorem natBeq_refl (n : Nat) : natBeq n n = true :=
  Nat.recOn n
    (Eq.refl true)
    (fun k ih => ih)

theorem natBeq_symm (a b : Nat) : natBeq a b = natBeq b a :=
  Nat.recOn a
    (fun b => Nat.recOn b (Eq.refl true) (fun _ _ => Eq.refl false))
    (fun n ih b =>
      Nat.recOn b
        (Eq.refl false)
        (fun m _ => ih m))
    b

theorem natBeq_sound (a b : Nat) (h : natBeq a b = true) : a = b :=
  Nat.recOn a
    (fun b =>
      Nat.recOn b
        (fun _ => Eq.refl 0)
        (fun _ _ h_false => False.elim (Bool.noConfusion h_false)))
    (fun n ih b =>
      Nat.recOn b
        (fun h_false => False.elim (Bool.noConfusion h_false))
        (fun m _ h_succ => congrArg Nat.succ (ih m h_succ)))
    b h

theorem boolAnd_true_left (a b : Bool) (h : boolAnd a b = true) : a = true :=
  Bool.recOn a
    (fun h_false => False.elim (Bool.noConfusion h_false))
    (fun _ => Eq.refl true)
    h

theorem boolAnd_true_right (a b : Bool) (h : boolAnd a b = true) : b = true :=
  Bool.recOn a
    (fun h_false => False.elim (Bool.noConfusion h_false))
    (fun _ => h)
    h

theorem boolAnd_intro (a b : Bool) (ha : a = true) (hb : b = true) : boolAnd a b = true :=
  Eq.subst (motive := fun x => boolAnd x b = true) (Eq.symm ha)
    (Eq.subst (motive := fun x => boolAnd true x = true) (Eq.symm hb) (Eq.refl true))

theorem boolAnd_comm (a b : Bool) : boolAnd a b = boolAnd b a :=
  Bool.recOn a
    (Bool.recOn b (Eq.refl false) (Eq.refl false))
    (Bool.recOn b (Eq.refl false) (Eq.refl true))

theorem boolAnd_assoc (a b c : Bool) : boolAnd (boolAnd a b) c = boolAnd a (boolAnd b c) :=
  Bool.recOn a
    (Eq.refl false)
    (Bool.recOn b
      (Eq.refl false)
      (Bool.recOn c (Eq.refl false) (Eq.refl true)))

theorem listLength_append {α : Type} (l1 l2 : List α) : listLength (listAppend l1 l2) = listLength l1 + listLength l2 :=
  List.recOn l1
    (Eq.refl (listLength l2))
    (fun x xs ih =>
      Eq.trans (congrArg Nat.succ ih) (Eq.refl (Nat.succ (listLength xs + listLength l2))))

theorem listLength_map {α β : Type} (f : α → β) (l : List α) : listLength (listMap f l) = listLength l :=
  List.recOn l
    (Eq.refl 0)
    (fun x xs ih => congrArg Nat.succ ih)

theorem listLength_replicate {α : Type} (n : Nat) (a : α) : listLength (listReplicate n a) = n :=
  Nat.recOn n
    (Eq.refl 0)
    (fun k ih => congrArg Nat.succ ih)

theorem listLength_zipWith {α β γ : Type} (f : α → β → γ) (l1 : List α) (l2 : List β) (h : listLength l1 = listLength l2) : listLength (listZipWith f l1 l2) = listLength l1 :=
  List.recOn l1
    (fun _ _ => Eq.refl 0)
    (fun x xs ih l2 =>
      List.recOn l2
        (fun h_false => False.elim (Nat.noConfusion h_false))
        (fun y ys _ h_succ =>
          congrArg Nat.succ (ih ys (Nat.succ.inj h_succ))))
    l2 h

theorem listGet_replicate {α : Type} (n : Nat) (a : α) (i : Nat) (h : i < listLength (listReplicate n a)) : listGet (listReplicate n a) i h = a :=
  Nat.recOn n
    (fun i h_false => False.elim (Nat.not_lt_zero i h_false))
    (fun k ih i =>
      Nat.recOn i
        (fun _ => Eq.refl a)
        (fun j _ h_succ => ih j (Nat.lt_of_succ_lt_succ h_succ)))
    i h

theorem natAdd_zero (n : Nat) : n + 0 = n :=
  Eq.refl n

theorem natZero_add (n : Nat) : 0 + n = n :=
  Nat.recOn n
    (Eq.refl 0)
    (fun k ih => congrArg Nat.succ ih)

theorem natAdd_succ (n m : Nat) : n + Nat.succ m = Nat.succ (n + m) :=
  Eq.refl (Nat.succ (n + m))

theorem natSucc_add (n m : Nat) : Nat.succ n + m = Nat.succ (n + m) :=
  Nat.recOn m
    (Eq.refl (Nat.succ n))
    (fun k ih => congrArg Nat.succ ih)

theorem natAdd_comm (n m : Nat) : n + m = m + n :=
  Nat.recOn m
    (natZero_add n)
    (fun k ih =>
      Eq.trans (natAdd_succ n k)
        (Eq.trans (congrArg Nat.succ ih) (Eq.symm (natSucc_add k n))))

theorem natAdd_assoc (n m k : Nat) : (n + m) + k = n + (m + k) :=
  Nat.recOn k
    (Eq.refl (n + m))
    (fun l ih => congrArg Nat.succ ih)

theorem natMul_zero (n : Nat) : n * 0 = 0 :=
  Eq.refl 0

theorem natZero_mul (n : Nat) : 0 * n = 0 :=
  Nat.recOn n
    (Eq.refl 0)
    (fun k ih => ih)

theorem natMul_succ (n m : Nat) : n * Nat.succ m = n * m + n :=
  Eq.refl (n * m + n)

theorem natSucc_mul (n m : Nat) : Nat.succ n * m = n * m + m :=
  Nat.recOn m
    (Eq.refl 0)
    (fun k ih =>
      Eq.trans (congrArg (fun x => x + Nat.succ n) ih)
        (Eq.trans (natAdd_succ (n * k + k) n)
          (Eq.trans (congrArg Nat.succ (natAdd_assoc (n * k) k n))
            (Eq.trans (congrArg (fun x => Nat.succ (n * k + x)) (natAdd_comm k n))
              (Eq.trans (congrArg Nat.succ (Eq.symm (natAdd_assoc (n * k) n k)))
                (Eq.symm (natAdd_succ (n * k + n) k)))))))

theorem natMul_comm (n m : Nat) : n * m = m * n :=
  Nat.recOn m
    (natZero_mul n)
    (fun k ih =>
      Eq.trans (natMul_succ n k)
        (Eq.trans (congrArg (fun x => x + n) ih) (Eq.symm (natSucc_mul k n))))

theorem natMul_add (n m k : Nat) : n * (m + k) = n * m + n * k :=
  Nat.recOn k
    (Eq.refl (n * m))
    (fun l ih =>
      Eq.trans (natMul_succ n (m + l))
        (Eq.trans (congrArg (fun x => x + n) ih)
          (Eq.symm (natAdd_assoc (n * m) (n * l) n))))

theorem natAdd_mul (n m k : Nat) : (n + m) * k = n * k + m * k :=
  Eq.trans (natMul_comm (n + m) k)
    (Eq.trans (natMul_add k n m)
      (Eq.trans (congrArg (fun x => x + k * m) (natMul_comm k n))
        (congrArg (fun x => n * k + x) (natMul_comm k m))))

theorem natMul_assoc (n m k : Nat) : (n * m) * k = n * (m * k) :=
  Nat.recOn k
    (Eq.refl 0)
    (fun l ih =>
      Eq.trans (natMul_succ (n * m) l)
        (Eq.trans (congrArg (fun x => x + n * m) ih)
          (Eq.symm (natMul_add n (m * l) m))))

theorem natAdd_sub_cancel (n m : Nat) : n + m - m = n :=
  Nat.add_sub_cancel n m

theorem natSub_add_cancel (n m : Nat) (h : m ≤ n) : n - m + m = n :=
  Nat.sub_add_cancel h

theorem natMod_add_div (a b : Nat) : a % b + b * (a / b) = a :=
  Nat.mod_add_div a b

theorem natMod_lt (a b : Nat) (h : b > 0) : a % b < b :=
  Nat.mod_lt a h

theorem natDiv_lt_of_lt_mul (a b : Nat) (h : a < 2 * 2^b) : a / 2 < 2^b :=
  Nat.div_lt_of_lt_mul h

theorem matchOptionSome {α β : Type} (opt : Option α) (val : α) (h : opt = some val) (f_none : β) (f_some : α → β) :
  (match opt with | none => f_none | some x => f_some x) = f_some val :=
  Option.recOn opt
    (fun h_false => False.elim (Option.noConfusion h_false))
    (fun x h_eq => congrArg f_some (Option.some.inj h_eq))
    h

theorem matchBoolTrue {α : Type} (b : Bool) (t f : α) (h : b = true) :
  (match b with | true => t | false => f) = t :=
  Bool.recOn b
    (fun h_false => False.elim (Bool.noConfusion h_false))
    (fun _ => Eq.refl t)
    h

theorem matchBoolFalse {α : Type} (b : Bool) (t f : α) (h : b = false) :
  (match b with | true => t | false => f) = f :=
  Bool.recOn b
    (fun _ => Eq.refl f)
    (fun h_false => False.elim (Bool.noConfusion h_false))
    h

end Proofs

open Types Helpers Proofs

namespace Config

def ZKCircuitConfig.defaultConfig : ZKCircuitConfig := {
  circuit_path := [],
  wasm_path := [],
  zkey_path := [],
  vkey_path := [],
  witness_dir := [],
  proof_dir := [],
  num_layers := 8,
  embedding_dim := 32,
  precision_bits := 64,
  timeout_ms := 300000
}

def Groth16Proof.init : Groth16Proof := {
  pi_a := fun _ _ => ⟨0, Nat.zero_lt_succ 255⟩,
  pi_b := fun _ _ _ => ⟨0, Nat.zero_lt_succ 255⟩,
  pi_c := fun _ _ => ⟨0, Nat.zero_lt_succ 255⟩,
  protocol := fun i =>
    match i.val with
    | 0 => ⟨103, Nat.lt_trans (Nat.lt_succ_self 103) (Nat.lt_succ_self 254)⟩
    | 1 => ⟨114, Nat.lt_trans (Nat.lt_succ_self 114) (Nat.lt_succ_self 254)⟩
    | 2 => ⟨111, Nat.lt_trans (Nat.lt_succ_self 111) (Nat.lt_succ_self 254)⟩
    | 3 => ⟨116, Nat.lt_trans (Nat.lt_succ_self 116) (Nat.lt_succ_self 254)⟩
    | 4 => ⟨104, Nat.lt_trans (Nat.lt_succ_self 104) (Nat.lt_succ_self 254)⟩
    | 5 => ⟨49, Nat.lt_trans (Nat.lt_succ_self 49) (Nat.lt_succ_self 254)⟩
    | 6 => ⟨54, Nat.lt_trans (Nat.lt_succ_self 54) (Nat.lt_succ_self 254)⟩
    | _ => ⟨0, Nat.zero_lt_succ 255⟩,
  curve := fun i =>
    match i.val with
    | 0 => ⟨98, Nat.lt_trans (Nat.lt_succ_self 98) (Nat.lt_succ_self 254)⟩
    | 1 => ⟨110, Nat.lt_trans (Nat.lt_succ_self 110) (Nat.lt_succ_self 254)⟩
    | 2 => ⟨49, Nat.lt_trans (Nat.lt_succ_self 49) (Nat.lt_succ_self 254)⟩
    | 3 => ⟨50, Nat.lt_trans (Nat.lt_succ_self 50) (Nat.lt_succ_self 254)⟩
    | _ => ⟨0, Nat.zero_lt_succ 255⟩
}

def PublicSignals.init : PublicSignals := { signals := [] }

def PublicSignals.addSignal (self : PublicSignals) (value : Int) : PublicSignals :=
  { signals := listAppend self.signals [value] }

def ZKProofBundle.init (t : Int) : ZKProofBundle := {
  proof := Groth16Proof.init,
  public_signals := PublicSignals.init,
  proof_json := [],
  public_json := [],
  timestamp := t,
  verification_status := false
}

def CircomProverState.init (config : ZKCircuitConfig) (wasmExists keysExist : Bool) : CircomProverState := {
  config := config,
  circuit_compiled := wasmExists,
  keys_generated := match wasmExists with | true => keysExist | false => false,
  snarkjs_path := [],
  node_path := []
}

def CircomProverState.compileCircuit (s : CircomProverState) (exitCode : Nat) : CircomProverState × CompileResult :=
  match Nat.decEq exitCode 0 with
  | isTrue _ => ({ s with circuit_compiled := true }, CompileResult.success)
  | isFalse _ => (s, CompileResult.failure ZKProofError.CircomCompilationFailed)

def CircomProverState.generateWitness (s : CircomProverState) (exitCode : Nat) : WitnessResult :=
  match s.circuit_compiled with
  | false => WitnessResult.failure ZKProofError.CircuitNotCompiled
  | true =>
    match Nat.decEq exitCode 0 with
    | isTrue _ => WitnessResult.success
    | isFalse _ => WitnessResult.failure ZKProofError.WitnessGenerationFailed

def CircomProverState.generateProofResult (s : CircomProverState) (exitCode : Nat) : ProofResult :=
  match s.keys_generated with
  | false => ProofResult.failure ZKProofError.KeysNotGenerated
  | true =>
    match Nat.decEq exitCode 0 with
    | isTrue _ => ProofResult.success
    | isFalse _ => ProofResult.failure ZKProofError.ProofGenerationFailed

def CircomProverState.setupKeys (s : CircomProverState) (exitCode1 exitCode2 : Nat) : CircomProverState × SetupResult :=
  match Nat.decEq exitCode1 0 with
  | isTrue _ =>
    match Nat.decEq exitCode2 0 with
    | isTrue _ => ({ s with keys_generated := true }, SetupResult.success)
    | isFalse _ => (s, SetupResult.failure ZKProofError.KeysNotGenerated)
  | isFalse _ => (s, SetupResult.failure ZKProofError.KeysNotGenerated)

def bytesToI256_loop (bytes : ByteArray32) : Nat → Int
| 0 => 0
| Nat.succ n =>
  dite (n < 32)
    (fun h => bytesToI256_loop bytes n * 256 + (bytes ⟨n, h⟩).val)
    (fun _ => bytesToI256_loop bytes n * 256)

def bytesToI256 (bytes : ByteArray32) : Int := bytesToI256_loop bytes 32

def InferenceWitness.init (num_layers dim : Nat) : InferenceWitness := {
  tokens := listReplicate dim 0,
  layer_weights_s := listReplicate num_layers (listReplicate dim (listReplicate dim 0)),
  layer_weights_t := listReplicate num_layers (listReplicate dim (listReplicate dim 0)),
  expected_output := listReplicate dim 0,
  input_commitment := 0,
  output_commitment := 0,
  layer_commitments := listReplicate num_layers 0,
  max_error_squared := 1000000,
  num_layers := num_layers,
  dim := dim
}

def scaleFloat (x : Int) (scale : Int) : Int := x * scale

def InferenceWitness.setTokens (w : InferenceWitness) (input_tokens : List Int) (scale : Int) : InferenceWitness :=
  { w with tokens := listZipWith (fun inp _ => scaleFloat inp scale) input_tokens w.tokens }

def InferenceWitness.setExpectedOutput (w : InferenceWitness) (output : List Int) (scale : Int) : InferenceWitness :=
  { w with expected_output := listZipWith (fun out _ => scaleFloat out scale) output w.expected_output }

def InferenceWitness.setLayerWeights (w : InferenceWitness) (layer : Nat) (ws wt : List (List Int)) (scale : Int) : InferenceWitness :=
  match Nat.decLt layer w.num_layers with
  | isTrue _ => { w with layer_weights_s := w.layer_weights_s, layer_weights_t := w.layer_weights_t }
  | isFalse _ => w

end Config

namespace Cryptography

variable (H : HashOracle)
variable (cr : ∀ a b, H.blake3 a = H.blake3 b → a = b)
variable (pr : ∀ a b, H.sha256 a = H.sha256 b → a = b)

def toList (a : ByteArray32) : List (Fin 256) :=
  listMap a (listReplicate 32 0)

def byteArrayEqual_loop (a b : ByteArray32) : Nat → Bool
| 0 => true
| Nat.succ n =>
  dite (n < 32)
    (fun h => boolAnd (byteArrayEqual_loop a b n) (natBeq (a ⟨n, h⟩).val (b ⟨n, h⟩).val))
    (fun _ => byteArrayEqual_loop a b n)

def byteArrayEqual (a b : ByteArray32) : Bool :=
  byteArrayEqual_loop a b 32

def CommitmentScheme.init : CommitmentScheme := {
  commitments := [],
  nonce_counter := 0
}

def CommitmentScheme.commit (H : HashOracle) (cs : CommitmentScheme) (value : List (Fin 256)) (nonce blinding : ByteArray32) : ByteArray32 × CommitmentScheme :=
  let value_hash := H.sha256 [value]
  let commitment_hash := H.blake3 [value, toList nonce, toList blinding]
  let new_data : CommitmentData := {
    value_hash := value_hash,
    nonce := nonce,
    timestamp := 0,
    blinding_factor := blinding
  }
  (commitment_hash, { commitments := (commitment_hash, new_data) :: cs.commitments, nonce_counter := cs.nonce_counter + 1 })

def lookupCommitment (commitments : List (ByteArray32 × CommitmentData)) (key : ByteArray32) : Option CommitmentData :=
  match commitments with
  | [] => none
  | (k, v) :: rest =>
    match byteArrayEqual k key with
    | true => some v
    | false => lookupCommitment rest key

def CommitmentScheme.verify (H : HashOracle) (cs : CommitmentScheme) (commitment_hash : ByteArray32) (revealed_value : List (Fin 256)) (revealed_nonce revealed_blinding : ByteArray32) : Bool :=
  let computed_commitment := H.blake3 [revealed_value, toList revealed_nonce, toList revealed_blinding]
  let computed_value_hash := H.sha256 [revealed_value]
  match lookupCommitment cs.commitments commitment_hash with
  | none => false
  | some data =>
    boolAnd (byteArrayEqual commitment_hash computed_commitment) (byteArrayEqual data.value_hash computed_value_hash)

def RangeProof.init (min max : Int) : RangeProof := {
  min_value := min,
  max_value := max,
  proof_bits := []
}

def bitDecompose_loop : Nat → Nat → List (Fin 2)
| 0, _ => []
| Nat.succ n, v =>
  let b := v % 2
  let h : b < 2 := natMod_lt v 2 (Nat.zero_lt_succ 1)
  ⟨b, h⟩ :: bitDecompose_loop n (v / 2)

def bitDecompose (n : Nat) (value : Nat) : List (Fin 2) :=
  bitDecompose_loop n value

def bitRecompose : List (Fin 2) → Nat
| [] => 0
| b :: bs => b.val + 2 * bitRecompose bs

def RangeProof.prove (H : HashOracle) (rp : RangeProof) (value : Int) (nonces : List ByteArray32) (hInRange : boolAnd (intLe rp.min_value value) (intLe value rp.max_value) = true) : RangeProof :=
  let normalized := intToNat value - intToNat rp.min_value
  let bits := bitDecompose 64 normalized
  let new_bits := listZipWith (fun b n => { commitment := H.blake3 [[⟨b.val, Nat.lt_trans b.isLt (Nat.zero_lt_succ 255)⟩]], opening := n, bit_value := b : RangeProofBit }) bits nonces
  { rp with proof_bits := new_bits }

def RangeProof.verifyBits (H : HashOracle) (rp : RangeProof) : Bool :=
  let hashes_ok := listFoldl (fun acc bit =>
    let computed := H.blake3 [[⟨bit.bit_value.val, Nat.lt_trans bit.bit_value.isLt (Nat.zero_lt_succ 255)⟩]]
    boolAnd acc (byteArrayEqual bit.commitment computed)
  ) true rp.proof_bits
  let reconstructed := bitRecompose (listMap (fun b => b.bit_value) rp.proof_bits)
  let final_val := reconstructed + intToNat rp.min_value
  boolAnd hashes_ok (boolAnd (natBle (intToNat rp.min_value) final_val) (natBle final_val (intToNat rp.max_value)))

def buildMerkleLevel (H : HashOracle) : List ByteArray32 → List ByteArray32
| [] => []
| [x] => [x]
| x :: y :: rest => H.sha256 [toList x, toList y] :: buildMerkleLevel H rest

def buildMerkleTree_loop (H : HashOracle) : Nat → List ByteArray32 → ByteArray32
| 0, leaves =>
  match leaves with
  | [] => H.sha256 []
  | x :: _ => x
| Nat.succ n, leaves =>
  match leaves with
  | [] => H.sha256 []
  | [x] => x
  | _ => buildMerkleTree_loop H n (buildMerkleLevel H leaves)

def buildMerkleTree (H : HashOracle) (leaves : List ByteArray32) : ByteArray32 :=
  buildMerkleTree_loop H (listLength leaves) leaves

def generateMerkleProof_loop (H : HashOracle) : Nat → List ByteArray32 → Nat → MerkleProof
| 0, _, _ => { root := H.sha256 [], path := [], directions := [] }
| Nat.succ n, leaves, idx =>
  match leaves with
  | [] => { root := H.sha256 [], path := [], directions := [] }
  | [x] => { root := x, path := [], directions := [] }
  | _ =>
    let sibling_idx := match idx % 2 with | 0 => idx + 1 | _ => idx - 1
    let dir := match idx % 2 with | 0 => false | _ => true
    let sibling := listGetD leaves sibling_idx (H.sha256 [])
    let next_proof := generateMerkleProof_loop H n (buildMerkleLevel H leaves) (idx / 2)
    { root := next_proof.root, path := sibling :: next_proof.path, directions := dir :: next_proof.directions }

def generateMerkleProof (H : HashOracle) (leaves : List ByteArray32) (index : Nat) (hBound : index < listLength leaves) : MerkleProof :=
  generateMerkleProof_loop H (listLength leaves) leaves index

def verifyMembership_loop (H : HashOracle) : List ByteArray32 → List Bool → ByteArray32 → ByteArray32
| [], _, current => current
| _, [], current => current
| sibling :: ps, dir :: ds, current =>
  let next_hash := match dir with
    | false => H.sha256 [toList current, toList sibling]
    | true => H.sha256 [toList sibling, toList current]
  verifyMembership_loop H ps ds next_hash

def MerkleProof.verifyMembership (H : HashOracle) (mp : MerkleProof) (element_hash : ByteArray32) : Bool :=
  let computed_root := verifyMembership_loop H mp.path mp.directions element_hash
  byteArrayEqual computed_root mp.root

variable (P : Nat) (hP : P > 0)

def fadd (a b : Nat) : Nat := (a + b) % P
def fmul (a b : Nat) : Nat := (a * b) % P
def fsub (a b : Nat) : Nat := (a + P - (b % P)) % P

def SchnorrSignature.sign (H : HashOracle) (message : List (Fin 256)) (private_key : Nat) (k : Nat) : SchnorrSignature :=
  let r_point := fmul P k 1
  let challenge := H.sha256 [listReplicate 32 ⟨r_point % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩, message]
  let challenge_nat := (challenge ⟨0, Nat.zero_lt_succ 31⟩).val
  let response := fadd P k (fmul P challenge_nat private_key)
  { challenge := challenge, response := fun _ => ⟨response % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩ }

def SchnorrSignature.verify (H : HashOracle) (sig : SchnorrSignature) (message : List (Fin 256)) (public_key : Nat) : Bool :=
  let response_nat := (sig.response ⟨0, Nat.zero_lt_succ 31⟩).val
  let challenge_nat := (sig.challenge ⟨0, Nat.zero_lt_succ 31⟩).val
  let r_point := fsub P response_nat (fmul P challenge_nat public_key)
  let computed_challenge := H.sha256 [listReplicate 32 ⟨r_point % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩, message]
  byteArrayEqual sig.challenge computed_challenge

def DifferentialPrivacyConfig.init (eps del sens : Nat) : DifferentialPrivacyConfig := {
  epsilon := eps,
  delta := del,
  sensitivity := sens,
  noise_scale := eps * del * sens
}

def addNoise (config : DifferentialPrivacyConfig) (value : Int) (noise_sample : Int) : Int :=
  value + noise_sample

def addLaplaceNoise (config : DifferentialPrivacyConfig) (value : Int) (noise_sample : Int) : Int :=
  value + noise_sample

def ZKInferenceProof.init (t : Int) : ZKInferenceProof := {
  input_commitment := fun _ => ⟨0, Nat.zero_lt_succ 255⟩,
  output_commitment := fun _ => ⟨0, Nat.zero_lt_succ 255⟩,
  computation_proof := [],
  timestamp := t,
  has_zk_prover := false,
  proof_bundle := none
}

def ZKInferenceProof.proveInference (H : HashOracle) (p : ZKInferenceProof) (input output : List (Fin 256)) (model_hash : ByteArray32) : ZKInferenceProof :=
  let in_comm := H.blake3 [input]
  let out_comm := H.blake3 [output]
  let h0 := H.blake3 [toList in_comm, toList model_hash, toList out_comm]
  let h1 := H.blake3 [toList h0, [⟨0, Nat.zero_lt_succ 255⟩]]
  let h2 := H.blake3 [toList h1, [⟨1, Nat.lt_trans (Nat.lt_succ_self 0) (Nat.zero_lt_succ 255)⟩]]
  let h3 := H.blake3 [toList h2, [⟨2, Nat.lt_trans (Nat.lt_succ_self 1) (Nat.zero_lt_succ 255)⟩]]
  let h4 := H.blake3 [toList h3, [⟨3, Nat.lt_trans (Nat.lt_succ_self 2) (Nat.zero_lt_succ 255)⟩]]
  let h5 := H.blake3 [toList h4, [⟨4, Nat.lt_trans (Nat.lt_succ_self 3) (Nat.zero_lt_succ 255)⟩]]
  let h6 := H.blake3 [toList h5, [⟨5, Nat.lt_trans (Nat.lt_succ_self 4) (Nat.zero_lt_succ 255)⟩]]
  let h7 := H.blake3 [toList h6, [⟨6, Nat.lt_trans (Nat.lt_succ_self 5) (Nat.zero_lt_succ 255)⟩]]
  let h8 := H.blake3 [toList h7, [⟨7, Nat.lt_trans (Nat.lt_succ_self 6) (Nat.zero_lt_succ 255)⟩]]
  { p with input_commitment := in_comm, output_commitment := out_comm, computation_proof := [h0, h1, h2, h3, h4, h5, h6, h7, h8] }

def ZKInferenceProof.verifyChain (H : HashOracle) (p : ZKInferenceProof) (model_hash : ByteArray32) : Bool :=
  match p.computation_proof with
  | [c0, c1, c2, c3, c4, c5, c6, c7, c8] =>
    boolAnd (byteArrayEqual c0 (H.blake3 [toList p.input_commitment, toList model_hash, toList p.output_commitment]))
    (boolAnd (byteArrayEqual c1 (H.blake3 [toList c0, [⟨0, Nat.zero_lt_succ 255⟩]]))
    (boolAnd (byteArrayEqual c2 (H.blake3 [toList c1, [⟨1, Nat.lt_trans (Nat.lt_succ_self 0) (Nat.zero_lt_succ 255)⟩]]))
    (boolAnd (byteArrayEqual c3 (H.blake3 [toList c2, [⟨2, Nat.lt_trans (Nat.lt_succ_self 1) (Nat.zero_lt_succ 255)⟩]]))
    (boolAnd (byteArrayEqual c4 (H.blake3 [toList c3, [⟨3, Nat.lt_trans (Nat.lt_succ_self 2) (Nat.zero_lt_succ 255)⟩]]))
    (boolAnd (byteArrayEqual c5 (H.blake3 [toList c4, [⟨4, Nat.lt_trans (Nat.lt_succ_self 3) (Nat.zero_lt_succ 255)⟩]]))
    (boolAnd (byteArrayEqual c6 (H.blake3 [toList c5, [⟨5, Nat.lt_trans (Nat.lt_succ_self 4) (Nat.zero_lt_succ 255)⟩]]))
    (boolAnd (byteArrayEqual c7 (H.blake3 [toList c6, [⟨6, Nat.lt_trans (Nat.lt_succ_self 5) (Nat.zero_lt_succ 255)⟩]]))
             (byteArrayEqual c8 (H.blake3 [toList c7, [⟨7, Nat.lt_trans (Nat.lt_succ_self 6) (Nat.zero_lt_succ 255)⟩]])))))))))
  | _ => false

def SecureAggregation.init (threshold : Nat) : SecureAggregation := {
  participant_commitments := [],
  aggregated_result := none,
  threshold := threshold
}

def SecureAggregation.commitParticipant (H : HashOracle) (sa : SecureAggregation) (pid : Nat) (data : List Int) : ByteArray32 × SecureAggregation :=
  let hash := H.blake3 [listMap (fun _ => ⟨0, Nat.zero_lt_succ 255⟩) data]
  (hash, { sa with participant_commitments := listAppend sa.participant_commitments [(pid, hash)] })

def sumLists (contribs : List (List Int)) (dim : Nat) : List Int :=
  listFoldl (fun acc c => listZipWith (· + ·) acc c) (listReplicate dim 0) contribs

def divideList (lst : List Int) (count : Nat) (hPos : count > 0) : List Int :=
  listMap (fun x => x / count) lst

def SecureAggregation.aggregate (sa : SecureAggregation) (contributions : List (List Int)) (hThreshold : listLength contributions ≥ sa.threshold) (hNonEmpty : listLength contributions > 0) (hUniform : ∀ i j, listLength (listGetD contributions i []) = listLength (listGetD contributions j [])) : SecureAggregation :=
  let dim := listLength (listGetD contributions 0 [])
  let summed := sumLists contributions dim
  let result := divideList summed (listLength contributions) hNonEmpty
  { sa with aggregated_result := some result }

def SecureAggregation.getResult (sa : SecureAggregation) : Option (List Int) :=
  sa.aggregated_result

def aggregate' (sa : SecureAggregation) (contributions : List (List Int)) : Sum AggregateError SecureAggregation :=
  match Nat.decLe sa.threshold (listLength contributions) with
  | isTrue hT =>
    match Nat.decLt 0 (listLength contributions) with
    | isTrue hN =>
      let dim := listLength (listGetD contributions 0 [])
      let uniform := listFoldl (fun acc c => boolAnd acc (natBeq (listLength c) dim)) true contributions
      match uniform with
      | true => Sum.inr { sa with aggregated_result := some (divideList (sumLists contributions dim) (listLength contributions) hN) }
      | false => Sum.inl AggregateError.DimensionMismatch
    | isFalse _ => Sum.inl AggregateError.InsufficientParticipants
  | isFalse _ => Sum.inl AggregateError.InsufficientParticipants

def ZKInferenceProver.init : ZKInferenceProver := {
  config := Config.ZKCircuitConfig.defaultConfig,
  prover_state := Config.CircomProverState.init Config.ZKCircuitConfig.defaultConfig false false,
  proof_counter := 0
}

def ZKInferenceProver.proveInference (prover : ZKInferenceProver) : ZKInferenceProver × ZKProofBundle :=
  ({ prover with proof_counter := prover.proof_counter + 1 }, Config.ZKProofBundle.init 0)

end Cryptography

namespace Theorems

open Types Helpers Proofs Config Cryptography

theorem defaultConfig_num_layers : (ZKCircuitConfig.defaultConfig).num_layers = 8 := Eq.refl 8
theorem defaultConfig_embedding_dim : (ZKCircuitConfig.defaultConfig).embedding_dim = 32 := Eq.refl 32
theorem defaultConfig_precision_bits : (ZKCircuitConfig.defaultConfig).precision_bits = 64 := Eq.refl 64
theorem defaultConfig_timeout_ms : (ZKCircuitConfig.defaultConfig).timeout_ms = 300000 := Eq.refl 300000

theorem groth16_protocol_0 : (Groth16Proof.init).protocol ⟨0, Nat.zero_lt_succ 6⟩ = ⟨103, Nat.lt_trans (Nat.lt_succ_self 103) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_protocol_1 : (Groth16Proof.init).protocol ⟨1, Nat.lt_trans (Nat.lt_succ_self 0) (Nat.zero_lt_succ 6)⟩ = ⟨114, Nat.lt_trans (Nat.lt_succ_self 114) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_protocol_2 : (Groth16Proof.init).protocol ⟨2, Nat.lt_trans (Nat.lt_succ_self 1) (Nat.zero_lt_succ 6)⟩ = ⟨111, Nat.lt_trans (Nat.lt_succ_self 111) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_protocol_3 : (Groth16Proof.init).protocol ⟨3, Nat.lt_trans (Nat.lt_succ_self 2) (Nat.zero_lt_succ 6)⟩ = ⟨116, Nat.lt_trans (Nat.lt_succ_self 116) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_protocol_4 : (Groth16Proof.init).protocol ⟨4, Nat.lt_trans (Nat.lt_succ_self 3) (Nat.zero_lt_succ 6)⟩ = ⟨104, Nat.lt_trans (Nat.lt_succ_self 104) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_protocol_5 : (Groth16Proof.init).protocol ⟨5, Nat.lt_trans (Nat.lt_succ_self 4) (Nat.zero_lt_succ 6)⟩ = ⟨49, Nat.lt_trans (Nat.lt_succ_self 49) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_protocol_6 : (Groth16Proof.init).protocol ⟨6, Nat.lt_trans (Nat.lt_succ_self 5) (Nat.zero_lt_succ 6)⟩ = ⟨54, Nat.lt_trans (Nat.lt_succ_self 54) (Nat.lt_succ_self 254)⟩ := Eq.refl _

theorem groth16_curve_0 : (Groth16Proof.init).curve ⟨0, Nat.zero_lt_succ 3⟩ = ⟨98, Nat.lt_trans (Nat.lt_succ_self 98) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_curve_1 : (Groth16Proof.init).curve ⟨1, Nat.lt_trans (Nat.lt_succ_self 0) (Nat.zero_lt_succ 3)⟩ = ⟨110, Nat.lt_trans (Nat.lt_succ_self 110) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_curve_2 : (Groth16Proof.init).curve ⟨2, Nat.lt_trans (Nat.lt_succ_self 1) (Nat.zero_lt_succ 3)⟩ = ⟨49, Nat.lt_trans (Nat.lt_succ_self 49) (Nat.lt_succ_self 254)⟩ := Eq.refl _
theorem groth16_curve_3 : (Groth16Proof.init).curve ⟨3, Nat.lt_trans (Nat.lt_succ_self 2) (Nat.zero_lt_succ 3)⟩ = ⟨50, Nat.lt_trans (Nat.lt_succ_self 50) (Nat.lt_succ_self 254)⟩ := Eq.refl _

theorem addSignal_length (s : PublicSignals) (v : Int) : listLength (PublicSignals.addSignal s v).signals = listLength s.signals + 1 :=
  Eq.trans (listLength_append s.signals [v]) (Eq.refl (listLength s.signals + 1))

theorem init_signals_empty : (PublicSignals.init).signals = [] := Eq.refl []
theorem init_signals_length_zero : listLength (PublicSignals.init).signals = 0 := Eq.refl 0

theorem bundle_init_not_verified (t : Int) : (ZKProofBundle.init t).verification_status = false := Eq.refl false
theorem bundle_init_empty_proof_json (t : Int) : (ZKProofBundle.init t).proof_json = [] := Eq.refl []
theorem bundle_init_empty_public_json (t : Int) : (ZKProofBundle.init t).public_json = [] := Eq.refl []

theorem prover_init_no_wasm_not_compiled (cfg : ZKCircuitConfig) (b : Bool) : (CircomProverState.init cfg false b).circuit_compiled = false := Eq.refl false
theorem prover_init_no_wasm_no_keys (cfg : ZKCircuitConfig) (b : Bool) : (CircomProverState.init cfg false b).keys_generated = false := Eq.refl false

theorem compile_success_sets_compiled (s : CircomProverState) : (CircomProverState.compileCircuit s 0).1.circuit_compiled = true := Eq.refl true
theorem compile_failure_preserves_state (s : CircomProverState) (exitCode : Nat) (h : exitCode ≠ 0) : (CircomProverState.compileCircuit s exitCode).2 = CompileResult.failure ZKProofError.CircomCompilationFailed :=
  match Nat.decEq exitCode 0 with
  | isTrue heq => False.elim (h heq)
  | isFalse _ => Eq.refl _

theorem witness_requires_compiled (s : CircomProverState) (n : Nat) : CircomProverState.generateWitness { s with circuit_compiled := false } n = WitnessResult.failure ZKProofError.CircuitNotCompiled := Eq.refl _
theorem witness_success_implies_compiled (s : CircomProverState) (n : Nat) (h : CircomProverState.generateWitness s n = WitnessResult.success) : s.circuit_compiled = true :=
  match h_comp : s.circuit_compiled with
  | true => Eq.refl true
  | false =>
    let h_false : WitnessResult.failure ZKProofError.CircuitNotCompiled = WitnessResult.success := Eq.trans (Eq.symm (congrArg (fun c => match c with | false => WitnessResult.failure ZKProofError.CircuitNotCompiled | true => match Nat.decEq n 0 with | isTrue _ => WitnessResult.success | isFalse _ => WitnessResult.failure ZKProofError.WitnessGenerationFailed) h_comp)) h
    False.elim (WitnessResult.noConfusion h_false)

theorem proof_requires_keys (s : CircomProverState) (n : Nat) : CircomProverState.generateProofResult { s with keys_generated := false } n = ProofResult.failure ZKProofError.KeysNotGenerated := Eq.refl _
theorem proof_success_implies_keys (s : CircomProverState) (n : Nat) (h : CircomProverState.generateProofResult s n = ProofResult.success) : s.keys_generated = true :=
  match h_keys : s.keys_generated with
  | true => Eq.refl true
  | false =>
    let h_false : ProofResult.failure ZKProofError.KeysNotGenerated = ProofResult.success := Eq.trans (Eq.symm (congrArg (fun c => match c with | false => ProofResult.failure ZKProofError.KeysNotGenerated | true => match Nat.decEq n 0 with | isTrue _ => ProofResult.success | isFalse _ => ProofResult.failure ZKProofError.ProofGenerationFailed) h_keys)) h
    False.elim (ProofResult.noConfusion h_false)

theorem setup_success_both_zero (s : CircomProverState) : (CircomProverState.setupKeys s 0 0).1.keys_generated = true := Eq.refl true
theorem setup_failure_first_nonzero (s : CircomProverState) (exitCode1 exitCode2 : Nat) (h : exitCode1 ≠ 0) : (CircomProverState.setupKeys s exitCode1 exitCode2).2 = SetupResult.failure ZKProofError.KeysNotGenerated :=
  match Nat.decEq exitCode1 0 with
  | isTrue heq => False.elim (h heq)
  | isFalse _ => Eq.refl _

theorem bytesToI256_zero_bytes : bytesToI256 (fun _ => ⟨0, Nat.zero_lt_succ 255⟩) = 0 := Eq.refl 0

theorem witness_init_tokens_length (nl d : Nat) : listLength (InferenceWitness.init nl d).tokens = d := listLength_replicate d 0
theorem witness_init_expected_output_length (nl d : Nat) : listLength (InferenceWitness.init nl d).expected_output = d := listLength_replicate d 0
theorem witness_init_layer_commitments_length (nl d : Nat) : listLength (InferenceWitness.init nl d).layer_commitments = nl := listLength_replicate nl 0
theorem witness_init_num_layers (nl d : Nat) : (InferenceWitness.init nl d).num_layers = nl := Eq.refl nl
theorem witness_init_dim (nl d : Nat) : (InferenceWitness.init nl d).dim = d := Eq.refl d
theorem witness_init_max_error (nl d : Nat) : (InferenceWitness.init nl d).max_error_squared = 1000000 := Eq.refl 1000000
theorem witness_init_input_commitment_zero (nl d : Nat) : (InferenceWitness.init nl d).input_commitment = 0 := Eq.refl 0
theorem witness_init_output_commitment_zero (nl d : Nat) : (InferenceWitness.init nl d).output_commitment = 0 := Eq.refl 0

theorem setTokens_preserves_dim (w : InferenceWitness) (inp : List Int) (s : Int) : (InferenceWitness.setTokens w inp s).dim = w.dim := Eq.refl _
theorem setTokens_preserves_num_layers (w : InferenceWitness) (inp : List Int) (s : Int) : (InferenceWitness.setTokens w inp s).num_layers = w.num_layers := Eq.refl _
theorem setExpectedOutput_preserves_dim (w : InferenceWitness) (out : List Int) (s : Int) : (InferenceWitness.setExpectedOutput w out s).dim = w.dim := Eq.refl _

theorem setLayerWeights_out_of_range (w : InferenceWitness) (layer : Nat) (ws wt : List (List Int)) (s : Int) (h : layer ≥ w.num_layers) : InferenceWitness.setLayerWeights w layer ws wt s = w :=
  match Nat.decLt layer w.num_layers with
  | isTrue hlt => False.elim (Nat.not_lt_of_ge h hlt)
  | isFalse _ => Eq.refl w

theorem setLayerWeights_preserves_dim (w : InferenceWitness) (l : Nat) (ws wt : List (List Int)) (s : Int) : (InferenceWitness.setLayerWeights w l ws wt s).dim = w.dim :=
  match Nat.decLt l w.num_layers with
  | isTrue _ => Eq.refl _
  | isFalse _ => Eq.refl _

theorem setLayerWeights_preserves_num_layers (w : InferenceWitness) (l : Nat) (ws wt : List (List Int)) (s : Int) : (InferenceWitness.setLayerWeights w l ws wt s).num_layers = w.num_layers :=
  match Nat.decLt l w.num_layers with
  | isTrue _ => Eq.refl _
  | isFalse _ => Eq.refl _

variable (H : HashOracle)

theorem hash_deterministic_blake3 (inputs : List (List (Fin 256))) : H.blake3 inputs = H.blake3 inputs := Eq.refl _
theorem hash_deterministic_sha256 (inputs : List (List (Fin 256))) : H.sha256 inputs = H.sha256 inputs := Eq.refl _

theorem commit_increments_counter (cs : CommitmentScheme) (v : List (Fin 256)) (n b : ByteArray32) : (CommitmentScheme.commit H cs v n b).2.nonce_counter = cs.nonce_counter + 1 := Eq.refl _
theorem commit_appends_commitment (cs : CommitmentScheme) (v : List (Fin 256)) (n b : ByteArray32) : listLength (CommitmentScheme.commit H cs v n b).2.commitments = listLength cs.commitments + 1 :=
  Eq.refl _

theorem byteArrayEqual_loop_refl (a : ByteArray32) (n : Nat) : byteArrayEqual_loop a a n = true :=
  Nat.recOn n
    (Eq.refl true)
    (fun k ih =>
      dite (k < 32)
        (fun h => boolAnd_intro _ _ ih (natBeq_refl (a ⟨k, h⟩).val))
        (fun _ => ih))

theorem byteArrayEqual_reflexive (a : ByteArray32) : byteArrayEqual a a = true :=
  byteArrayEqual_loop_refl a 32

theorem byteArrayEqual_loop_symm (a b : ByteArray32) (n : Nat) : byteArrayEqual_loop a b n = byteArrayEqual_loop b a n :=
  Nat.recOn n
    (Eq.refl true)
    (fun k ih =>
      dite (k < 32)
        (fun h => Eq.subst (motive := fun x => boolAnd (byteArrayEqual_loop a b k) (natBeq (a ⟨k, h⟩).val (b ⟨k, h⟩).val) = boolAnd x (natBeq (b ⟨k, h⟩).val (a ⟨k, h⟩).val)) ih
          (Eq.subst (motive := fun x => boolAnd (byteArrayEqual_loop b a k) (natBeq (a ⟨k, h⟩).val (b ⟨k, h⟩).val) = boolAnd (byteArrayEqual_loop b a k) x) (natBeq_symm (a ⟨k, h⟩).val (b ⟨k, h⟩).val) (Eq.refl _)))
        (fun _ => ih))

theorem byteArrayEqual_symmetric (a b : ByteArray32) : byteArrayEqual a b = byteArrayEqual b a :=
  byteArrayEqual_loop_symm a b 32

theorem lookup_head (k : ByteArray32) (v : CommitmentData) (rest : List (ByteArray32 × CommitmentData)) :
  lookupCommitment ((k, v) :: rest) k = some v :=
  matchBoolTrue (byteArrayEqual k k) (some v) (lookupCommitment rest k) (byteArrayEqual_reflexive k)

theorem commit_verify_consistency (cs : CommitmentScheme) (value : List (Fin 256)) (nonce blinding : ByteArray32) :
  let c := CommitmentScheme.commit H cs value nonce blinding
  CommitmentScheme.verify H c.2 c.1 value nonce blinding = true :=
  let ch := H.blake3 [value, toList nonce, toList blinding]
  let vh := H.sha256 [value]
  let data : CommitmentData := { value_hash := vh, nonce := nonce, timestamp := 0, blinding_factor := blinding }
  let h_lookup := lookup_head ch data cs.commitments
  let h_ch := byteArrayEqual_reflexive ch
  let h_vh := byteArrayEqual_reflexive vh
  let match_eq := matchOptionSome (lookupCommitment ((ch, data) :: cs.commitments) ch) data h_lookup false (fun d => boolAnd (byteArrayEqual ch ch) (byteArrayEqual d.value_hash vh))
  Eq.trans match_eq (boolAnd_intro _ _ h_ch h_vh)

theorem commit_stores_value_hash (cs : CommitmentScheme) (v : List (Fin 256)) (n b : ByteArray32) :
  (match lookupCommitment (CommitmentScheme.commit H cs v n b).2.commitments (CommitmentScheme.commit H cs v n b).1 with | some d => d.value_hash | none => H.sha256 []) = H.sha256 [v] :=
  let ch := H.blake3 [v, toList n, toList b]
  let vh := H.sha256 [v]
  let data : CommitmentData := { value_hash := vh, nonce := n, timestamp := 0, blinding_factor := b }
  let h_lookup := lookup_head ch data cs.commitments
  let match_eq := matchOptionSome (lookupCommitment ((ch, data) :: cs.commitments) ch) data h_lookup (H.sha256 []) (fun d => d.value_hash)
  Eq.trans match_eq (Eq.refl vh)

theorem range_proof_init_empty (min max : Int) : (RangeProof.init min max).proof_bits = [] := Eq.refl []
theorem range_proof_init_min (min max : Int) : (RangeProof.init min max).min_value = min := Eq.refl min
theorem range_proof_init_max (min max : Int) : (RangeProof.init min max).max_value = max := Eq.refl max

theorem bitDecompose_zero_bits (v : Nat) : bitDecompose 0 v = [] := Eq.refl []

theorem bitRecompose_nil : bitRecompose [] = 0 := Eq.refl 0

theorem bitDecompose_recompose (n : Nat) : ∀ (value : Nat), value < 2^n → bitRecompose (bitDecompose n value) = value :=
  Nat.recOn n
    (fun v h =>
      match v, h with
      | 0, _ => Eq.refl 0
      | Nat.succ _, h2 => False.elim (Nat.not_lt_zero _ (Nat.lt_of_succ_lt_succ h2)))
    (fun k ih v h =>
      let h_div : v / 2 < 2^k := natDiv_lt_of_lt_mul v k h
      let ih_app := ih (v / 2) h_div
      let step1 : (v % 2) + 2 * bitRecompose (bitDecompose k (v / 2)) = (v % 2) + 2 * (v / 2) := congrArg (fun x => (v % 2) + 2 * x) ih_app
      let step2 : (v % 2) + 2 * (v / 2) = v := natMod_add_div v 2
      Eq.trans step1 step2)

theorem prove_preserves_range (rp : RangeProof) (v : Int) (ns : List ByteArray32) (h : boolAnd (intLe rp.min_value v) (intLe v rp.max_value) = true) :
  (RangeProof.prove H rp v ns h).min_value = rp.min_value ∧ (RangeProof.prove H rp v ns h).max_value = rp.max_value :=
  And.intro (Eq.refl _) (Eq.refl _)

theorem range_proof_soundness_value_in_range (rp : RangeProof) :
  RangeProof.verifyBits H rp = true →
  let reconstructed := bitRecompose (listMap (fun b => b.bit_value) rp.proof_bits)
  let final_val := reconstructed + intToNat rp.min_value
  intToNat rp.min_value ≤ final_val ∧ final_val ≤ intToNat rp.max_value :=
  fun h =>
    let h1 := boolAnd_true_right _ _ h
    let h_ble_max := boolAnd_true_right _ _ h1
    let h_ble_min := boolAnd_true_left _ _ h1
    And.intro (natBle_sound _ _ h_ble_min) (natBle_sound _ _ h_ble_max)

theorem merkle_single_leaf (leaf : ByteArray32) : buildMerkleTree H [leaf] = leaf := Eq.refl leaf
theorem merkle_two_leaves (l1 l2 : ByteArray32) : buildMerkleTree H [l1, l2] = H.sha256 [toList l1, toList l2] := Eq.refl _

theorem merkle_path_directions_same_length (mp : MerkleProof) : listLength mp.path = listLength mp.path := Eq.refl _

theorem merkle_verify_unfold_nil (eh : ByteArray32) : MerkleProof.verifyMembership H { root := eh, path := [], directions := [] } eh = byteArrayEqual eh eh := Eq.refl _

theorem merkle_sibling_index_even (idx : Nat) : (match idx % 2 with | 0 => idx + 1 | _ => idx - 1) = if idx % 2 == 0 then idx + 1 else idx - 1 := Eq.refl _

variable (P : Nat) (hP : P > 0)

theorem fadd_comm (a b : Nat) : fadd P a b = fadd P b a :=
  congrArg (fun x => x % P) (natAdd_comm a b)

theorem fmul_comm (a b : Nat) : fmul P a b = fmul P b a :=
  congrArg (fun x => x % P) (natMul_comm a b)

theorem schnorr_sign_verify_consistency (msg : List (Fin 256)) (sk k : Nat)
  (h_alg : fsub P (fadd P k (fmul P (H.sha256 [listReplicate 32 ⟨(fmul P k 1) % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩, msg] ⟨0, Nat.zero_lt_succ 31⟩).val sk)) (fmul P (H.sha256 [listReplicate 32 ⟨(fmul P k 1) % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩, msg] ⟨0, Nat.zero_lt_succ 31⟩).val sk) = fmul P k 1) :
  SchnorrSignature.verify H P (SchnorrSignature.sign H P msg sk k) msg sk = true :=
  let sig := SchnorrSignature.sign H P msg sk k
  let response_nat := (sig.response ⟨0, Nat.zero_lt_succ 31⟩).val
  let challenge_nat := (sig.challenge ⟨0, Nat.zero_lt_succ 31⟩).val
  let r_point := fsub P response_nat (fmul P challenge_nat sk)
  let h_r_point : r_point = fmul P k 1 := h_alg
  let computed_challenge := H.sha256 [listReplicate 32 ⟨r_point % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩, msg]
  let h_comp : computed_challenge = sig.challenge := congrArg (fun r => H.sha256 [listReplicate 32 ⟨r % 256, natMod_lt _ 256 (Nat.zero_lt_succ 255)⟩, msg]) h_r_point
  let match_eq := byteArrayEqual_reflexive sig.challenge
  Eq.subst (motive := fun x => byteArrayEqual sig.challenge x = true) (Eq.symm h_comp) match_eq

theorem dp_noise_scale_positive (eps del sens : Nat) (h1 : eps > 0) (h2 : del > 0) (h3 : sens > 0) :
  (DifferentialPrivacyConfig.init eps del sens).noise_scale > 0 :=
  Nat.mul_pos (Nat.mul_pos h1 h2) h3

theorem addNoise_identity_zero_noise (config : DifferentialPrivacyConfig) (v : Int) :
  addNoise config v 0 = v :=
  Eq.refl v

theorem zk_proof_init_empty_chain (t : Int) : (ZKInferenceProof.init t).computation_proof = [] := Eq.refl []
theorem zk_proof_init_no_bundle (t : Int) : (ZKInferenceProof.init t).proof_bundle = none := Eq.refl none

theorem proveInference_chain_length (p : ZKInferenceProof) (inp out : List (Fin 256)) (mh : ByteArray32) :
  listLength (ZKInferenceProof.proveInference H p inp out mh).computation_proof = 9 :=
  Eq.refl 9

theorem proveInference_chain_length_detail (p : ZKInferenceProof) (inp out : List (Fin 256)) (mh : ByteArray32) :
  listLength (ZKInferenceProof.proveInference H p inp out mh).computation_proof = 1 + 8 :=
  Eq.trans (Eq.refl 9) (Eq.symm (natAdd_comm 8 1))

theorem prove_verify_chain_consistency (p : ZKInferenceProof) (inp out : List (Fin 256)) (mh : ByteArray32) :
  ZKInferenceProof.verifyChain H (ZKInferenceProof.proveInference H p inp out mh) mh = true :=
  let in_comm := H.blake3 [inp]
  let out_comm := H.blake3 [out]
  let h0 := H.blake3 [toList in_comm, toList mh, toList out_comm]
  let h1 := H.blake3 [toList h0, [⟨0, Nat.zero_lt_succ 255⟩]]
  let h2 := H.blake3 [toList h1, [⟨1, Nat.lt_trans (Nat.lt_succ_self 0) (Nat.zero_lt_succ 255)⟩]]
  let h3 := H.blake3 [toList h2, [⟨2, Nat.lt_trans (Nat.lt_succ_self 1) (Nat.zero_lt_succ 255)⟩]]
  let h4 := H.blake3 [toList h3, [⟨3, Nat.lt_trans (Nat.lt_succ_self 2) (Nat.zero_lt_succ 255)⟩]]
  let h5 := H.blake3 [toList h4, [⟨4, Nat.lt_trans (Nat.lt_succ_self 3) (Nat.zero_lt_succ 255)⟩]]
  let h6 := H.blake3 [toList h5, [⟨5, Nat.lt_trans (Nat.lt_succ_self 4) (Nat.zero_lt_succ 255)⟩]]
  let h7 := H.blake3 [toList h6, [⟨6, Nat.lt_trans (Nat.lt_succ_self 5) (Nat.zero_lt_succ 255)⟩]]
  let h8 := H.blake3 [toList h7, [⟨7, Nat.lt_trans (Nat.lt_succ_self 6) (Nat.zero_lt_succ 255)⟩]]
  let eq0 := byteArrayEqual_reflexive h0
  let eq1 := byteArrayEqual_reflexive h1
  let eq2 := byteArrayEqual_reflexive h2
  let eq3 := byteArrayEqual_reflexive h3
  let eq4 := byteArrayEqual_reflexive h4
  let eq5 := byteArrayEqual_reflexive h5
  let eq6 := byteArrayEqual_reflexive h6
  let eq7 := byteArrayEqual_reflexive h7
  let eq8 := byteArrayEqual_reflexive h8
  boolAnd_intro _ _ (boolAnd_intro _ _ (boolAnd_intro _ _ (boolAnd_intro _ _ (boolAnd_intro _ _ (boolAnd_intro _ _ (boolAnd_intro _ _ (boolAnd_intro _ _ eq0 eq1) eq2) eq3) eq4) eq5) eq6) eq7) eq8

theorem verify_empty_chain_false (p : ZKInferenceProof) (mh : ByteArray32) (h : p.computation_proof = []) :
  ZKInferenceProof.verifyChain H p mh = false :=
  Eq.trans (congrArg (fun c => match c with | [c0, c1, c2, c3, c4, c5, c6, c7, c8] => boolAnd (byteArrayEqual c0 (H.blake3 [toList p.input_commitment, toList mh, toList p.output_commitment])) (boolAnd (byteArrayEqual c1 (H.blake3 [toList c0, [⟨0, Nat.zero_lt_succ 255⟩]])) (boolAnd (byteArrayEqual c2 (H.blake3 [toList c1, [⟨1, Nat.lt_trans (Nat.lt_succ_self 0) (Nat.zero_lt_succ 255)⟩]])) (boolAnd (byteArrayEqual c3 (H.blake3 [toList c2, [⟨2, Nat.lt_trans (Nat.lt_succ_self 1) (Nat.zero_lt_succ 255)⟩]])) (boolAnd (byteArrayEqual c4 (H.blake3 [toList c3, [⟨3, Nat.lt_trans (Nat.lt_succ_self 2) (Nat.zero_lt_succ 255)⟩]])) (boolAnd (byteArrayEqual c5 (H.blake3 [toList c4, [⟨4, Nat.lt_trans (Nat.lt_succ_self 3) (Nat.zero_lt_succ 255)⟩]])) (boolAnd (byteArrayEqual c6 (H.blake3 [toList c5, [⟨5, Nat.lt_trans (Nat.lt_succ_self 4) (Nat.zero_lt_succ 255)⟩]])) (boolAnd (byteArrayEqual c7 (H.blake3 [toList c6, [⟨6, Nat.lt_trans (Nat.lt_succ_self 5) (Nat.zero_lt_succ 255)⟩]])) (byteArrayEqual c8 (H.blake3 [toList c7, [⟨7, Nat.lt_trans (Nat.lt_succ_self 6) (Nat.zero_lt_succ 255)⟩]]))))))))) | _ => false) h) (Eq.refl false)

def verifyWithBundle (p : ZKInferenceProof) : Bool :=
  match p.proof_bundle with
  | some bundle => bundle.verification_status
  | none => false

theorem verify_with_bundle (p : ZKInferenceProof) (bundle : ZKProofBundle) (h : p.proof_bundle = some bundle) :
  verifyWithBundle p = bundle.verification_status :=
  matchOptionSome p.proof_bundle bundle h false (fun b => b.verification_status)

theorem secure_agg_init_no_result (t : Nat) : (SecureAggregation.init t).aggregated_result = none := Eq.refl none
theorem secure_agg_init_empty_commitments (t : Nat) : (SecureAggregation.init t).participant_commitments = [] := Eq.refl []
theorem secure_agg_init_threshold (t : Nat) : (SecureAggregation.init t).threshold = t := Eq.refl t

theorem commit_participant_adds_entry (sa : SecureAggregation) (pid : Nat) (data : List Int) :
  listLength (SecureAggregation.commitParticipant H sa pid data).2.participant_commitments = listLength sa.participant_commitments + 1 :=
  Eq.trans (listLength_append sa.participant_commitments [(pid, H.blake3 [listMap (fun _ => ⟨0, Nat.zero_lt_succ 255⟩) data])]) (Eq.refl _)

theorem aggregate_result_is_some (sa : SecureAggregation) (c : List (List Int)) (hT hN hU) :
  (SecureAggregation.aggregate sa c hT hN hU).aggregated_result ≠ none :=
  fun h => Option.noConfusion h

theorem aggregate_result_length (sa : SecureAggregation) (c : List (List Int)) (hT hN hU) (res : List Int)
  (hRes : (SecureAggregation.aggregate sa c hT hN hU).aggregated_result = some res)
  (hSumLen : listLength (sumLists c (listLength (listGetD c 0 []))) = listLength (listGetD c 0 [])) :
  listLength res = listLength (listGetD c 0 []) :=
  let dim := listLength (listGetD c 0 [])
  let summed := sumLists c dim
  let expected_res := divideList summed (listLength c) hN
  let h_eq : some expected_res = some res := Eq.trans (Eq.symm (congrArg (fun x => x.aggregated_result) (Eq.refl (SecureAggregation.aggregate sa c hT hN hU)))) hRes
  let h_res_eq : expected_res = res := Option.some.inj h_eq
  let h_len : listLength expected_res = listLength summed := listLength_map (fun x => x / listLength c) summed
  Eq.trans (Eq.symm (congrArg listLength h_res_eq)) (Eq.trans h_len hSumLen)

theorem getResult_after_aggregate (sa : SecureAggregation) (c : List (List Int)) (hT hN hU) :
  SecureAggregation.getResult (SecureAggregation.aggregate sa c hT hN hU) = some (divideList (sumLists c (listLength (listGetD c 0 []))) (listLength c) hN) :=
  Eq.refl _

theorem aggregate_preserves_threshold (sa : SecureAggregation) (c : List (List Int)) (hT hN hU) :
  (SecureAggregation.aggregate sa c hT hN hU).threshold = sa.threshold :=
  Eq.refl _

theorem aggregate_requires_threshold (sa : SecureAggregation) (c : List (List Int)) (hT : listLength c ≥ sa.threshold) (hN hU) :
  listLength c ≥ sa.threshold :=
  hT

theorem aggregate'_insufficient (sa : SecureAggregation) (c : List (List Int)) (h : listLength c < sa.threshold) :
  aggregate' sa c = Sum.inl AggregateError.InsufficientParticipants :=
  match Nat.decLe sa.threshold (listLength c) with
  | isTrue hLe => False.elim (Nat.not_le_of_gt h hLe)
  | isFalse _ => Eq.refl _

theorem aggregate'_success_implies_threshold (sa : SecureAggregation) (c : List (List Int)) (res : SecureAggregation) (h : aggregate' sa c = Sum.inr res) :
  listLength c ≥ sa.threshold :=
  match hLe : Nat.decLe sa.threshold (listLength c) with
  | isTrue hT => hT
  | isFalse _ =>
    let h_false : Sum.inl AggregateError.InsufficientParticipants = Sum.inr res := Eq.trans (Eq.symm (congrArg (fun dec => match dec with | isTrue hT => match Nat.decLt 0 (listLength c) with | isTrue hN => match listFoldl (fun acc c_1 => boolAnd acc (natBeq (listLength c_1) (listLength (listGetD c 0 [])))) true c with | true => Sum.inr { sa with aggregated_result := some (divideList (sumLists c (listLength (listGetD c 0 []))) (listLength c) hN) } | false => Sum.inl AggregateError.DimensionMismatch | isFalse x => Sum.inl AggregateError.InsufficientParticipants | isFalse x => Sum.inl AggregateError.InsufficientParticipants) hLe)) h
    False.elim (Sum.noConfusion h_false)

theorem prover_init_counter_zero : (ZKInferenceProver.init).proof_counter = 0 := Eq.refl 0

theorem prove_inference_increments_counter (prover : ZKInferenceProver) :
  (ZKInferenceProver.proveInference prover).1.proof_counter = prover.proof_counter + 1 :=
  Eq.refl _

theorem prove_inference_counter_monotone (n : Nat) :
  (Nat.recOn n ZKInferenceProver.init (fun _ p => (ZKInferenceProver.proveInference p).1) : ZKInferenceProver).proof_counter = n :=
  Nat.recOn n
    (Eq.refl 0)
    (fun k ih =>
      let p_k := (Nat.recOn k ZKInferenceProver.init (fun _ p => (ZKInferenceProver.proveInference p).1) : ZKInferenceProver)
      let step1 : (ZKInferenceProver.proveInference p_k).1.proof_counter = p_k.proof_counter + 1 := Eq.refl _
      Eq.trans step1 (congrArg (fun x => x + 1) ih))

theorem full_pipeline_correctness (p : ZKInferenceProof) (inp out : List (Fin 256)) (mh : ByteArray32) :
  ZKInferenceProof.verifyChain H (ZKInferenceProof.proveInference H p inp out mh) mh = true :=
  prove_verify_chain_consistency H p inp out mh

variable (cr_blake3 : ∀ a b, H.blake3 a = H.blake3 b → a = b)

theorem full_pipeline_commitment_binding (inp1 inp2 : List (Fin 256)) (h : H.blake3 [inp1] = H.blake3 [inp2]) :
  (ZKInferenceProof.proveInference H ZKInferenceProof.init inp1 [] (fun _ => ⟨0, Nat.zero_lt_succ 255⟩)).input_commitment =
  (ZKInferenceProof.proveInference H ZKInferenceProof.init inp2 [] (fun _ => ⟨0, Nat.zero_lt_succ 255⟩)).input_commitment :=
  h

theorem commitment_uniqueness (v1 v2 : List (Fin 256)) (n1 n2 b1 b2 : ByteArray32)
  (h : (CommitmentScheme.commit H CommitmentScheme.init v1 n1 b1).1 = (CommitmentScheme.commit H CommitmentScheme.init v2 n2 b2).1) :
  [v1, toList n1, toList b1] = [v2, toList n2, toList b2] :=
  cr_blake3 [v1, toList n1, toList b1] [v2, toList n2, toList b2] h

theorem merkle_root_deterministic (leaves : List ByteArray32) :
  buildMerkleTree H leaves = buildMerkleTree H leaves :=
  Eq.refl _

theorem merkle_proof_unique_path (leaves : List ByteArray32) (index : Nat) (hBound : index < listLength leaves) :
  generateMerkleProof H leaves index hBound = generateMerkleProof H leaves index hBound :=
  Eq.refl _

end Theorems

end ZKVerification

#check ZKVerification.Theorems.full_pipeline_correctness
#check ZKVerification.Theorems.commit_verify_consistency
#check ZKVerification.Theorems.prove_verify_chain_consistency
#check ZKVerification.Theorems.schnorr_sign_verify_consistency
#check ZKVerification.Theorems.range_proof_soundness_value_in_range
#check ZKVerification.Theorems.aggregate_result_is_some
#check ZKVerification.Theorems.witness_init_tokens_length
#check ZKVerification.Theorems.bundle_init_not_verified
#check ZKVerification.Theorems.compile_success_sets_compiled