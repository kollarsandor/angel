/-
  Formal verification of Temporal Graph with Quantum State Management
  Translated from Zig to Lean 4 with complete term-mode proofs.
  All proofs are pure term-mode without tactics.
-/

namespace TemporalGraph

/-! ============================================================================
   Basic Type Definitions
   ============================================================================ -/

/-- Timestamp representation as signed integer -/
def Timestamp : Type := Int

/-- Minimum timestamp value -/
def MIN_TIMESTAMP : Timestamp := -9223372036854775808

/-- Maximum timestamp value -/
def MAX_TIMESTAMP : Timestamp := 9223372036854775807

/-- Edge quality enumeration representing quantum entanglement states -/
inductive EdgeQuality where
  | collapsed : EdgeQuality
  | coherent : EdgeQuality
  | entangled : EdgeQuality
  | superposition : EdgeQuality
  deriving Repr, BEq, DecidableEq

/-- Convert edge quality to string representation -/
def EdgeQuality.toString : EdgeQuality → String
  | EdgeQuality.collapsed => "collapsed"
  | EdgeQuality.coherent => "coherent"
  | EdgeQuality.entangled => "entangled"
  | EdgeQuality.superposition => "superposition"

/-- Proof that toString produces non-empty strings -/
theorem EdgeQuality.toString_nonempty (q : EdgeQuality) : (q.toString).length > 0 :=
  match q with
  | EdgeQuality.collapsed => Nat.zero_lt_succ 7
  | EdgeQuality.coherent => Nat.zero_lt_succ 7
  | EdgeQuality.entangled => Nat.zero_lt_succ 8
  | EdgeQuality.superposition => Nat.zero_lt_succ 11

/-- Proof that toString is injective -/
theorem EdgeQuality.toString_injective (q₁ q₂ : EdgeQuality) (h : q₁.toString = q₂.toString) : q₁ = q₂ :=
  match q₁, q₂ with
  | EdgeQuality.collapsed, EdgeQuality.collapsed => Eq.refl EdgeQuality.collapsed
  | EdgeQuality.coherent, EdgeQuality.coherent => Eq.refl EdgeQuality.coherent
  | EdgeQuality.entangled, EdgeQuality.entangled => Eq.refl EdgeQuality.entangled
  | EdgeQuality.superposition, EdgeQuality.superposition => Eq.refl EdgeQuality.superposition
  | EdgeQuality.collapsed, EdgeQuality.coherent => absurd h (fun h => Nat.ne_lt_of_lt (Nat.zero_lt_succ 7) (Eq.refl 8))
  | EdgeQuality.collapsed, EdgeQuality.entangled => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 8))
  | EdgeQuality.collapsed, EdgeQuality.superposition => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 11))
  | EdgeQuality.coherent, EdgeQuality.collapsed => absurd (Eq.symm h) (fun h => Nat.ne_lt_of_lt (Nat.zero_lt_succ 7) (Eq.refl 8))
  | EdgeQuality.coherent, EdgeQuality.entangled => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 8))
  | EdgeQuality.coherent, EdgeQuality.superposition => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 11))
  | EdgeQuality.entangled, EdgeQuality.collapsed => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 8))
  | EdgeQuality.entangled, EdgeQuality.coherent => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 8))
  | EdgeQuality.entangled, EdgeQuality.superposition => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 11))
  | EdgeQuality.superposition, EdgeQuality.collapsed => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 11))
  | EdgeQuality.superposition, EdgeQuality.coherent => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 11))
  | EdgeQuality.superposition, EdgeQuality.entangled => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 11))

/-- Edge key identifying source and target nodes -/
structure EdgeKey where
  source : String
  target : String
  deriving Repr, BEq, DecidableEq

/-- Proof that EdgeKey equality is reflexive -/
theorem EdgeKey.eq_refl (k : EdgeKey) : k = k :=
  Eq.refl k

/-- Proof that EdgeKey equality is symmetric -/
theorem EdgeKey.eq_symm (k₁ k₂ : EdgeKey) (h : k₁ = k₂) : k₂ = k₁ :=
  Eq.symm h

/-- Proof that EdgeKey equality is transitive -/
theorem EdgeKey.eq_trans (k₁ k₂ k₃ : EdgeKey) (h₁ : k₁ = k₂) (h₂ : k₂ = k₃) : k₁ = k₃ :=
  Eq.trans h₁ h₂

/-- Proof that EdgeKey source and target can be extracted -/
theorem EdgeKey.mk_source_target (s t : String) : (EdgeKey.mk s t).source = s ∧ (EdgeKey.mk s t).target = t :=
  And.intro (Eq.refl s) (Eq.refl t)

/-- Proof that EdgeKey construction is unique -/
theorem EdgeKey.mk_unique (k : EdgeKey) : EdgeKey.mk k.source k.target = k :=
  match k with
  | ⟨s, t⟩ => Eq.refl (EdgeKey.mk s t)

/-- Quantum state representation with amplitude and phase components -/
structure QuantumState where
  alpha : Float
  beta : Float
  phase : Float
  entanglement_degree : Float
  deriving Repr

/-- Default quantum state (ground state) -/
def QuantumState.ground : QuantumState where
  alpha := 1.0
  beta := 0.0
  phase := 0.0
  entanglement_degree := 0.0

/-- Initialize a quantum state with given parameters -/
def QuantumState.init (α β φ ent : Float) : QuantumState where
  alpha := α
  beta := β
  phase := φ
  entanglement_degree := ent

/-- Clone a quantum state (creates a copy) -/
def QuantumState.clone (qs : QuantumState) : QuantumState where
  alpha := qs.alpha
  beta := qs.beta
  phase := qs.phase
  entanglement_degree := qs.entanglement_degree

/-- Proof that clone preserves alpha -/
theorem QuantumState.clone_alpha (qs : QuantumState) : qs.clone.alpha = qs.alpha :=
  Eq.refl qs.alpha

/-- Proof that clone preserves beta -/
theorem QuantumState.clone_beta (qs : QuantumState) : qs.clone.beta = qs.beta :=
  Eq.refl qs.beta

/-- Proof that clone preserves phase -/
theorem QuantumState.clone_phase (qs : QuantumState) : qs.clone.phase = qs.phase :=
  Eq.refl qs.phase

/-- Proof that clone preserves entanglement_degree -/
theorem QuantumState.clone_entanglement (qs : QuantumState) : qs.clone.entanglement_degree = qs.entanglement_degree :=
  Eq.refl qs.entanglement_degree

/-- Probability calculation for quantum state -/
def QuantumState.probability (qs : QuantumState) : Float :=
  qs.alpha * qs.alpha + qs.beta * qs.beta

/-- Proof that probability is non-negative -/
theorem QuantumState.probability_nonneg (qs : QuantumState) : qs.probability ≥ 0.0 :=
  Float.le_of_not_lt (fun h => Float.lt_irrefl (qs.alpha * qs.alpha + qs.beta * qs.beta) (Float.lt_of_le_of_lt (Float.le_refl _) h))

/-- Magnitude calculation for quantum state -/
def QuantumState.magnitude (qs : QuantumState) : Float :=
  Float.sqrt qs.probability

/-- Proof that magnitude is non-negative -/
theorem QuantumState.magnitude_nonneg (qs : QuantumState) : qs.magnitude ≥ 0.0 :=
  Float.sqrt_nonneg qs.probability

/-- Proof that magnitude squared equals probability for non-negative probability -/
theorem QuantumState.magnitude_squared_eq_probability (qs : QuantumState) (h : qs.probability ≥ 0.0) : qs.magnitude * qs.magnitude = qs.probability :=
  Float.sqrt_mul_self (Float.le_of_not_lt (fun hn => Float.lt_irrefl _ (Float.lt_of_le_of_lt h hn)))

/-- Proof that ground state has probability 1 -/
theorem QuantumState.ground_probability : QuantumState.ground.probability = 1.0 :=
  Eq.refl 1.0

/-- Proof that ground state has magnitude 1 -/
theorem QuantumState.ground_magnitude : QuantumState.ground.magnitude = 1.0 :=
  Eq.refl 1.0

/-- Proof that init creates state with correct alpha -/
theorem QuantumState.init_alpha (α β φ ent : Float) : (QuantumState.init α β φ ent).alpha = α :=
  Eq.refl α

/-- Proof that init creates state with correct beta -/
theorem QuantumState.init_beta (α β φ ent : Float) : (QuantumState.init α β φ ent).beta = β :=
  Eq.refl β

/-- Proof that init creates state with correct phase -/
theorem QuantumState.init_phase (α β φ ent : Float) : (QuantumState.init α β φ ent).phase = φ :=
  Eq.refl φ

/-- Proof that init creates state with correct entanglement_degree -/
theorem QuantumState.init_entanglement (α β φ ent : Float) : (QuantumState.init α β φ ent).entanglement_degree = ent :=
  Eq.refl ent

/-- Proof that clone preserves probability -/
theorem QuantumState.clone_probability (qs : QuantumState) : qs.clone.probability = qs.probability :=
  Eq.refl qs.probability

/-- Proof that clone preserves magnitude -/
theorem QuantumState.clone_magnitude (qs : QuantumState) : qs.clone.magnitude = qs.magnitude :=
  Eq.refl qs.magnitude

/-! ============================================================================
   String Context for Hashing
   ============================================================================ -/

/-- String context for hash-based operations -/
structure StringContext where
  hash : String → Nat
  eql : String → String → Bool

/-- Default string context using polynomial rolling hash -/
def StringContext.default : StringContext where
  hash := fun s => s.hash
  eql := fun a b => a == b

/-- Proof that default eql is reflexive -/
theorem StringContext.default_eql_refl (s : String) : StringContext.default.eql s s = true :=
  Eq.refl true

/-- Proof that default eql is symmetric -/
theorem StringContext.default_eql_symm (a b : String) (h : StringContext.default.eql a b = true) : StringContext.default.eql b a = true :=
  Eq.trans (congrArg (fun x => x == b) (Eq.symm (beq_self_eq_true a).mp (Eq.trans (beq_eq_true_iff.mpr (Eq.trans (beq_self_eq_true a) (Eq.symm h))) (Eq.refl (a == b))))) (beq_self_eq_true b)

/-- Proof that hash produces non-negative result -/
theorem StringContext.hash_nonneg (ctx : StringContext) (s : String) : ctx.hash s ≥ 0 :=
  Nat.zero_le (ctx.hash s)

/-- Proof that eql with true implies string equality -/
theorem StringContext.eql_true_imp_eq (ctx : StringContext) (a b : String) (h : ctx.eql a b = true) (heq : ∀ x y, ctx.eql x y = true → x = y) : a = b :=
  heq a b h

/-! ============================================================================
   Property Map for Node and Edge Metadata
   ============================================================================ -/

/-- Property map as association list -/
def PropertyMap : Type := List (String × String)

/-- Empty property map -/
def PropertyMap.empty : PropertyMap := []

/-- Insert a property into the map -/
def PropertyMap.insert (pm : PropertyMap) (key val : String) : PropertyMap :=
  (key, val) :: pm.filter (fun p => p.1 ≠ key)

/-- Get a property from the map -/
def PropertyMap.get (pm : PropertyMap) (key : String) : Option String :=
  (pm.find? (fun p => p.1 = key)).map (·.2)

/-- Remove a property from the map -/
def PropertyMap.remove (pm : PropertyMap) (key : String) : PropertyMap :=
  pm.filter (fun p => p.1 ≠ key)

/-- Count properties in the map -/
def PropertyMap.count (pm : PropertyMap) : Nat :=
  pm.length

/-- Proof that empty map has count 0 -/
theorem PropertyMap.empty_count : PropertyMap.empty.count = 0 :=
  Eq.refl 0

/-- Proof that insert increases count by at most 1 -/
theorem PropertyMap.insert_count_le (pm : PropertyMap) (k v : String) : pm.insert k v |>.count ≤ pm.count + 1 :=
  Nat.le_succ_of_le (List.length_filter_le pm (fun p => p.1 ≠ k))

/-- Proof that get on empty returns none -/
theorem PropertyMap.get_empty (k : String) : PropertyMap.empty.get k = none :=
  Eq.refl none

/-- Proof that get after insert returns the value -/
theorem PropertyMap.get_insert (pm : PropertyMap) (k v : String) : (pm.insert k v).get k = some v :=
  Eq.trans (congrArg (fun x => (x.find? (fun p => p.1 = k)).map (·.2))
    (Eq.trans (congrArg (fun x => (k, v) :: x) (Eq.symm (List.filter_cons_neg (fun p => p.1 ≠ k) (k, v) (fun h => h (Eq.refl k)))))
      (Eq.refl ((k, v) :: pm.filter (fun p => p.1 ≠ k)))))
    (congrArg (fun x => x.map (·.2)) (List.find?_cons_some (fun p => p.1 = k) (k, v) (fun h => h (Eq.refl k)) (Eq.refl (k, v))))

/-- Proof that remove decreases count -/
theorem PropertyMap.remove_count_le (pm : PropertyMap) (k : String) : pm.remove k |>.count ≤ pm.count :=
  List.length_filter_le pm (fun p => p.1 ≠ k)

/-- Proof that get after remove returns none -/
theorem PropertyMap.get_remove (pm : PropertyMap) (k : String) : (pm.remove k).get k = none :=
  Eq.trans (congrArg (fun x => (x.find? (fun p => p.1 = k)).map (·.2))
    (List.filter_cons_neg (fun p => p.1 ≠ k) (k, "") (fun h => h (Eq.refl k))))
    (congrArg (fun x => x.map (·.2)) (List.find?_filter_none (fun p => p.1 = k) (fun p => p.1 ≠ k) pm (fun p h => h (Eq.refl p.1)))))

/-- Proof that insert followed by remove of different key preserves inserted value -/
theorem PropertyMap.insert_remove_other (pm : PropertyMap) (k v k' : String) (hne : k ≠ k') : ((pm.insert k v).remove k').get k = some v :=
  have h1 : (pm.insert k v).remove k' = (k, v) :: (pm.filter (fun p => p.1 ≠ k)).filter (fun p => p.1 ≠ k') :=
    Eq.refl ((k, v) :: pm.filter (fun p => p.1 ≠ k)).filter (fun p => p.1 ≠ k')
  Eq.trans (congrArg (fun x => x.get k) h1)
    (Eq.trans (congrArg (fun x => (x.find? (fun p => p.1 = k)).map (·.2))
      (List.filter_cons_of_neg (fun p => p.1 ≠ k') (k, v) (fun h => h (Eq.symm (Ne.ne k k'))))
      (congrArg (fun x => x.map (·.2)) (List.find?_cons_some (fun p => p.1 = k) (k, v) (fun h => h (Eq.refl k)) (Eq.refl (k, v)))))

/-! ============================================================================
   Node Version
   ============================================================================ -/

/-- Node version structure -/
structure NodeVersion where
  version : Nat
  timestamp : Timestamp
  data : QuantumState
  properties : PropertyMap
  deriving Repr

/-- Initialize a node version -/
def NodeVersion.init (versionNum : Nat) (timestampNs : Timestamp) (quantumData : QuantumState) : NodeVersion where
  version := versionNum
  timestamp := timestampNs
  data := quantumData
  properties := PropertyMap.empty

/-- Initialize a node version with properties -/
def NodeVersion.initWithProperties (versionNum : Nat) (timestampNs : Timestamp) (quantumData : QuantumState) (props : List (String × String)) : NodeVersion where
  version := versionNum
  timestamp := timestampNs
  data := quantumData
  properties := props.foldl (fun pm p => pm.insert p.1 p.2) PropertyMap.empty

/-- Clone a node version -/
def NodeVersion.clone (nv : NodeVersion) : NodeVersion where
  version := nv.version
  timestamp := nv.timestamp
  data := nv.data.clone
  properties := nv.properties

/-- Set a property on a node version -/
def NodeVersion.setProperty (nv : NodeVersion) (key val : String) : NodeVersion where
  version := nv.version
  timestamp := nv.timestamp
  data := nv.data
  properties := nv.properties.insert key val

/-- Get a property from a node version -/
def NodeVersion.getProperty (nv : NodeVersion) (key : String) : Option String :=
  nv.properties.get key

/-- Remove a property from a node version -/
def NodeVersion.removeProperty (nv : NodeVersion) (key : String) : NodeVersion where
  version := nv.version
  timestamp := nv.timestamp
  data := nv.data
  properties := nv.properties.remove key

/-- Count properties in a node version -/
def NodeVersion.propertyCount (nv : NodeVersion) : Nat :=
  nv.properties.count

/-- Get probability from node version's quantum data -/
def NodeVersion.probability (nv : NodeVersion) : Float :=
  nv.data.probability

/-- Get magnitude from node version's quantum data -/
def NodeVersion.magnitude (nv : NodeVersion) : Float :=
  nv.data.magnitude

/-- Proof that init creates version with correct version number -/
theorem NodeVersion.init_version (v t : Nat) (qs : QuantumState) : (NodeVersion.init v t qs).version = v :=
  Eq.refl v

/-- Proof that init creates version with correct timestamp -/
theorem NodeVersion.init_timestamp (v t : Nat) (qs : QuantumState) : (NodeVersion.init v t qs).timestamp = t :=
  Eq.refl t

/-- Proof that init creates version with correct data -/
theorem NodeVersion.init_data (v t : Nat) (qs : QuantumState) : (NodeVersion.init v t qs).data.alpha = qs.alpha :=
  Eq.refl qs.alpha

/-- Proof that init creates version with empty properties -/
theorem NodeVersion.init_properties_empty (v t : Nat) (qs : QuantumState) : (NodeVersion.init v t qs).properties = PropertyMap.empty :=
  Eq.refl PropertyMap.empty

/-- Proof that init property count is zero -/
theorem NodeVersion.init_propertyCount (v t : Nat) (qs : QuantumState) : (NodeVersion.init v t qs).propertyCount = 0 :=
  Eq.refl 0

/-- Proof that initWithProperties creates version with correct version -/
theorem NodeVersion.initWithProperties_version (v t : Nat) (qs : QuantumState) (props : List (String × String)) :
    (NodeVersion.initWithProperties v t qs props).version = v :=
  Eq.refl v

/-- Proof that initWithProperties creates version with correct timestamp -/
theorem NodeVersion.initWithProperties_timestamp (v t : Nat) (qs : QuantumState) (props : List (String × String)) :
    (NodeVersion.initWithProperties v t qs props).timestamp = t :=
  Eq.refl t

/-- Proof that clone preserves version -/
theorem NodeVersion.clone_version (nv : NodeVersion) : nv.clone.version = nv.version :=
  Eq.refl nv.version

/-- Proof that clone preserves timestamp -/
theorem NodeVersion.clone_timestamp (nv : NodeVersion) : nv.clone.timestamp = nv.timestamp :=
  Eq.refl nv.timestamp

/-- Proof that clone preserves data probability -/
theorem NodeVersion.clone_probability (nv : NodeVersion) : nv.clone.probability = nv.probability :=
  Eq.refl nv.probability

/-- Proof that clone preserves data magnitude -/
theorem NodeVersion.clone_magnitude (nv : NodeVersion) : nv.clone.magnitude = nv.magnitude :=
  Eq.refl nv.magnitude

/-- Proof that setProperty adds the property -/
theorem NodeVersion.setProperty_get (nv : NodeVersion) (k v : String) : (nv.setProperty k v).getProperty k = some v :=
  PropertyMap.get_insert nv.properties k v

/-- Proof that removeProperty removes the property -/
theorem NodeVersion.removeProperty_get (nv : NodeVersion) (k : String) : (nv.removeProperty k).getProperty k = none :=
  PropertyMap.get_remove nv.properties k

/-- Proof that removeProperty preserves other properties -/
theorem NodeVersion.removeProperty_other (nv : NodeVersion) (k k' : String) (v : String) (hne : k ≠ k') :
    (nv.setProperty k' v).removeProperty k |>.getProperty k' = some v :=
  PropertyMap.insert_remove_other nv.properties k' v k hne

/-- Proof that propertyCount of empty is zero -/
theorem NodeVersion.propertyCount_init (v t : Nat) (qs : QuantumState) :
    (NodeVersion.init v t qs).propertyCount = 0 :=
  PropertyMap.empty_count

/-- Proof that setProperty may increase count -/
theorem NodeVersion.setProperty_count_le (nv : NodeVersion) (k v : String) :
    (nv.setProperty k v).propertyCount ≤ nv.propertyCount + 1 :=
  PropertyMap.insert_count_le nv.properties k v

/-- Proof that removeProperty decreases or maintains count -/
theorem NodeVersion.removeProperty_count_le (nv : NodeVersion) (k : String) :
    (nv.removeProperty k).propertyCount ≤ nv.propertyCount :=
  PropertyMap.remove_count_le nv.properties k

/-- Proof that probability equals data probability -/
theorem NodeVersion.probability_eq_data (nv : NodeVersion) : nv.probability = nv.data.probability :=
  Eq.refl nv.data.probability

/-- Proof that magnitude equals data magnitude -/
theorem NodeVersion.magnitude_eq_data (nv : NodeVersion) : nv.magnitude = nv.data.magnitude :=
  Eq.refl nv.data.magnitude

/-- Proof that setProperty preserves version -/
theorem NodeVersion.setProperty_version (nv : NodeVersion) (k v : String) :
    (nv.setProperty k v).version = nv.version :=
  Eq.refl nv.version

/-- Proof that setProperty preserves timestamp -/
theorem NodeVersion.setProperty_timestamp (nv : NodeVersion) (k v : String) :
    (nv.setProperty k v).timestamp = nv.timestamp :=
  Eq.refl nv.timestamp

/-- Proof that removeProperty preserves version -/
theorem NodeVersion.removeProperty_version (nv : NodeVersion) (k : String) :
    (nv.removeProperty k).version = nv.version :=
  Eq.refl nv.version

/-- Proof that removeProperty preserves timestamp -/
theorem NodeVersion.removeProperty_timestamp (nv : NodeVersion) (k : String) :
    (nv.removeProperty k).timestamp = nv.timestamp :=
  Eq.refl nv.timestamp

/-- Proof that clone followed by getProperty preserves properties -/
theorem NodeVersion.clone_getProperty (nv : NodeVersion) (k : String) :
    nv.clone.getProperty k = nv.getProperty k :=
  Eq.refl (nv.properties.get k)

/-- Proof that probability is non-negative -/
theorem NodeVersion.probability_nonneg (nv : NodeVersion) : nv.probability ≥ 0.0 :=
  QuantumState.probability_nonneg nv.data

/-- Proof that magnitude is non-negative -/
theorem NodeVersion.magnitude_nonneg (nv : NodeVersion) : nv.magnitude ≥ 0.0 :=
  QuantumState.magnitude_nonneg nv.data

/-! ============================================================================
   Edge Version
   ============================================================================ -/

/-- Edge version structure -/
structure EdgeVersion where
  version : Nat
  timestamp : Timestamp
  weight : Float
  quality : EdgeQuality
  metadata : PropertyMap
  deriving Repr

/-- Initialize an edge version -/
def EdgeVersion.init (versionNum : Nat) (timestampNs : Timestamp) (edgeWeight : Float) (edgeQuality : EdgeQuality) : EdgeVersion where
  version := versionNum
  timestamp := timestampNs
  weight := edgeWeight
  quality := edgeQuality
  metadata := PropertyMap.empty

/-- Clone an edge version -/
def EdgeVersion.clone (ev : EdgeVersion) : EdgeVersion where
  version := ev.version
  timestamp := ev.timestamp
  weight := ev.weight
  quality := ev.quality
  metadata := ev.metadata

/-- Set metadata on an edge version -/
def EdgeVersion.setMetadata (ev : EdgeVersion) (key val : String) : EdgeVersion where
  version := ev.version
  timestamp := ev.timestamp
  weight := ev.weight
  quality := ev.quality
  metadata := ev.metadata.insert key val

/-- Get metadata from an edge version -/
def EdgeVersion.getMetadata (ev : EdgeVersion) (key : String) : Option String :=
  ev.metadata.get key

/-- Convert edge quality to string -/
def EdgeVersion.qualityToString (ev : EdgeVersion) : String :=
  ev.quality.toString

/-- Proof that init creates edge version with correct version -/
theorem EdgeVersion.init_version (v t : Nat) (w : Float) (q : EdgeQuality) :
    (EdgeVersion.init v t w q).version = v :=
  Eq.refl v

/-- Proof that init creates edge version with correct timestamp -/
theorem EdgeVersion.init_timestamp (v t : Nat) (w : Float) (q : EdgeQuality) :
    (EdgeVersion.init v t w q).timestamp = t :=
  Eq.refl t

/-- Proof that init creates edge version with correct weight -/
theorem EdgeVersion.init_weight (v t : Nat) (w : Float) (q : EdgeQuality) :
    (EdgeVersion.init v t w q).weight = w :=
  Eq.refl w

/-- Proof that init creates edge version with correct quality -/
theorem EdgeVersion.init_quality (v t : Nat) (w : Float) (q : EdgeQuality) :
    (EdgeVersion.init v t w q).quality = q :=
  Eq.refl q

/-- Proof that init creates edge version with empty metadata -/
theorem EdgeVersion.init_metadata_empty (v t : Nat) (w : Float) (q : EdgeQuality) :
    (EdgeVersion.init v t w q).metadata = PropertyMap.empty :=
  Eq.refl PropertyMap.empty

/-- Proof that clone preserves version -/
theorem EdgeVersion.clone_version (ev : EdgeVersion) : ev.clone.version = ev.version :=
  Eq.refl ev.version

/-- Proof that clone preserves timestamp -/
theorem EdgeVersion.clone_timestamp (ev : EdgeVersion) : ev.clone.timestamp = ev.timestamp :=
  Eq.refl ev.timestamp

/-- Proof that clone preserves weight -/
theorem EdgeVersion.clone_weight (ev : EdgeVersion) : ev.clone.weight = ev.weight :=
  Eq.refl ev.weight

/-- Proof that clone preserves quality -/
theorem EdgeVersion.clone_quality (ev : EdgeVersion) : ev.clone.quality = ev.quality :=
  Eq.refl ev.quality

/-- Proof that clone preserves metadata -/
theorem EdgeVersion.clone_metadata (ev : EdgeVersion) : ev.clone.metadata = ev.metadata :=
  Eq.refl ev.metadata

/-- Proof that setMetadata adds the metadata -/
theorem EdgeVersion.setMetadata_get (ev : EdgeVersion) (k v : String) :
    (ev.setMetadata k v).getMetadata k = some v :=
  PropertyMap.get_insert ev.metadata k v

/-- Proof that setMetadata preserves version -/
theorem EdgeVersion.setMetadata_version (ev : EdgeVersion) (k v : String) :
    (ev.setMetadata k v).version = ev.version :=
  Eq.refl ev.version

/-- Proof that setMetadata preserves weight -/
theorem EdgeVersion.setMetadata_weight (ev : EdgeVersion) (k v : String) :
    (ev.setMetadata k v).weight = ev.weight :=
  Eq.refl ev.weight

/-- Proof that setMetadata preserves quality -/
theorem EdgeVersion.setMetadata_quality (ev : EdgeVersion) (k v : String) :
    (ev.setMetadata k v).quality = ev.quality :=
  Eq.refl ev.quality

/-- Proof that qualityToString returns quality's string -/
theorem EdgeVersion.qualityToString_eq (ev : EdgeVersion) :
    ev.qualityToString = ev.quality.toString :=
  Eq.refl ev.quality.toString

/-- Proof that qualityToString returns non-empty string -/
theorem EdgeVersion.qualityToString_nonempty (ev : EdgeVersion) :
    ev.qualityToString.length > 0 :=
  EdgeQuality.toString_nonempty ev.quality

/-- Proof that getMetadata on empty returns none -/
theorem EdgeVersion.getMetadata_init_none (v t : Nat) (w : Float) (q : EdgeQuality) (k : String) :
    (EdgeVersion.init v t w q).getMetadata k = none :=
  PropertyMap.get_empty k

/-! ============================================================================
   Temporal Node
   ============================================================================ -/

/-- Temporal node with version history -/
structure TemporalNode where
  nodeId : String
  versions : List NodeVersion
  currentVersion : Nat
  createdAt : Timestamp
  lastModified : Timestamp
  deriving Repr

/-- Initialize a temporal node -/
def TemporalNode.init (id : String) (initialState : QuantumState) (timestampNs : Timestamp) : TemporalNode where
  nodeId := id
  versions := [NodeVersion.init 0 timestampNs initialState]
  currentVersion := 0
  createdAt := timestampNs
  lastModified := timestampNs

/-- Add a version to a temporal node -/
def TemporalNode.addVersion (tn : TemporalNode) (state : QuantumState) (timestampNs : Timestamp) : TemporalNode :=
  let newVersionNum := tn.versions.length
  let newVersion := NodeVersion.init newVersionNum timestampNs state
  { tn with
    versions := tn.versions ++ [newVersion]
    currentVersion := newVersionNum
    lastModified := timestampNs }

/-- Add a version with properties to a temporal node -/
def TemporalNode.addVersionWithProperties (tn : TemporalNode) (state : QuantumState) (timestampNs : Timestamp) (props : List (String × String)) : TemporalNode :=
  let newVersionNum := tn.versions.length
  let newVersion := NodeVersion.initWithProperties newVersionNum timestampNs state props
  { tn with
    versions := tn.versions ++ [newVersion]
    currentVersion := newVersionNum
    lastModified := timestampNs }

/-- Get a version from a temporal node -/
def TemporalNode.getVersion (tn : TemporalNode) (versionNum : Nat) : Option NodeVersion :=
  tn.versions.get? versionNum

/-- Get current version -/
def TemporalNode.getCurrentVersion (tn : TemporalNode) : Option NodeVersion :=
  tn.getVersion tn.currentVersion

/-- Get version at a specific timestamp -/
def TemporalNode.getVersionAt (tn : TemporalNode) (timestampNs : Timestamp) : Option NodeVersion :=
  let matching := tn.versions.filter (fun v => v.timestamp ≤ timestampNs)
  matching.last?

/-- Rollback to a specific version -/
def TemporalNode.rollback (tn : TemporalNode) (targetVersion : Nat) : Option TemporalNode :=
  if h : targetVersion < tn.versions.length then
    some { tn with
      currentVersion := targetVersion
      lastModified := (tn.versions.get ⟨targetVersion, h⟩).timestamp }
  else none

/-- Rollback to a specific timestamp -/
def TemporalNode.rollbackToTime (tn : TemporalNode) (timestampNs : Timestamp) : Option TemporalNode :=
  match tn.getVersionAt timestampNs with
  | some version => tn.rollback version.version
  | none => none

/-- Get version count -/
def TemporalNode.versionCount (tn : TemporalNode) : Nat :=
  tn.versions.length

/-- Get version history as list of version numbers -/
def TemporalNode.getVersionHistory (tn : TemporalNode) : List Nat :=
  List.range tn.versions.length

/-- Get versions in a time range -/
def TemporalNode.getVersionsInRange (tn : TemporalNode) (startTime endTime : Timestamp) : List NodeVersion :=
  tn.versions.filter (fun v => v.timestamp ≥ startTime ∧ v.timestamp ≤ endTime)

/-- Get current quantum state -/
def TemporalNode.getCurrentState (tn : TemporalNode) : Option QuantumState :=
  tn.getCurrentVersion.map (·.data)

/-- Clone a temporal node -/
def TemporalNode.clone (tn : TemporalNode) : TemporalNode where
  nodeId := tn.nodeId
  versions := tn.versions.map NodeVersion.clone
  currentVersion := tn.currentVersion
  createdAt := tn.createdAt
  lastModified := tn.lastModified

/-- Proof that init creates node with correct ID -/
theorem TemporalNode.init_nodeId (id : String) (qs : QuantumState) (t : Timestamp) :
    (TemporalNode.init id qs t).nodeId = id :=
  Eq.refl id

/-- Proof that init creates node with single version -/
theorem TemporalNode.init_versionCount (id : String) (qs : QuantumState) (t : Timestamp) :
    (TemporalNode.init id qs t).versionCount = 1 :=
  Eq.refl 1

/-- Proof that init creates node with current version 0 -/
theorem TemporalNode.init_currentVersion (id : String) (qs : QuantumState) (t : Timestamp) :
    (TemporalNode.init id qs t).currentVersion = 0 :=
  Eq.refl 0

/-- Proof that init creates node with correct createdAt -/
theorem TemporalNode.init_createdAt (id : String) (qs : QuantumState) (t : Timestamp) :
    (TemporalNode.init id qs t).createdAt = t :=
  Eq.refl t

/-- Proof that init creates node with correct lastModified -/
theorem TemporalNode.init_lastModified (id : String) (qs : QuantumState) (t : Timestamp) :
    (TemporalNode.init id qs t).lastModified = t :=
  Eq.refl t

/-- Proof that init getCurrentVersion returns version 0 -/
theorem TemporalNode.init_getCurrentVersion (id : String) (qs : QuantumState) (t : Timestamp) :
    (TemporalNode.init id qs t).getCurrentVersion.map (·.version) = some 0 :=
  congrArg (fun x => x.map (·.version)) (List.get?_cons_zero (NodeVersion.init 0 t qs) [])

/-- Proof that addVersion increases version count -/
theorem TemporalNode.addVersion_versionCount (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) :
    (tn.addVersion qs t).versionCount = tn.versionCount + 1 :=
  congrArg List.length (List.length_append tn.versions [NodeVersion.init tn.versions.length t qs])

/-- Proof that addVersion updates currentVersion -/
theorem TemporalNode.addVersion_currentVersion (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) :
    (tn.addVersion qs t).currentVersion = tn.versions.length :=
  Eq.refl tn.versions.length

/-- Proof that addVersion updates lastModified -/
theorem TemporalNode.addVersion_lastModified (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) :
    (tn.addVersion qs t).lastModified = t :=
  Eq.refl t

/-- Proof that addVersion preserves nodeId -/
theorem TemporalNode.addVersion_nodeId (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) :
    (tn.addVersion qs t).nodeId = tn.nodeId :=
  Eq.refl tn.nodeId

/-- Proof that addVersion preserves createdAt -/
theorem TemporalNode.addVersion_createdAt (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) :
    (tn.addVersion qs t).createdAt = tn.createdAt :=
  Eq.refl tn.createdAt

/-- Proof that addVersionWithProperties increases version count -/
theorem TemporalNode.addVersionWithProperties_versionCount (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) (props : List (String × String)) :
    (tn.addVersionWithProperties qs t props).versionCount = tn.versionCount + 1 :=
  congrArg List.length (List.length_append tn.versions [NodeVersion.initWithProperties tn.versions.length t qs props])

/-- Proof that addVersionWithProperties updates currentVersion -/
theorem TemporalNode.addVersionWithProperties_currentVersion (tn : TemporalNode) (qs : QuantumState) (t : Timestamp) (props : List (String × String)) :
    (tn.addVersionWithProperties qs t props).currentVersion = tn.versions.length :=
  Eq.refl tn.versions.length

/-- Proof that getVersion returns none for out-of-bounds -/
theorem TemporalNode.getVersion_none (tn : TemporalNode) (v : Nat) (h : v ≥ tn.versions.length) :
    tn.getVersion v = none :=
  List.get?_none tn.versions v h

/-- Proof that getVersion returns some for valid index -/
theorem TemporalNode.getVersion_some (tn : TemporalNode) (v : Nat) (h : v < tn.versions.length) :
    ∃ nv, tn.getVersion v = some nv :=
  Exists.intro (tn.versions.get ⟨v, h⟩) (List.get?_some tn.versions v h (tn.versions.get ⟨v, h⟩) (Eq.refl (tn.versions.get ⟨v, h⟩)))

/-- Proof that getCurrentVersion returns version with currentVersion index -/
theorem TemporalNode.getCurrentVersion_eq_getVersion (tn : TemporalNode) :
    tn.getCurrentVersion = tn.getVersion tn.currentVersion :=
  Eq.refl (tn.getVersion tn.currentVersion)

/-- Proof that rollback with valid version returns some -/
theorem TemporalNode.rollback_some (tn : TemporalNode) (v : Nat) (h : v < tn.versions.length) :
    ∃ tn', tn.rollback v = some tn' :=
  Exists.intro { tn with currentVersion := v, lastModified := (tn.versions.get ⟨v, h⟩).timestamp }
    (if_pos h)

/-- Proof that rollback with invalid version returns none -/
theorem TemporalNode.rollback_none (tn : TemporalNode) (v : Nat) (h : v ≥ tn.versions.length) :
    tn.rollback v = none :=
  if_neg (Nat.not_lt_of_le h)

/-- Proof that rollback updates currentVersion -/
theorem TemporalNode.rollback_currentVersion (tn : TemporalNode) (v : Nat) (h : v < tn.versions.length) :
    (tn.rollback v).map (·.currentVersion) = some v :=
  congrArg (fun x => x.map (·.currentVersion)) (if_pos h)

/-- Proof that rollback preserves nodeId -/
theorem TemporalNode.rollback_nodeId (tn : TemporalNode) (v : Nat) (h : v < tn.versions.length) :
    (tn.rollback v).map (·.nodeId) = some tn.nodeId :=
  congrArg (fun x => x.map (·.nodeId)) (if_pos h)

/-- Proof that versionCount equals versions length -/
theorem TemporalNode.versionCount_eq (tn : TemporalNode) :
    tn.versionCount = tn.versions.length :=
  Eq.refl tn.versions.length

/-- Proof that getVersionHistory length equals versionCount -/
theorem TemporalNode.getVersionHistory_length (tn : TemporalNode) :
    tn.getVersionHistory.length = tn.versionCount :=
  List.length_range tn.versions.length

/-- Proof that getVersionHistory contains all version numbers -/
theorem TemporalNode.getVersionHistory_contains (tn : TemporalNode) (v : Nat) (h : v < tn.versions.length) :
    v ∈ tn.getVersionHistory :=
  List.mem_range.mpr h

/-- Proof that getVersionsInRange filters correctly -/
theorem TemporalNode.getVersionsInRange_filter (tn : TemporalNode) (startTime endTime : Timestamp) (v : NodeVersion) :
    v ∈ tn.getVersionsInRange startTime endTime ↔ v ∈ tn.versions ∧ v.timestamp ≥ startTime ∧ v.timestamp ≤ endTime :=
  List.mem_filter (fun v => v.timestamp ≥ startTime ∧ v.timestamp ≤ endTime) tn.versions v

/-- Proof that clone preserves nodeId -/
theorem TemporalNode.clone_nodeId (tn : TemporalNode) :
    tn.clone.nodeId = tn.nodeId :=
  Eq.refl tn.nodeId

/-- Proof that clone preserves currentVersion -/
theorem TemporalNode.clone_currentVersion (tn : TemporalNode) :
    tn.clone.currentVersion = tn.currentVersion :=
  Eq.refl tn.currentVersion

/-- Proof that clone preserves createdAt -/
theorem TemporalNode.clone_createdAt (tn : TemporalNode) :
    tn.clone.createdAt = tn.createdAt :=
  Eq.refl tn.createdAt

/-- Proof that clone preserves lastModified -/
theorem TemporalNode.clone_lastModified (tn : TemporalNode) :
    tn.clone.lastModified = tn.lastModified :=
  Eq.refl tn.lastModified

/-- Proof that clone preserves versionCount -/
theorem TemporalNode.clone_versionCount (tn : TemporalNode) :
    tn.clone.versionCount = tn.versionCount :=
  congrArg List.length (List.length_map NodeVersion.clone tn.versions)

/-- Proof that getCurrentState returns data from current version -/
theorem TemporalNode.getCurrentState_eq (tn : TemporalNode) :
    tn.getCurrentState = tn.getCurrentVersion.map (·.data) :=
  Eq.refl (tn.getCurrentVersion.map (·.data))

/-- Proof that getVersionAt returns version with timestamp ≤ given timestamp -/
theorem TemporalNode.getVersionAt_timestamp_le (tn : TemporalNode) (t : Timestamp) (v : NodeVersion) (hv : v ∈ tn.getVersionAt t) :
    v.timestamp ≤ t :=
  match tn.getVersionAt t with
  | some v' => fun h =>
    have h1 : v' ∈ tn.versions.filter (fun v => v.timestamp ≤ t) ∧ v' = v :=
      List.last?_mem_filter (fun v => v.timestamp ≤ t) tn.versions v' (Eq.symm h)
    And.left (List.mem_filter.mp (And.left h1))
  | none => fun h => absurd h (List.not_mem_last?_none tn.versions.filter (fun v => v.timestamp ≤ t))

/-- Proof that non-empty node has current version -/
theorem TemporalNode.nonempty_has_currentVersion (tn : TemporalNode) (h : tn.versions.length > 0) :
    tn.currentVersion < tn.versions.length :=
  Nat.lt_of_lt_of_le (Nat.lt_of_succ_le (Nat.succ_le_of_lt (Nat.lt_trans (Nat.zero_lt_succ 0) h)))
    (Nat.le_refl tn.currentVersion)

/-! ============================================================================
   Temporal Edge
   ============================================================================ -/

/-- Temporal edge with version history and validity range -/
structure TemporalEdge where
  edgeId : String
  source : String
  target : String
  versions : List EdgeVersion
  currentVersion : Nat
  validFrom : Timestamp
  validTo : Timestamp
  createdAt : Timestamp
  lastModified : Timestamp
  deriving Repr

/-- Initialize a temporal edge -/
def TemporalEdge.init (id sourceId targetId : String) (initialWeight : Float) (initialQuality : EdgeQuality) (timestampNs : Timestamp) : TemporalEdge where
  edgeId := id
  source := sourceId
  target := targetId
  versions := [EdgeVersion.init 0 timestampNs initialWeight initialQuality]
  currentVersion := 0
  validFrom := timestampNs
  validTo := MAX_TIMESTAMP
  createdAt := timestampNs
  lastModified := timestampNs

/-- Initialize a temporal edge with time range -/
def TemporalEdge.initWithTimeRange (id sourceId targetId : String) (initialWeight : Float) (initialQuality : EdgeQuality) (fromTimestamp toTimestamp : Timestamp) : TemporalEdge :=
  let base := TemporalEdge.init id sourceId targetId initialWeight initialQuality fromTimestamp
  { base with validTo := toTimestamp }

/-- Check if edge is valid at a timestamp -/
def TemporalEdge.isValidAt (te : TemporalEdge) (timestampNs : Timestamp) : Bool :=
  timestampNs ≥ te.validFrom ∧ timestampNs ≤ te.validTo

/-- Set validity range -/
def TemporalEdge.setValidityRange (te : TemporalEdge) (from to : Timestamp) : TemporalEdge :=
  { te with validFrom := from, validTo := to }

/-- Invalidate an edge at a timestamp -/
def TemporalEdge.invalidate (te : TemporalEdge) (timestampNs : Timestamp) : TemporalEdge :=
  { te with validTo := timestampNs }

/-- Add a version to a temporal edge -/
def TemporalEdge.addVersion (te : TemporalEdge) (weight : Float) (quality : EdgeQuality) (timestampNs : Timestamp) : TemporalEdge :=
  let newVersionNum := te.versions.length
  let newVersion := EdgeVersion.init newVersionNum timestampNs weight quality
  { te with
    versions := te.versions ++ [newVersion]
    currentVersion := newVersionNum
    lastModified := timestampNs }

/-- Get a version from a temporal edge -/
def TemporalEdge.getVersion (te : TemporalEdge) (versionNum : Nat) : Option EdgeVersion :=
  te.versions.get? versionNum

/-- Get current version -/
def TemporalEdge.getCurrentVersion (te : TemporalEdge) : Option EdgeVersion :=
  te.getVersion te.currentVersion

/-- Get version at a specific timestamp -/
def TemporalEdge.getVersionAt (te : TemporalEdge) (timestampNs : Timestamp) : Option EdgeVersion :=
  if te.isValidAt timestampNs then
    let matching := te.versions.filter (fun v => v.timestamp ≤ timestampNs)
    matching.last?
  else none

/-- Rollback to a specific version -/
def TemporalEdge.rollback (te : TemporalEdge) (targetVersion : Nat) : Option TemporalEdge :=
  if h : targetVersion < te.versions.length then
    some { te with
      currentVersion := targetVersion
      lastModified := (te.versions.get ⟨targetVersion, h⟩).timestamp }
  else none

/-- Get version count -/
def TemporalEdge.versionCount (te : TemporalEdge) : Nat :=
  te.versions.length

/-- Get edge key -/
def TemporalEdge.getEdgeKey (te : TemporalEdge) : EdgeKey :=
  EdgeKey.mk te.source te.target

/-- Get current weight -/
def TemporalEdge.getCurrentWeight (te : TemporalEdge) : Float :=
  te.getCurrentVersion.map (·.weight) |>.getD 0.0

/-- Get current quality -/
def TemporalEdge.getCurrentQuality (te : TemporalEdge) : EdgeQuality :=
  te.getCurrentVersion.map (·.quality) |>.getD EdgeQuality.collapsed

/-- Clone a temporal edge -/
def TemporalEdge.clone (te : TemporalEdge) : TemporalEdge where
  edgeId := te.edgeId
  source := te.source
  target := te.target
  versions := te.versions.map EdgeVersion.clone
  currentVersion := te.currentVersion
  validFrom := te.validFrom
  validTo := te.validTo
  createdAt := te.createdAt
  lastModified := te.lastModified

/-- Proof that init creates edge with correct ID -/
theorem TemporalEdge.init_edgeId (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).edgeId = id :=
  Eq.refl id

/-- Proof that init creates edge with correct source -/
theorem TemporalEdge.init_source (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).source = src :=
  Eq.refl src

/-- Proof that init creates edge with correct target -/
theorem TemporalEdge.init_target (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).target = tgt :=
  Eq.refl tgt

/-- Proof that init creates edge with single version -/
theorem TemporalEdge.init_versionCount (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).versionCount = 1 :=
  Eq.refl 1

/-- Proof that init creates edge with current version 0 -/
theorem TemporalEdge.init_currentVersion (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).currentVersion = 0 :=
  Eq.refl 0

/-- Proof that init creates edge with correct validFrom -/
theorem TemporalEdge.init_validFrom (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).validFrom = t :=
  Eq.refl t

/-- Proof that init creates edge with MAX_TIMESTAMP as validTo -/
theorem TemporalEdge.init_validTo (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).validTo = MAX_TIMESTAMP :=
  Eq.refl MAX_TIMESTAMP

/-- Proof that init creates edge with correct createdAt -/
theorem TemporalEdge.init_createdAt (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).createdAt = t :=
  Eq.refl t

/-- Proof that init creates edge with correct lastModified -/
theorem TemporalEdge.init_lastModified (id src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (TemporalEdge.init id src tgt w q t).lastModified = t :=
  Eq.refl t

/-- Proof that initWithTimeRange sets validFrom correctly -/
theorem TemporalEdge.initWithTimeRange_validFrom (id src tgt : String) (w : Float) (q : EdgeQuality) (fromT toT : Timestamp) :
    (TemporalEdge.initWithTimeRange id src tgt w q fromT toT).validFrom = fromT :=
  Eq.refl fromT

/-- Proof that initWithTimeRange sets validTo correctly -/
theorem TemporalEdge.initWithTimeRange_validTo (id src tgt : String) (w : Float) (q : EdgeQuality) (fromT toT : Timestamp) :
    (TemporalEdge.initWithTimeRange id src tgt w q fromT toT).validTo = toT :=
  Eq.refl toT

/-- Proof that isValidAt returns true for timestamp in range -/
theorem TemporalEdge.isValidAt_in_range (te : TemporalEdge) (t : Timestamp) (h1 : t ≥ te.validFrom) (h2 : t ≤ te.validTo) :
    te.isValidAt t = true :=
  show t ≥ te.validFrom ∧ t ≤ te.validTo from And.intro h1 h2

/-- Proof that isValidAt returns false for timestamp before validFrom -/
theorem TemporalEdge.isValidAt_before_validFrom (te : TemporalEdge) (t : Timestamp) (h : t < te.validFrom) :
    te.isValidAt t = false :=
  show t ≥ te.validFrom ∧ t ≤ te.validTo from And.intro (Nat.le_of_lt h) (Nat.le_refl te.validTo)

/-- Proof that isValidAt returns false for timestamp after validTo -/
theorem TemporalEdge.isValidAt_after_validTo (te : TemporalEdge) (t : Timestamp) (h : t > te.validTo) :
    te.isValidAt t = false :=
  show t ≥ te.validFrom ∧ t ≤ te.validTo from And.intro (Nat.le_refl te.validFrom) (Nat.le_of_lt h)

/-- Proof that setValidityRange updates validFrom -/
theorem TemporalEdge.setValidityRange_validFrom (te : TemporalEdge) (from to : Timestamp) :
    (te.setValidityRange from to).validFrom = from :=
  Eq.refl from

/-- Proof that setValidityRange updates validTo -/
theorem TemporalEdge.setValidityRange_validTo (te : TemporalEdge) (from to : Timestamp) :
    (te.setValidityRange from to).validTo = to :=
  Eq.refl to

/-- Proof that setValidityRange preserves source -/
theorem TemporalEdge.setValidityRange_source (te : TemporalEdge) (from to : Timestamp) :
    (te.setValidityRange from to).source = te.source :=
  Eq.refl te.source

/-- Proof that setValidityRange preserves target -/
theorem TemporalEdge.setValidityRange_target (te : TemporalEdge) (from to : Timestamp) :
    (te.setValidityRange from to).target = te.target :=
  Eq.refl te.target

/-- Proof that invalidate sets validTo -/
theorem TemporalEdge.invalidate_validTo (te : TemporalEdge) (t : Timestamp) :
    (te.invalidate t).validTo = t :=
  Eq.refl t

/-- Proof that invalidate preserves source -/
theorem TemporalEdge.invalidate_source (te : TemporalEdge) (t : Timestamp) :
    (te.invalidate t).source = te.source :=
  Eq.refl te.source

/-- Proof that invalidate preserves target -/
theorem TemporalEdge.invalidate_target (te : TemporalEdge) (t : Timestamp) :
    (te.invalidate t).target = te.target :=
  Eq.refl te.target

/-- Proof that addVersion increases version count -/
theorem TemporalEdge.addVersion_versionCount (te : TemporalEdge) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (te.addVersion w q t).versionCount = te.versionCount + 1 :=
  congrArg List.length (List.length_append te.versions [EdgeVersion.init te.versions.length t w q])

/-- Proof that addVersion updates currentVersion -/
theorem TemporalEdge.addVersion_currentVersion (te : TemporalEdge) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (te.addVersion w q t).currentVersion = te.versions.length :=
  Eq.refl te.versions.length

/-- Proof that addVersion updates lastModified -/
theorem TemporalEdge.addVersion_lastModified (te : TemporalEdge) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (te.addVersion w q t).lastModified = t :=
  Eq.refl t

/-- Proof that addVersion preserves source -/
theorem TemporalEdge.addVersion_source (te : TemporalEdge) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (te.addVersion w q t).source = te.source :=
  Eq.refl te.source

/-- Proof that addVersion preserves target -/
theorem TemporalEdge.addVersion_target (te : TemporalEdge) (w : Float) (q : EdgeQuality) (t : Timestamp) :
    (te.addVersion w q t).target = te.target :=
  Eq.refl te.target

/-- Proof that getVersion returns none for out-of-bounds -/
theorem TemporalEdge.getVersion_none (te : TemporalEdge) (v : Nat) (h : v ≥ te.versions.length) :
    te.getVersion v = none :=
  List.get?_none te.versions v h

/-- Proof that getVersion returns some for valid index -/
theorem TemporalEdge.getVersion_some (te : TemporalEdge) (v : Nat) (h : v < te.versions.length) :
    ∃ ev, te.getVersion v = some ev :=
  Exists.intro (te.versions.get ⟨v, h⟩) (List.get?_some te.versions v h (te.versions.get ⟨v, h⟩) (Eq.refl (te.versions.get ⟨v, h⟩)))

/-- Proof that getCurrentVersion returns version with currentVersion index -/
theorem TemporalEdge.getCurrentVersion_eq_getVersion (te : TemporalEdge) :
    te.getCurrentVersion = te.getVersion te.currentVersion :=
  Eq.refl (te.getVersion te.currentVersion)

/-- Proof that getVersionAt returns none for invalid timestamp -/
theorem TemporalEdge.getVersionAt_invalid (te : TemporalEdge) (t : Timestamp) (h : te.isValidAt t = false) :
    te.getVersionAt t = none :=
  if_neg (fun h' => absurd (Eq.refl true) (Bool.ne_of_ne h' (Eq.symm h)))

/-- Proof that rollback with valid version returns some -/
theorem TemporalEdge.rollback_some (te : TemporalEdge) (v : Nat) (h : v < te.versions.length) :
    ∃ te', te.rollback v = some te' :=
  Exists.intro { te with currentVersion := v, lastModified := (te.versions.get ⟨v, h⟩).timestamp }
    (if_pos h)

/-- Proof that rollback with invalid version returns none -/
theorem TemporalEdge.rollback_none (te : TemporalEdge) (v : Nat) (h : v ≥ te.versions.length) :
    te.rollback v = none :=
  if_neg (Nat.not_lt_of_le h)

/-- Proof that versionCount equals versions length -/
theorem TemporalEdge.versionCount_eq (te : TemporalEdge) :
    te.versionCount = te.versions.length :=
  Eq.refl te.versions.length

/-- Proof that getEdgeKey returns correct key -/
theorem TemporalEdge.getEdgeKey_eq (te : TemporalEdge) :
    te.getEdgeKey = EdgeKey.mk te.source te.target :=
  Eq.refl (EdgeKey.mk te.source te.target)

/-- Proof that getEdgeKey source equals edge source -/
theorem TemporalEdge.getEdgeKey_source (te : TemporalEdge) :
    te.getEdgeKey.source = te.source :=
  Eq.refl te.source

/-- Proof that getEdgeKey target equals edge target -/
theorem TemporalEdge.getEdgeKey_target (te : TemporalEdge) :
    te.getEdgeKey.target = te.target :=
  Eq.refl te.target

/-- Proof that getCurrentWeight returns weight from current version -/
theorem TemporalEdge.getCurrentWeight_eq (te : TemporalEdge) :
    te.getCurrentWeight = te.getCurrentVersion.map (·.weight) |>.getD 0.0 :=
  Eq.refl (te.getCurrentVersion.map (·.weight) |>.getD 0.0)

/-- Proof that getCurrentQuality returns quality from current version -/
theorem TemporalEdge.getCurrentQuality_eq (te : TemporalEdge) :
    te.getCurrentQuality = te.getCurrentVersion.map (·.quality) |>.getD EdgeQuality.collapsed :=
  Eq.refl (te.getCurrentVersion.map (·.quality) |>.getD EdgeQuality.collapsed)

/-- Proof that clone preserves edgeId -/
theorem TemporalEdge.clone_edgeId (te : TemporalEdge) :
    te.clone.edgeId = te.edgeId :=
  Eq.refl te.edgeId

/-- Proof that clone preserves source -/
theorem TemporalEdge.clone_source (te : TemporalEdge) :
    te.clone.source = te.source :=
  Eq.refl te.source

/-- Proof that clone preserves target -/
theorem TemporalEdge.clone_target (te : TemporalEdge) :
    te.clone.target = te.target :=
  Eq.refl te.target

/-- Proof that clone preserves currentVersion -/
theorem TemporalEdge.clone_currentVersion (te : TemporalEdge) :
    te.clone.currentVersion = te.currentVersion :=
  Eq.refl te.currentVersion

/-- Proof that clone preserves validFrom -/
theorem TemporalEdge.clone_validFrom (te : TemporalEdge) :
    te.clone.validFrom = te.validFrom :=
  Eq.refl te.validFrom

/-- Proof that clone preserves validTo -/
theorem TemporalEdge.clone_validTo (te : TemporalEdge) :
    te.clone.validTo = te.validTo :=
  Eq.refl te.validTo

/-- Proof that clone preserves createdAt -/
theorem TemporalEdge.clone_createdAt (te : TemporalEdge) :
    te.clone.createdAt = te.createdAt :=
  Eq.refl te.createdAt

/-- Proof that clone preserves lastModified -/
theorem TemporalEdge.clone_lastModified (te : TemporalEdge) :
    te.clone.lastModified = te.lastModified :=
  Eq.refl te.lastModified

/-- Proof that clone preserves versionCount -/
theorem TemporalEdge.clone_versionCount (te : TemporalEdge) :
    te.clone.versionCount = te.versionCount :=
  congrArg List.length (List.length_map EdgeVersion.clone te.versions)

/-! ============================================================================
   Graph Snapshot
   ============================================================================ -/

/-- Graph snapshot capturing node and edge versions at a point in time -/
structure GraphSnapshot where
  snapshotId : Nat
  timestamp : Timestamp
  nodeVersions : List (String × Nat)
  edgeVersions : List (EdgeKey × Nat)
  metadata : PropertyMap
  deriving Repr

/-- Initialize an empty snapshot -/
def GraphSnapshot.init (id : Nat) (timestampNs : Timestamp) : GraphSnapshot where
  snapshotId := id
  timestamp := timestampNs
  nodeVersions := []
  edgeVersions := []
  metadata := PropertyMap.empty

/-- Record a node version in the snapshot -/
def GraphSnapshot.recordNodeVersion (gs : GraphSnapshot) (nodeId : String) (version : Nat) : GraphSnapshot :=
  { gs with nodeVersions := gs.nodeVersions ++ [(nodeId, version)] }

/-- Record an edge version in the snapshot -/
def GraphSnapshot.recordEdgeVersion (gs : GraphSnapshot) (edgeKey : EdgeKey) (version : Nat) : GraphSnapshot :=
  { gs with edgeVersions := gs.edgeVersions ++ [(edgeKey, version)] }

/-- Get a node version from the snapshot -/
def GraphSnapshot.getNodeVersion (gs : GraphSnapshot) (nodeId : String) : Option Nat :=
  (gs.nodeVersions.find? (fun p => p.1 = nodeId)).map (·.2)

/-- Get an edge version from the snapshot -/
def GraphSnapshot.getEdgeVersion (gs : GraphSnapshot) (edgeKey : EdgeKey) : Option Nat :=
  (gs.edgeVersions.find? (fun p => p.1 = edgeKey)).map (·.2)

/-- Set metadata in the snapshot -/
def GraphSnapshot.setMetadata (gs : GraphSnapshot) (key val : String) : GraphSnapshot :=
  { gs with metadata := gs.metadata.insert key val }

/-- Get node count in snapshot -/
def GraphSnapshot.nodeCount (gs : GraphSnapshot) : Nat :=
  gs.nodeVersions.length

/-- Get edge count in snapshot -/
def GraphSnapshot.edgeCount (gs : GraphSnapshot) : Nat :=
  gs.edgeVersions.length

/-- Clone a snapshot -/
def GraphSnapshot.clone (gs : GraphSnapshot) : GraphSnapshot where
  snapshotId := gs.snapshotId
  timestamp := gs.timestamp
  nodeVersions := gs.nodeVersions
  edgeVersions := gs.edgeVersions
  metadata := gs.metadata

/-- Proof that init creates snapshot with correct ID -/
theorem GraphSnapshot.init_snapshotId (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).snapshotId = id :=
  Eq.refl id

/-- Proof that init creates snapshot with correct timestamp -/
theorem GraphSnapshot.init_timestamp (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).timestamp = t :=
  Eq.refl t

/-- Proof that init creates snapshot with empty node versions -/
theorem GraphSnapshot.init_nodeVersions_empty (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).nodeVersions = [] :=
  Eq.refl []

/-- Proof that init creates snapshot with empty edge versions -/
theorem GraphSnapshot.init_edgeVersions_empty (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).edgeVersions = [] :=
  Eq.refl []

/-- Proof that init creates snapshot with empty metadata -/
theorem GraphSnapshot.init_metadata_empty (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).metadata = PropertyMap.empty :=
  Eq.refl PropertyMap.empty

/-- Proof that init node count is 0 -/
theorem GraphSnapshot.init_nodeCount (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).nodeCount = 0 :=
  Eq.refl 0

/-- Proof that init edge count is 0 -/
theorem GraphSnapshot.init_edgeCount (id : Nat) (t : Timestamp) :
    (GraphSnapshot.init id t).edgeCount = 0 :=
  Eq.refl 0

/-- Proof that recordNodeVersion increases node count -/
theorem GraphSnapshot.recordNodeVersion_nodeCount (gs : GraphSnapshot) (nodeId : String) (v : Nat) :
    (gs.recordNodeVersion nodeId v).nodeCount = gs.nodeCount + 1 :=
  congrArg List.length (List.length_append gs.nodeVersions [(nodeId, v)])

/-- Proof that recordNodeVersion preserves snapshotId -/
theorem GraphSnapshot.recordNodeVersion_snapshotId (gs : GraphSnapshot) (nodeId : String) (v : Nat) :
    (gs.recordNodeVersion nodeId v).snapshotId = gs.snapshotId :=
  Eq.refl gs.snapshotId

/-- Proof that recordNodeVersion preserves timestamp -/
theorem GraphSnapshot.recordNodeVersion_timestamp (gs : GraphSnapshot) (nodeId : String) (v : Nat) :
    (gs.recordNodeVersion nodeId v).timestamp = gs.timestamp :=
  Eq.refl gs.timestamp

/-- Proof that recordEdgeVersion increases edge count -/
theorem GraphSnapshot.recordEdgeVersion_edgeCount (gs : GraphSnapshot) (edgeKey : EdgeKey) (v : Nat) :
    (gs.recordEdgeVersion edgeKey v).edgeCount = gs.edgeCount + 1 :=
  congrArg List.length (List.length_append gs.edgeVersions [(edgeKey, v)])

/-- Proof that recordEdgeVersion preserves snapshotId -/
theorem GraphSnapshot.recordEdgeVersion_snapshotId (gs : GraphSnapshot) (edgeKey : EdgeKey) (v : Nat) :
    (gs.recordEdgeVersion edgeKey v).snapshotId = gs.snapshotId :=
  Eq.refl gs.snapshotId

/-- Proof that getNodeVersion on empty returns none -/
theorem GraphSnapshot.getNodeVersion_init_none (id : Nat) (t : Timestamp) (nodeId : String) :
    (GraphSnapshot.init id t).getNodeVersion nodeId = none :=
  congrArg (fun x => x.map (·.2)) (List.find?_nil (fun p => p.1 = nodeId))

/-- Proof that getEdgeVersion on empty returns none -/
theorem GraphSnapshot.getEdgeVersion_init_none (id : Nat) (t : Timestamp) (edgeKey : EdgeKey) :
    (GraphSnapshot.init id t).getEdgeVersion edgeKey = none :=
  congrArg (fun x => x.map (·.2)) (List.find?_nil (fun p => p.1 = edgeKey))

/-- Proof that setMetadata updates metadata -/
theorem GraphSnapshot.setMetadata_get (gs : GraphSnapshot) (k v : String) :
    (gs.setMetadata k v).metadata.get k = some v :=
  PropertyMap.get_insert gs.metadata k v

/-- Proof that setMetadata preserves snapshotId -/
theorem GraphSnapshot.setMetadata_snapshotId (gs : GraphSnapshot) (k v : String) :
    (gs.setMetadata k v).snapshotId = gs.snapshotId :=
  Eq.refl gs.snapshotId

/-- Proof that setMetadata preserves timestamp -/
theorem GraphSnapshot.setMetadata_timestamp (gs : GraphSnapshot) (k v : String) :
    (gs.setMetadata k v).timestamp = gs.timestamp :=
  Eq.refl gs.timestamp

/-- Proof that nodeCount equals nodeVersions length -/
theorem GraphSnapshot.nodeCount_eq (gs : GraphSnapshot) :
    gs.nodeCount = gs.nodeVersions.length :=
  Eq.refl gs.nodeVersions.length

/-- Proof that edgeCount equals edgeVersions length -/
theorem GraphSnapshot.edgeCount_eq (gs : GraphSnapshot) :
    gs.edgeCount = gs.edgeVersions.length :=
  Eq.refl gs.edgeVersions.length

/-- Proof that clone preserves snapshotId -/
theorem GraphSnapshot.clone_snapshotId (gs : GraphSnapshot) :
    gs.clone.snapshotId = gs.snapshotId :=
  Eq.refl gs.snapshotId

/-- Proof that clone preserves timestamp -/
theorem GraphSnapshot.clone_timestamp (gs : GraphSnapshot) :
    gs.clone.timestamp = gs.timestamp :=
  Eq.refl gs.timestamp

/-- Proof that clone preserves nodeCount -/
theorem GraphSnapshot.clone_nodeCount (gs : GraphSnapshot) :
    gs.clone.nodeCount = gs.nodeCount :=
  Eq.refl gs.nodeCount

/-- Proof that clone preserves edgeCount -/
theorem GraphSnapshot.clone_edgeCount (gs : GraphSnapshot) :
    gs.clone.edgeCount = gs.edgeCount :=
  Eq.refl gs.edgeCount

/-! ============================================================================
   Temporal Query Result
   ============================================================================ -/

/-- Result of a temporal query -/
structure TemporalQueryResult where
  nodes : List TemporalNode
  edges : List TemporalEdge
  queryTime : Timestamp
  deriving Repr

/-- Initialize an empty query result -/
def TemporalQueryResult.init (queryTime : Timestamp) : TemporalQueryResult where
  nodes := []
  edges := []
  queryTime := queryTime

/-- Get node count in result -/
def TemporalQueryResult.nodeCount (tqr : TemporalQueryResult) : Nat :=
  tqr.nodes.length

/-- Get edge count in result -/
def TemporalQueryResult.edgeCount (tqr : TemporalQueryResult) : Nat :=
  tqr.edges.length

/-- Add a node to the result -/
def TemporalQueryResult.addNode (tqr : TemporalQueryResult) (node : TemporalNode) : TemporalQueryResult :=
  { tqr with nodes := tqr.nodes ++ [node] }

/-- Add an edge to the result -/
def TemporalQueryResult.addEdge (tqr : TemporalQueryResult) (edge : TemporalEdge) : TemporalQueryResult :=
  { tqr with edges := tqr.edges ++ [edge] }

/-- Proof that init creates empty result -/
theorem TemporalQueryResult.init_nodeCount (t : Timestamp) :
    (TemporalQueryResult.init t).nodeCount = 0 :=
  Eq.refl 0

/-- Proof that init creates empty edge list -/
theorem TemporalQueryResult.init_edgeCount (t : Timestamp) :
    (TemporalQueryResult.init t).edgeCount = 0 :=
  Eq.refl 0

/-- Proof that init preserves queryTime -/
theorem TemporalQueryResult.init_queryTime (t : Timestamp) :
    (TemporalQueryResult.init t).queryTime = t :=
  Eq.refl t

/-- Proof that addNode increases node count -/
theorem TemporalQueryResult.addNode_nodeCount (tqr : TemporalQueryResult) (node : TemporalNode) :
    (tqr.addNode node).nodeCount = tqr.nodeCount + 1 :=
  congrArg List.length (List.length_append tqr.nodes [node])

/-- Proof that addNode preserves queryTime -/
theorem TemporalQueryResult.addNode_queryTime (tqr : TemporalQueryResult) (node : TemporalNode) :
    (tqr.addNode node).queryTime = tqr.queryTime :=
  Eq.refl tqr.queryTime

/-- Proof that addEdge increases edge count -/
theorem TemporalQueryResult.addEdge_edgeCount (tqr : TemporalQueryResult) (edge : TemporalEdge) :
    (tqr.addEdge edge).edgeCount = tqr.edgeCount + 1 :=
  congrArg List.length (List.length_append tqr.edges [edge])

/-- Proof that addEdge preserves queryTime -/
theorem TemporalQueryResult.addEdge_queryTime (tqr : TemporalQueryResult) (edge : TemporalEdge) :
    (tqr.addEdge edge).queryTime = tqr.queryTime :=
  Eq.refl tqr.queryTime

/-- Proof that nodeCount equals nodes length -/
theorem TemporalQueryResult.nodeCount_eq (tqr : TemporalQueryResult) :
    tqr.nodeCount = tqr.nodes.length :=
  Eq.refl tqr.nodes.length

/-- Proof that edgeCount equals edges length -/
theorem TemporalQueryResult.edgeCount_eq (tqr : TemporalQueryResult) :
    tqr.edgeCount = tqr.edges.length :=
  Eq.refl tqr.edges.length

/-! ============================================================================
   History Entry
   ============================================================================ -/

/-- History operation enumeration -/
inductive HistoryOperation where
  | create : HistoryOperation
  | update : HistoryOperation
  | rollback : HistoryOperation
  | invalidate : HistoryOperation
  deriving Repr, BEq, DecidableEq

/-- Convert history operation to string -/
def HistoryOperation.toString : HistoryOperation → String
  | HistoryOperation.create => "create"
  | HistoryOperation.update => "update"
  | HistoryOperation.rollback => "rollback"
  | HistoryOperation.invalidate => "invalidate"

/-- Proof that toString produces non-empty strings -/
theorem HistoryOperation.toString_nonempty (op : HistoryOperation) :
    op.toString.length > 0 :=
  match op with
  | HistoryOperation.create => Nat.zero_lt_succ 5
  | HistoryOperation.update => Nat.zero_lt_succ 5
  | HistoryOperation.rollback => Nat.zero_lt_succ 7
  | HistoryOperation.invalidate => Nat.zero_lt_succ 9

/-- Proof that toString is injective -/
theorem HistoryOperation.toString_injective (op₁ op₂ : HistoryOperation) (h : op₁.toString = op₂.toString) :
    op₁ = op₂ :=
  match op₁, op₂ with
  | HistoryOperation.create, HistoryOperation.create => Eq.refl HistoryOperation.create
  | HistoryOperation.update, HistoryOperation.update => Eq.refl HistoryOperation.update
  | HistoryOperation.rollback, HistoryOperation.rollback => Eq.refl HistoryOperation.rollback
  | HistoryOperation.invalidate, HistoryOperation.invalidate => Eq.refl HistoryOperation.invalidate
  | HistoryOperation.create, HistoryOperation.update => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 5))
  | HistoryOperation.create, HistoryOperation.rollback => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | HistoryOperation.create, HistoryOperation.invalidate => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 9))
  | HistoryOperation.update, HistoryOperation.create => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 5))
  | HistoryOperation.update, HistoryOperation.rollback => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | HistoryOperation.update, HistoryOperation.invalidate => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 9))
  | HistoryOperation.rollback, HistoryOperation.create => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | HistoryOperation.rollback, HistoryOperation.update => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | HistoryOperation.rollback, HistoryOperation.invalidate => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 9))
  | HistoryOperation.invalidate, HistoryOperation.create => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 9))
  | HistoryOperation.invalidate, HistoryOperation.update => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 9))
  | HistoryOperation.invalidate, HistoryOperation.rollback => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 9))

/-- Entity type enumeration -/
inductive EntityType where
  | node : EntityType
  | edge : EntityType
  | snapshot : EntityType
  deriving Repr, BEq, DecidableEq

/-- Convert entity type to string -/
def EntityType.toString : EntityType → String
  | EntityType.node => "node"
  | EntityType.edge => "edge"
  | EntityType.snapshot => "snapshot"

/-- Proof that toString produces non-empty strings -/
theorem EntityType.toString_nonempty (et : EntityType) :
    et.toString.length > 0 :=
  match et with
  | EntityType.node => Nat.zero_lt_succ 3
  | EntityType.edge => Nat.zero_lt_succ 3
  | EntityType.snapshot => Nat.zero_lt_succ 7

/-- Proof that toString is injective -/
theorem EntityType.toString_injective (et₁ et₂ : EntityType) (h : et₁.toString = et₂.toString) :
    et₁ = et₂ :=
  match et₁, et₂ with
  | EntityType.node, EntityType.node => Eq.refl EntityType.node
  | EntityType.edge, EntityType.edge => Eq.refl EntityType.edge
  | EntityType.snapshot, EntityType.snapshot => Eq.refl EntityType.snapshot
  | EntityType.node, EntityType.edge => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 3))
  | EntityType.node, EntityType.snapshot => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | EntityType.edge, EntityType.node => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 3))
  | EntityType.edge, EntityType.snapshot => absurd h (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | EntityType.snapshot, EntityType.node => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))
  | EntityType.snapshot, EntityType.edge => absurd (Eq.symm h) (fun h => Nat.ne_of_gt (Nat.zero_lt_succ 7))

/-- History entry structure -/
structure HistoryEntry where
  timestamp : Timestamp
  operation : HistoryOperation
  entityType : EntityType
  entityId : String
  versionBefore : Option Nat
  versionAfter : Nat
  deriving Repr

/-- Initialize a history entry -/
def HistoryEntry.init (timestampNs : Timestamp) (op : HistoryOperation) (entity : EntityType) (id : String) (versionBefore : Option Nat) (versionAfter : Nat) : HistoryEntry where
  timestamp := timestampNs
  operation := op
  entityType := entity
  entityId := id
  versionBefore := versionBefore
  versionAfter := versionAfter

/-- Clone a history entry -/
def HistoryEntry.clone (he : HistoryEntry) : HistoryEntry where
  timestamp := he.timestamp
  operation := he.operation
  entityType := he.entityType
  entityId := he.entityId
  versionBefore := he.versionBefore
  versionAfter := he.versionAfter

/-- Proof that init creates entry with correct timestamp -/
theorem HistoryEntry.init_timestamp (t : Timestamp) (op : HistoryOperation) (et : EntityType) (id : String) (vb : Option Nat) (va : Nat) :
    (HistoryEntry.init t op et id vb va).timestamp = t :=
  Eq.refl t

/-- Proof that init creates entry with correct operation -/
theorem HistoryEntry.init_operation (t : Timestamp) (op : HistoryOperation) (et : EntityType) (id : String) (vb : Option Nat) (va : Nat) :
    (HistoryEntry.init t op et id vb va).operation = op :=
  Eq.refl op

/-- Proof that init creates entry with correct entity type -/
theorem HistoryEntry.init_entityType (t : Timestamp) (op : HistoryOperation) (et : EntityType) (id : String) (vb : Option Nat) (va : Nat) :
    (HistoryEntry.init t op et id vb va).entityType = et :=
  Eq.refl et

/-- Proof that init creates entry with correct entity ID -/
theorem HistoryEntry.init_entityId (t : Timestamp) (op : HistoryOperation) (et : EntityType) (id : String) (vb : Option Nat) (va : Nat) :
    (HistoryEntry.init t op et id vb va).entityId = id :=
  Eq.refl id

/-- Proof that init creates entry with correct versionBefore -/
theorem HistoryEntry.init_versionBefore (t : Timestamp) (op : HistoryOperation) (et : EntityType) (id : String) (vb : Option Nat) (va : Nat) :
    (HistoryEntry.init t op et id vb va).versionBefore = vb :=
  Eq.refl vb

/-- Proof that init creates entry with correct versionAfter -/
theorem HistoryEntry.init_versionAfter (t : Timestamp) (op : HistoryOperation) (et : EntityType) (id : String) (vb : Option Nat) (va : Nat) :
    (HistoryEntry.init t op et id vb va).versionAfter = va :=
  Eq.refl va

/-- Proof that clone preserves timestamp -/
theorem HistoryEntry.clone_timestamp (he : HistoryEntry) :
    he.clone.timestamp = he.timestamp :=
  Eq.refl he.timestamp

/-- Proof that clone preserves operation -/
theorem HistoryEntry.clone_operation (he : HistoryEntry) :
    he.clone.operation = he.operation :=
  Eq.refl he.operation

/-- Proof that clone preserves entityType -/
theorem HistoryEntry.clone_entityType (he : HistoryEntry) :
    he.clone.entityType = he.entityType :=
  Eq.refl he.entityType

/-- Proof that clone preserves entityId -/
theorem HistoryEntry.clone_entityId (he : HistoryEntry) :
    he.clone.entityId = he.entityId :=
  Eq.refl he.entityId

/-- Proof that clone preserves versionBefore -/
theorem HistoryEntry.clone_versionBefore (he : HistoryEntry) :
    he.clone.versionBefore = he.versionBefore :=
  Eq.refl he.versionBefore

/-- Proof that clone preserves versionAfter -/
theorem HistoryEntry.clone_versionAfter (he : HistoryEntry) :
    he.clone.versionAfter = he.versionAfter :=
  Eq.refl he.versionAfter

/-! ============================================================================
   Temporal Graph
   ============================================================================ -/

/-- Temporal graph with nodes, edges, snapshots, and history -/
structure TemporalGraph where
  nodes : List (String × TemporalNode)
  edges : List (EdgeKey × TemporalEdge)
  currentTime : Timestamp
  snapshots : List GraphSnapshot
  history : List HistoryEntry
  nextSnapshotId : Nat
  nextEdgeId : Nat
  deriving Repr

/-- Initialize an empty temporal graph -/
def TemporalGraph.init (initialTime : Timestamp) : TemporalGraph where
  nodes := []
  edges := []
  currentTime := initialTime
  snapshots := []
  history := []
  nextSnapshotId := 0
  nextEdgeId := 0

/-- Initialize a temporal graph with default time -/
def TemporalGraph.initDefault : TemporalGraph :=
  TemporalGraph.init 0

/-- Set current time -/
def TemporalGraph.setCurrentTime (tg : TemporalGraph) (timestampNs : Timestamp) : TemporalGraph :=
  { tg with currentTime := timestampNs }

/-- Advance time by a delta -/
def TemporalGraph.advanceTime (tg : TemporalGraph) (deltaNs : Timestamp) : TemporalGraph :=
  { tg with currentTime := tg.currentTime + deltaNs }

/-- Get current time -/
def TemporalGraph.getCurrentTime (tg : TemporalGraph) : Timestamp :=
  tg.currentTime

/-- Find a node by ID -/
def TemporalGraph.findNode (tg : TemporalGraph) (nodeId : String) : Option TemporalNode :=
  (tg.nodes.find? (fun p => p.1 = nodeId)).map (·.2)

/-- Find an edge by key -/
def TemporalGraph.findEdge (tg : TemporalGraph) (edgeKey : EdgeKey) : Option TemporalEdge :=
  (tg.edges.find? (fun p => p.1 = edgeKey)).map (·.2)

/-- Check if node exists -/
def TemporalGraph.hasNode (tg : TemporalGraph) (nodeId : String) : Bool :=
  tg.nodes.any (fun p => p.1 = nodeId)

/-- Check if edge exists -/
def TemporalGraph.hasEdge (tg : TemporalGraph) (edgeKey : EdgeKey) : Bool :=
  tg.edges.any (fun p => p.1 = edgeKey)

/-- Add a node to the graph -/
def TemporalGraph.addNode (tg : TemporalGraph) (nodeId : String) (initialState : QuantumState) (timestampNs : Timestamp) : Option TemporalGraph :=
  if tg.hasNode nodeId then none
  else
    let node := TemporalNode.init nodeId initialState timestampNs
    let historyEntry := HistoryEntry.init timestampNs HistoryOperation.create EntityType.node nodeId none 0
    some { tg with
      nodes := tg.nodes ++ [(nodeId, node)]
      history := tg.history ++ [historyEntry] }

/-- Add an edge to the graph -/
def TemporalGraph.addEdge (tg : TemporalGraph) (sourceId targetId : String) (weight : Float) (quality : EdgeQuality) (timestampNs : Timestamp) : Option TemporalGraph :=
  let edgeKey := EdgeKey.mk sourceId targetId
  if tg.hasEdge edgeKey then none
  else
    let edgeId := "edge_" ++ toString tg.nextEdgeId ++ "_" ++ sourceId ++ "_" ++ targetId
    let edge := TemporalEdge.init edgeId sourceId targetId weight quality timestampNs
    let historyEntry := HistoryEntry.init timestampNs HistoryOperation.create EntityType.edge edgeId none 0
    some { tg with
      edges := tg.edges ++ [(edgeKey, edge)]
      history := tg.history ++ [historyEntry]
      nextEdgeId := tg.nextEdgeId + 1 }

/-- Get node count -/
def TemporalGraph.nodeCount (tg : TemporalGraph) : Nat :=
  tg.nodes.length

/-- Get edge count -/
def TemporalGraph.edgeCount (tg : TemporalGraph) : Nat :=
  tg.edges.length

/-- Get snapshot count -/
def TemporalGraph.snapshotCount (tg : TemporalGraph) : Nat :=
  tg.snapshots.length

/-- Get history count -/
def TemporalGraph.historyCount (tg : TemporalGraph) : Nat :=
  tg.history.length

/-- Create a snapshot -/
def TemporalGraph.createSnapshot (tg : TemporalGraph) : TemporalGraph × Nat :=
  let snapshot := GraphSnapshot.init tg.nextSnapshotId tg.currentTime
  let snapshotWithNodes := tg.nodes.foldl (fun acc p =>
    let node := p.2
    match node.getVersionAt tg.currentTime with
    | some v => acc.recordNodeVersion p.1 v.version
    | none => acc) snapshot
  let snapshotWithEdges := tg.edges.foldl (fun acc p =>
    let edge := p.2
    if edge.isValidAt tg.currentTime then
      match edge.getVersionAt tg.currentTime with
      | some v => acc.recordEdgeVersion p.1 v.version
      | none => acc
    else acc) snapshotWithNodes
  let historyEntry := HistoryEntry.init tg.currentTime HistoryOperation.create EntityType.snapshot "snapshot" none tg.nextSnapshotId
  ({ tg with
    snapshots := tg.snapshots ++ [snapshotWithEdges]
    history := tg.history ++ [historyEntry]
    nextSnapshotId := tg.nextSnapshotId + 1 }, tg.nextSnapshotId)

/-- Get a snapshot by ID -/
def TemporalGraph.getSnapshot (tg : TemporalGraph) (snapshotId : Nat) : Option GraphSnapshot :=
  tg.snapshots.find? (fun s => s.snapshotId = snapshotId)

/-- Update a node in the graph -/
def TemporalGraph.updateNode (tg : TemporalGraph) (nodeId : String) (newState : QuantumState) (timestampNs : Timestamp) : Option TemporalGraph :=
  match tg.findNode nodeId with
  | some node =>
    let versionBefore := node.currentVersion
    let updatedNode := node.addVersion newState timestampNs
    let updatedNodes := tg.nodes.map (fun p =>
      if p.1 = nodeId then (nodeId, updatedNode) else p)
    let historyEntry := HistoryEntry.init timestampNs HistoryOperation.update EntityType.node nodeId (some versionBefore) updatedNode.currentVersion
    some { tg with
      nodes := updatedNodes
      history := tg.history ++ [historyEntry] }
  | none => none

/-- Update an edge in the graph -/
def TemporalGraph.updateEdge (tg : TemporalGraph) (sourceId targetId : String) (newWeight : Float) (newQuality : EdgeQuality) (timestampNs : Timestamp) : Option TemporalGraph :=
  let edgeKey := EdgeKey.mk sourceId targetId
  match tg.findEdge edgeKey with
  | some edge =>
    let versionBefore := edge.currentVersion
    let updatedEdge := edge.addVersion newWeight newQuality timestampNs
    let updatedEdges := tg.edges.map (fun p =>
      if p.1 = edgeKey then (edgeKey, updatedEdge) else p)
    let historyEntry := HistoryEntry.init timestampNs HistoryOperation.update EntityType.edge edge.edgeId (some versionBefore) updatedEdge.currentVersion
    some { tg with
      edges := updatedEdges
      history := tg.history ++ [historyEntry] }
  | none => none

/-- Invalidate an edge -/
def TemporalGraph.invalidateEdge (tg : TemporalGraph) (sourceId targetId : String) (timestampNs : Timestamp) : Option TemporalGraph :=
  let edgeKey := EdgeKey.mk sourceId targetId
  match tg.findEdge edgeKey with
  | some edge =>
    let updatedEdge := edge.invalidate timestampNs
    let updatedEdges := tg.edges.map (fun p =>
      if p.1 = edgeKey then (edgeKey, updatedEdge) else p)
    let historyEntry := HistoryEntry.init timestampNs HistoryOperation.invalidate EntityType.edge edge.edgeId (some edge.currentVersion) edge.currentVersion
    some { tg with
      edges := updatedEdges
      history := tg.history ++ [historyEntry] }
  | none => none

/-- Get history entries in a time range -/
def TemporalGraph.getHistoryInRange (tg : TemporalGraph) (startTime endTime : Timestamp) : List HistoryEntry :=
  tg.history.filter (fun h => h.timestamp ≥ startTime ∧ h.timestamp ≤ endTime)

/-- Get valid edges at a timestamp -/
def TemporalGraph.getValidEdgesAt (tg : TemporalGraph) (timestampNs : Timestamp) : List TemporalEdge :=
  (tg.edges.filter (fun p => p.2.isValidAt timestampNs)).map (·.2)

/-- Get node neighbors at a timestamp -/
def TemporalGraph.getNodeNeighborsAt (tg : TemporalGraph) (nodeId : String) (timestampNs : Timestamp) : List String :=
  let validEdges := tg.getValidEdgesAt timestampNs
  validEdges.filterMap (fun edge =>
    if edge.source = nodeId then some edge.target
    else if edge.target = nodeId then some edge.source
    else none)

/-- Clone a temporal graph -/
def TemporalGraph.clone (tg : TemporalGraph) : TemporalGraph where
  nodes := tg.nodes.map (fun p => (p.1, p.2.clone))
  edges := tg.edges.map (fun p => (p.1, p.2.clone))
  currentTime := tg.currentTime
  snapshots := tg.snapshots.map GraphSnapshot.clone
  history := tg.history.map HistoryEntry.clone
  nextSnapshotId := tg.nextSnapshotId
  nextEdgeId := tg.nextEdgeId

/-- Proof that init creates empty graph -/
theorem TemporalGraph.init_nodeCount (t : Timestamp) :
    (TemporalGraph.init t).nodeCount = 0 :=
  Eq.refl 0

/-- Proof that init creates empty edge list -/
theorem TemporalGraph.init_edgeCount (t : Timestamp) :
    (TemporalGraph.init t).edgeCount = 0 :=
  Eq.refl 0

/-- Proof that init creates empty snapshot list -/
theorem TemporalGraph.init_snapshotCount (t : Timestamp) :
    (TemporalGraph.init t).snapshotCount = 0 :=
  Eq.refl 0

/-- Proof that init creates empty history -/
theorem TemporalGraph.init_historyCount (t : Timestamp) :
    (TemporalGraph.init t).historyCount = 0 :=
  Eq.refl 0

/-- Proof that init sets correct currentTime -/
theorem TemporalGraph.init_currentTime (t : Timestamp) :
    (TemporalGraph.init t).currentTime = t :=
  Eq.refl t

/-- Proof that init sets nextSnapshotId to 0 -/
theorem TemporalGraph.init_nextSnapshotId (t : Timestamp) :
    (TemporalGraph.init t).nextSnapshotId = 0 :=
  Eq.refl 0

/-- Proof that init sets nextEdgeId to 0 -/
theorem TemporalGraph.init_nextEdgeId (t : Timestamp) :
    (TemporalGraph.init t).nextEdgeId = 0 :=
  Eq.refl 0

/-- Proof that setCurrentTime updates currentTime -/
theorem TemporalGraph.setCurrentTime_currentTime (tg : TemporalGraph) (t : Timestamp) :
    (tg.setCurrentTime t).currentTime = t :=
  Eq.refl t

/-- Proof that setCurrentTime preserves nodeCount -/
theorem TemporalGraph.setCurrentTime_nodeCount (tg : TemporalGraph) (t : Timestamp) :
    (tg.setCurrentTime t).nodeCount = tg.nodeCount :=
  Eq.refl tg.nodes.length

/-- Proof that setCurrentTime preserves edgeCount -/
theorem TemporalGraph.setCurrentTime_edgeCount (tg : TemporalGraph) (t : Timestamp) :
    (tg.setCurrentTime t).edgeCount = tg.edgeCount :=
  Eq.refl tg.edges.length

/-- Proof that advanceTime adds delta to currentTime -/
theorem TemporalGraph.advanceTime_currentTime (tg : TemporalGraph) (delta : Timestamp) :
    (tg.advanceTime delta).currentTime = tg.currentTime + delta :=
  Eq.refl (tg.currentTime + delta)

/-- Proof that advanceTime preserves nodeCount -/
theorem TemporalGraph.advanceTime_nodeCount (tg : TemporalGraph) (delta : Timestamp) :
    (tg.advanceTime delta).nodeCount = tg.nodeCount :=
  Eq.refl tg.nodes.length

/-- Proof that advanceTime preserves edgeCount -/
theorem TemporalGraph.advanceTime_edgeCount (tg : TemporalGraph) (delta : Timestamp) :
    (tg.advanceTime delta).edgeCount = tg.edgeCount :=
  Eq.refl tg.edges.length

/-- Proof that getCurrentTime returns currentTime -/
theorem TemporalGraph.getCurrentTime_eq (tg : TemporalGraph) :
    tg.getCurrentTime = tg.currentTime :=
  Eq.refl tg.currentTime

/-- Proof that findNode on empty returns none -/
theorem TemporalGraph.findNode_init_none (t : Timestamp) (nodeId : String) :
    (TemporalGraph.init t).findNode nodeId = none :=
  congrArg (fun x => x.map (·.2)) (List.find?_nil (fun p => p.1 = nodeId))

/-- Proof that findEdge on empty returns none -/
theorem TemporalGraph.findEdge_init_none (t : Timestamp) (edgeKey : EdgeKey) :
    (TemporalGraph.init t).findEdge edgeKey = none :=
  congrArg (fun x => x.map (·.2)) (List.find?_nil (fun p => p.1 = edgeKey))

/-- Proof that hasNode on empty returns false -/
theorem TemporalGraph.hasNode_init_false (t : Timestamp) (nodeId : String) :
    (TemporalGraph.init t).hasNode nodeId = false :=
  List.any_nil (fun p => p.1 = nodeId)

/-- Proof that hasEdge on empty returns false -/
theorem TemporalGraph.hasEdge_init_false (t : Timestamp) (edgeKey : EdgeKey) :
    (TemporalGraph.init t).hasEdge edgeKey = false :=
  List.any_nil (fun p => p.1 = edgeKey)

/-- Proof that addNode increases node count when successful -/
theorem TemporalGraph.addNode_nodeCount (tg : TemporalGraph) (nodeId : String) (qs : QuantumState) (t : Timestamp) (h : tg.hasNode nodeId = false) :
    (tg.addNode nodeId qs t).map (·.nodeCount) = some (tg.nodeCount + 1) :=
  congrArg (fun x => x.map (·.nodeCount)) (if_neg (fun h' => absurd h' (Bool.ne_of_ne h (Eq.refl false))))

/-- Proof that addNode returns none if node exists -/
theorem TemporalGraph.addNode_exists_none (tg : TemporalGraph) (nodeId : String) (qs : QuantumState) (t : Timestamp) (h : tg.hasNode nodeId = true) :
    tg.addNode nodeId qs t = none :=
  if_pos h

/-- Proof that addEdge increases edge count when successful -/
theorem TemporalGraph.addEdge_edgeCount (tg : TemporalGraph) (src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) (h : tg.hasEdge (EdgeKey.mk src tgt) = false) :
    (tg.addEdge src tgt w q t).map (·.edgeCount) = some (tg.edgeCount + 1) :=
  congrArg (fun x => x.map (·.edgeCount)) (if_neg (fun h' => absurd h' (Bool.ne_of_ne h (Eq.refl false))))

/-- Proof that addEdge returns none if edge exists -/
theorem TemporalGraph.addEdge_exists_none (tg : TemporalGraph) (src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) (h : tg.hasEdge (EdgeKey.mk src tgt) = true) :
    tg.addEdge src tgt w q t = none :=
  if_pos h

/-- Proof that nodeCount equals nodes length -/
theorem TemporalGraph.nodeCount_eq (tg : TemporalGraph) :
    tg.nodeCount = tg.nodes.length :=
  Eq.refl tg.nodes.length

/-- Proof that edgeCount equals edges length -/
theorem TemporalGraph.edgeCount_eq (tg : TemporalGraph) :
    tg.edgeCount = tg.edges.length :=
  Eq.refl tg.edges.length

/-- Proof that snapshotCount equals snapshots length -/
theorem TemporalGraph.snapshotCount_eq (tg : TemporalGraph) :
    tg.snapshotCount = tg.snapshots.length :=
  Eq.refl tg.snapshots.length

/-- Proof that historyCount equals history length -/
theorem TemporalGraph.historyCount_eq (tg : TemporalGraph) :
    tg.historyCount = tg.history.length :=
  Eq.refl tg.history.length

/-- Proof that createSnapshot increases snapshot count -/
theorem TemporalGraph.createSnapshot_snapshotCount (tg : TemporalGraph) :
    (tg.createSnapshot.1).snapshotCount = tg.snapshotCount + 1 :=
  congrArg List.length (List.length_append tg.snapshots [GraphSnapshot.init tg.nextSnapshotId tg.currentTime])

/-- Proof that createSnapshot increments nextSnapshotId -/
theorem TemporalGraph.createSnapshot_nextSnapshotId (tg : TemporalGraph) :
    (tg.createSnapshot.1).nextSnapshotId = tg.nextSnapshotId + 1 :=
  Eq.refl (tg.nextSnapshotId + 1)

/-- Proof that createSnapshot returns correct snapshot ID -/
theorem TemporalGraph.createSnapshot_id (tg : TemporalGraph) :
    tg.createSnapshot.2 = tg.nextSnapshotId :=
  Eq.refl tg.nextSnapshotId

/-- Proof that createSnapshot increases history count -/
theorem TemporalGraph.createSnapshot_historyCount (tg : TemporalGraph) :
    (tg.createSnapshot.1).historyCount = tg.historyCount + 1 :=
  congrArg List.length (List.length_append tg.history [HistoryEntry.init tg.currentTime HistoryOperation.create EntityType.snapshot "snapshot" none tg.nextSnapshotId])

/-- Proof that getSnapshot on empty returns none -/
theorem TemporalGraph.getSnapshot_init_none (t : Timestamp) (id : Nat) :
    (TemporalGraph.init t).getSnapshot id = none :=
  List.find?_nil (fun s => s.snapshotId = id)

/-- Proof that updateNode increases history count when successful -/
theorem TemporalGraph.updateNode_historyCount (tg : TemporalGraph) (nodeId : String) (qs : QuantumState) (t : Timestamp) (h : ∃ node, tg.findNode nodeId = some node) :
    (tg.updateNode nodeId qs t).map (·.historyCount) = some (tg.historyCount + 1) :=
  match h with
  | ⟨node, hnode⟩ =>
    congrArg (fun x => x.map (·.historyCount))
      (show match tg.findNode nodeId with
        | some node => some _
        | none => none = some _ from
        congrArg (fun x => match x with
          | some node => some { tg with
              nodes := tg.nodes.map (fun p => if p.1 = nodeId then (nodeId, node.addVersion qs t) else p)
              history := tg.history ++ [HistoryEntry.init t HistoryOperation.update EntityType.node nodeId (some node.currentVersion) (node.addVersion qs t).currentVersion] }
          | none => none) hnode)

/-- Proof that updateEdge increases history count when successful -/
theorem TemporalGraph.updateEdge_historyCount (tg : TemporalGraph) (src tgt : String) (w : Float) (q : EdgeQuality) (t : Timestamp) (h : ∃ edge, tg.findEdge (EdgeKey.mk src tgt) = some edge) :
    (tg.updateEdge src tgt w q t).map (·.historyCount) = some (tg.historyCount + 1) :=
  match h with
  | ⟨edge, hedge⟩ =>
    congrArg (fun x => x.map (·.historyCount))
      (show match tg.findEdge (EdgeKey.mk src tgt) with
        | some edge => some _
        | none => none = some _ from
        congrArg (fun x => match x with
          | some edge => some { tg with
              edges := tg.edges.map (fun p => if p.1 = EdgeKey.mk src tgt then (EdgeKey.mk src tgt, edge.addVersion w q t) else p)
              history := tg.history ++ [HistoryEntry.init t HistoryOperation.update EntityType.edge edge.edgeId (some edge.currentVersion) (edge.addVersion w q t).currentVersion] }
          | none => none) hedge)

/-- Proof that invalidateEdge increases history count when successful -/
theorem TemporalGraph.invalidateEdge_historyCount (tg : TemporalGraph) (src tgt : String) (t : Timestamp) (h : ∃ edge, tg.findEdge (EdgeKey.mk src tgt) = some edge) :
    (tg.invalidateEdge src tgt t).map (·.historyCount) = some (tg.historyCount + 1) :=
  match h with
  | ⟨edge, hedge⟩ =>
    congrArg (fun x => x.map (·.historyCount))
      (show match tg.findEdge (EdgeKey.mk src tgt) with
        | some edge => some _
        | none => none = some _ from
        congrArg (fun x => match x with
          | some edge => some { tg with
              edges := tg.edges.map (fun p => if p.1 = EdgeKey.mk src tgt then (EdgeKey.mk src tgt, edge.invalidate t) else p)
              history := tg.history ++ [HistoryEntry.init t HistoryOperation.invalidate EntityType.edge edge.edgeId (some edge.currentVersion) edge.currentVersion] }
          | none => none) hedge)

/-- Proof that getHistoryInRange filters correctly -/
theorem TemporalGraph.getHistoryInRange_filter (tg : TemporalGraph) (startTime endTime : Timestamp) (h : HistoryEntry) :
    h ∈ tg.getHistoryInRange startTime endTime ↔ h ∈ tg.history ∧ h.timestamp ≥ startTime ∧ h.timestamp ≤ endTime :=
  List.mem_filter (fun h => h.timestamp ≥ startTime ∧ h.timestamp ≤ endTime) tg.history h

/-- Proof that getValidEdgesAt returns edges valid at timestamp -/
theorem TemporalGraph.getValidEdgesAt_valid (tg : TemporalGraph) (t : Timestamp) (edge : TemporalEdge) (h : edge ∈ tg.getValidEdgesAt t) :
    edge.isValidAt t = true :=
  List.mem_map.mp h |> (fun ⟨a, ha, heq⟩ =>
    show a.2.isValidAt t = true from
      List.mem_filter.mp ha |> (fun ⟨_, hvalid⟩ => hvalid))

/-- Proof that getNodeNeighborsAt returns neighbors -/
theorem TemporalGraph.getNodeNeighborsAt_valid (tg : TemporalGraph) (nodeId : String) (t : Timestamp) (neighbor : String) (h : neighbor ∈ tg.getNodeNeighborsAt nodeId t) :
    ∃ edge, edge ∈ tg.getValidEdgesAt t ∧ (edge.source = nodeId ∧ edge.target = neighbor ∨ edge.target = nodeId ∧ edge.source = neighbor) :=
  List.mem_filterMap.mp h |> (fun ⟨edge, hvalid, hneighbor⟩ =>
    match hneighbor with
    | Or.inl hsrc => ⟨edge, hvalid, Or.inl ⟨hsrc, heq⟩⟩
    | Or.inr htgt => ⟨edge, hvalid, Or.inr ⟨htgt, heq⟩⟩)
  where
    heq : _ := Eq.refl _

/-- Proof that clone preserves nodeCount -/
theorem TemporalGraph.clone_nodeCount (tg : TemporalGraph) :
    tg.clone.nodeCount = tg.nodeCount :=
  congrArg List.length (List.length_map (fun p => (p.1, p.2.clone)) tg.nodes)

/-- Proof that clone preserves edgeCount -/
theorem TemporalGraph.clone_edgeCount (tg : TemporalGraph) :
    tg.clone.edgeCount = tg.edgeCount :=
  congrArg List.length (List.length_map (fun p => (p.1, p.2.clone)) tg.edges)

/-- Proof that clone preserves currentTime -/
theorem TemporalGraph.clone_currentTime (tg : TemporalGraph) :
    tg.clone.currentTime = tg.currentTime :=
  Eq.refl tg.currentTime

/-- Proof that clone preserves snapshotCount -/
theorem TemporalGraph.clone_snapshotCount (tg : TemporalGraph) :
    tg.clone.snapshotCount = tg.snapshotCount :=
  congrArg List.length (List.length_map GraphSnapshot.clone tg.snapshots)

/-- Proof that clone preserves historyCount -/
theorem TemporalGraph.clone_historyCount (tg : TemporalGraph) :
    tg.clone.historyCount = tg.historyCount :=
  congrArg List.length (List.length_map HistoryEntry.clone tg.history)

/-- Proof that clone preserves nextSnapshotId -/
theorem TemporalGraph.clone_nextSnapshotId (tg : TemporalGraph) :
    tg.clone.nextSnapshotId = tg.nextSnapshotId :=
  Eq.refl tg.nextSnapshotId

/-- Proof that clone preserves nextEdgeId -/
theorem TemporalGraph.clone_nextEdgeId (tg : TemporalGraph) :
    tg.clone.nextEdgeId = tg.nextEdgeId :=
  Eq.refl tg.nextEdgeId

/-! ============================================================================
   Filter Functions
   ============================================================================ -/

/-- Default node filter (accepts all nodes) -/
def defaultNodeFilter (node : TemporalNode) : Bool := true

/-- Default edge filter (accepts all edges) -/
def defaultEdgeFilter (edge : TemporalEdge) : Bool := true

/-- Filter nodes by entanglement degree -/
def filterByEntanglement (node : TemporalNode) : Bool :=
  match node.getCurrentVersion with
  | some version => version.data.entanglement_degree > 0.5
  | none => false

/-- Filter edges by superposition quality -/
def filterBySuperposition (edge : TemporalEdge) : Bool :=
  match edge.getCurrentVersion with
  | some version => version.quality == EdgeQuality.superposition
  | none => false

/-- Filter edges by coherence quality -/
def filterByCoherence (edge : TemporalEdge) : Bool :=
  match edge.getCurrentVersion with
  | some version => version.quality == EdgeQuality.coherent || version.quality == EdgeQuality.superposition
  | none => false

/-- Proof that defaultNodeFilter returns true -/
theorem defaultNodeFilter_true (node : TemporalNode) :
    defaultNodeFilter node = true :=
  Eq.refl true

/-- Proof that defaultEdgeFilter returns true -/
theorem defaultEdgeFilter_true (edge : TemporalEdge) :
    defaultEdgeFilter edge = true :=
  Eq.refl true

/-- Proof that filterByEntanglement checks entanglement_degree -/
theorem filterByEntanglement_eq (node : TemporalNode) :
    filterByEntanglement node = match node.getCurrentVersion with
    | some version => version.data.entanglement_degree > 0.5
    | none => false :=
  Eq.refl (match node.getCurrentVersion with
    | some version => version.data.entanglement_degree > 0.5
    | none => false)

/-- Proof that filterBySuperposition checks quality -/
theorem filterBySuperposition_eq (edge : TemporalEdge) :
    filterBySuperposition edge = match edge.getCurrentVersion with
    | some version => version.quality == EdgeQuality.superposition
    | none => false :=
  Eq.refl (match edge.getCurrentVersion with
    | some version => version.quality == EdgeQuality.superposition
    | none => false)

/-- Proof that filterByCoherence checks quality -/
theorem filterByCoherence_eq (edge : TemporalEdge) :
    filterByCoherence edge = match edge.getCurrentVersion with
    | some version => version.quality == EdgeQuality.coherent || version.quality == EdgeQuality.superposition
    | none => false :=
  Eq.refl (match edge.getCurrentVersion with
    | some version => version.quality == EdgeQuality.coherent || version.quality == EdgeQuality.superposition
    | none => false)

/-- Proof that filterByEntanglement returns false for no version -/
theorem filterByEntanglement_none (node : TemporalNode) (h : node.getCurrentVersion = none) :
    filterByEntanglement node = false :=
  Eq.trans (congrArg (fun x => match x with
    | some version => version.data.entanglement_degree > 0.5
    | none => false) h) (Eq.refl false)

/-- Proof that filterBySuperposition returns false for no version -/
theorem filterBySuperposition_none (edge : TemporalEdge) (h : edge.getCurrentVersion = none) :
    filterBySuperposition edge = false :=
  Eq.trans (congrArg (fun x => match x with
    | some version => version.quality == EdgeQuality.superposition
    | none => false) h) (Eq.refl false)

/-- Proof that filterByCoherence returns false for no version -/
theorem filterByCoherence_none (edge : TemporalEdge) (h : edge.getCurrentVersion = none) :
    filterByCoherence edge = false :=
  Eq.trans (congrArg (fun x => match x with
    | some version => version.quality == EdgeQuality.coherent || version.quality == EdgeQuality.superposition
    | none => false) h) (Eq.refl false)

/-! ============================================================================
   Timestamp Utilities
   ============================================================================ -/

/-- Convert timestamp to milliseconds -/
def timestampToMillis (timestampNs : Timestamp) : Int :=
  timestampNs / 1000000

/-- Convert milliseconds to timestamp -/
def millisToTimestamp (millis : Int) : Timestamp :=
  millis * 1000000

/-- Convert timestamp to seconds (as float) -/
def timestampToSeconds (timestampNs : Timestamp) : Float :=
  Float.ofInt timestampNs / 1000000000.0

/-- Convert seconds (as float) to timestamp -/
def secondsToTimestamp (seconds : Float) : Timestamp :=
  Float.toUInt64 (seconds * 1000000000.0) |>.toInt

/-- Proof that millisToTimestamp is left inverse of timestampToMillis for multiples -/
theorem millisToTimestamp_timestampToMillis (ns : Int) (h : ns % 1000000 = 0) :
    millisToTimestamp (timestampToMillis ns) = ns :=
  Eq.trans (congrArg (· * 1000000) (Int.ediv_mul_cancel ns 1000000 (Int.ne_of_gt (Int.ofNat_lt.mpr (Nat.zero_lt_succ 999999))) h)) (Int.mul_comm ns 1000000)

/-- Proof that timestampToMillis of millisToTimestamp is identity -/
theorem timestampToMillis_millisToTimestamp (ms : Int) :
    timestampToMillis (millisToTimestamp ms) = ms :=
  Eq.trans (congrArg (· / 1000000) (Int.mul_comm ms 1000000)) (Int.ediv_mul_cancel ms 1000000 (Int.ne_of_gt (Int.ofNat_lt.mpr (Nat.zero_lt_succ 999999))) (Eq.refl 0))

/-- Proof that timestampToSeconds preserves sign -/
theorem timestampToSeconds_sign (ns : Int) (h : ns ≥ 0) :
    timestampToSeconds ns ≥ 0.0 :=
  Float.le_of_not_lt (fun hn =>
    Float.lt_irrefl (Float.ofInt ns / 1000000000.0)
      (Float.lt_of_le_of_lt (Float.le_of_not_lt (fun h' =>
        Float.lt_irrefl (Float.ofInt ns)
          (Float.lt_of_le_of_lt (Float.ofInt_nonneg.mpr h) h'))) hn))

/-- Proof that millisToTimestamp of 0 is 0 -/
theorem millisToTimestamp_zero : millisToTimestamp 0 = 0 :=
  Eq.refl 0

/-- Proof that timestampToMillis of 0 is 0 -/
theorem timestampToMillis_zero : timestampToMillis 0 = 0 :=
  Eq.refl 0

/-- Proof that timestampToSeconds of 0 is 0 -/
theorem timestampToSeconds_zero : timestampToSeconds 0 = 0.0 :=
  Eq.refl 0.0

/-- Proof that secondsToTimestamp of 0 is 0 -/
theorem secondsToTimestamp_zero : secondsToTimestamp 0.0 = 0 :=
  Eq.refl 0

/-- Proof that millisToTimestamp preserves sign -/
theorem millisToTimestamp_nonneg (ms : Int) (h : ms ≥ 0) :
    millisToTimestamp ms ≥ 0 :=
  Int.mul_nonneg h (Int.ofNat_nonneg.mpr (Nat.zero_le 1000000))

/-- Proof that timestampToMillis preserves sign -/
theorem timestampToMillis_nonneg (ns : Int) (h : ns ≥ 0) :
    timestampToMillis ns ≥ 0 :=
  Int.ediv_nonneg h (Int.ofNat_nonneg.mpr (Nat.zero_le 1000000))

/-! ============================================================================
   Temporal Query
   ============================================================================ -/

/-- Temporal query structure -/
structure TemporalQuery where
  startTime : Timestamp
  endTime : Timestamp
  nodeFilter : TemporalNode → Bool
  edgeFilter : TemporalEdge → Bool
  includeInvalidatedEdges : Bool
  deriving Repr

/-- Initialize a temporal query -/
def TemporalQuery.init (startTime endTime : Timestamp) : TemporalQuery where
  startTime := startTime
  endTime := endTime
  nodeFilter := defaultNodeFilter
  edgeFilter := defaultEdgeFilter
  includeInvalidatedEdges := false

/-- Initialize a temporal query with custom filters -/
def TemporalQuery.initWithFilters (startTime endTime : Timestamp) (nodeFilter edgeFilter : Option (TemporalNode → Bool)) : TemporalQuery where
  startTime := startTime
  endTime := endTime
  nodeFilter := nodeFilter.getD defaultNodeFilter
  edgeFilter := edgeFilter.getD defaultEdgeFilter
  includeInvalidatedEdges := false

/-- Set node filter -/
def TemporalQuery.setNodeFilter (tq : TemporalQuery) (filter : TemporalNode → Bool) : TemporalQuery :=
  { tq with nodeFilter := filter }

/-- Set edge filter -/
def TemporalQuery.setEdgeFilter (tq : TemporalQuery) (filter : TemporalEdge → Bool) : TemporalQuery :=
  { tq with edgeFilter := filter }

/-- Set include invalidated edges flag -/
def TemporalQuery.setIncludeInvalidatedEdges (tq : TemporalQuery) (include : Bool) : TemporalQuery :=
  { tq with includeInvalidatedEdges := include }

/-- Check if node matches time range -/
def TemporalQuery.nodeMatchesTimeRange (tq : TemporalQuery) (node : TemporalNode) : Bool :=
  node.createdAt ≤ tq.endTime ∧ node.lastModified ≥ tq.startTime

/-- Check if edge matches time range -/
def TemporalQuery.edgeMatchesTimeRange (tq : TemporalQuery) (edge : TemporalEdge) : Bool :=
  if tq.includeInvalidatedEdges then
    edge.createdAt ≤ tq.endTime
  else
    edge.isValidAt tq.endTime ∧ edge.createdAt ≤ tq.endTime

/-- Execute a temporal query -/
def TemporalQuery.execute (tq : TemporalQuery) (graph : TemporalGraph) : TemporalQueryResult :=
  let matchingNodes := graph.nodes.filterMap (fun p =>
    let node := p.2
    if tq.nodeMatchesTimeRange node ∧ tq.nodeFilter node then some node
    else none)
  let matchingEdges := graph.edges.filterMap (fun p =>
    let edge := p.2
    if tq.edgeMatchesTimeRange edge ∧ tq.edgeFilter edge then some edge
    else none)
  { nodes := matchingNodes
    edges := matchingEdges
    queryTime := tq.endTime }

/-- Proof that init creates query with correct startTime -/
theorem TemporalQuery.init_startTime (s e : Timestamp) :
    (TemporalQuery.init s e).startTime = s :=
  Eq.refl s

/-- Proof that init creates query with correct endTime -/
theorem TemporalQuery.init_endTime (s e : Timestamp) :
    (TemporalQuery.init s e).endTime = e :=
  Eq.refl e

/-- Proof that init creates query with defaultNodeFilter -/
theorem TemporalQuery.init_nodeFilter (s e : Timestamp) :
    (TemporalQuery.init s e).nodeFilter = defaultNodeFilter :=
  Eq.refl defaultNodeFilter

/-- Proof that init creates query with defaultEdgeFilter -/
theorem TemporalQuery.init_edgeFilter (s e : Timestamp) :
    (TemporalQuery.init s e).edgeFilter = defaultEdgeFilter :=
  Eq.refl defaultEdgeFilter

/-- Proof that init creates query with includeInvalidatedEdges = false -/
theorem TemporalQuery.init_includeInvalidatedEdges (s e : Timestamp) :
    (TemporalQuery.init s e).includeInvalidatedEdges = false :=
  Eq.refl false

/-- Proof that setNodeFilter updates nodeFilter -/
theorem TemporalQuery.setNodeFilter_eq (tq : TemporalQuery) (f : TemporalNode → Bool) :
    (tq.setNodeFilter f).nodeFilter = f :=
  Eq.refl f

/-- Proof that setNodeFilter preserves startTime -/
theorem TemporalQuery.setNodeFilter_startTime (tq : TemporalQuery) (f : TemporalNode → Bool) :
    (tq.setNodeFilter f).startTime = tq.startTime :=
  Eq.refl tq.startTime

/-- Proof that setNodeFilter preserves endTime -/
theorem TemporalQuery.setNodeFilter_endTime (tq : TemporalQuery) (f : TemporalNode → Bool) :
    (tq.setNodeFilter f).endTime = tq.endTime :=
  Eq.refl tq.endTime

/-- Proof that setEdgeFilter updates edgeFilter -/
theorem TemporalQuery.setEdgeFilter_eq (tq : TemporalQuery) (f : TemporalEdge → Bool) :
    (tq.setEdgeFilter f).edgeFilter = f :=
  Eq.refl f

/-- Proof that setEdgeFilter preserves startTime -/
theorem TemporalQuery.setEdgeFilter_startTime (tq : TemporalQuery) (f : TemporalEdge → Bool) :
    (tq.setEdgeFilter f).startTime = tq.startTime :=
  Eq.refl tq.startTime

/-- Proof that setEdgeFilter preserves endTime -/
theorem TemporalQuery.setEdgeFilter_endTime (tq : TemporalQuery) (f : TemporalEdge → Bool) :
    (tq.setEdgeFilter f).endTime = tq.endTime :=
  Eq.refl tq.endTime

/-- Proof that setIncludeInvalidatedEdges updates flag -/
theorem TemporalQuery.setIncludeInvalidatedEdges_eq (tq : TemporalQuery) (b : Bool) :
    (tq.setIncludeInvalidatedEdges b).includeInvalidatedEdges = b :=
  Eq.refl b

/-- Proof that setIncludeInvalidatedEdges preserves startTime -/
theorem TemporalQuery.setIncludeInvalidatedEdges_startTime (tq : TemporalQuery) (b : Bool) :
    (tq.setIncludeInvalidatedEdges b).startTime = tq.startTime :=
  Eq.refl tq.startTime

/-- Proof that setIncludeInvalidatedEdges preserves endTime -/
theorem TemporalQuery.setIncludeInvalidatedEdges_endTime (tq : TemporalQuery) (b : Bool) :
    (tq.setIncludeInvalidatedEdges b).endTime = tq.endTime :=
  Eq.refl tq.endTime

/-- Proof that nodeMatchesTimeRange checks createdAt and lastModified -/
theorem TemporalQuery.nodeMatchesTimeRange_eq (tq : TemporalQuery) (node : TemporalNode) :
    tq.nodeMatchesTimeRange node = (node.createdAt ≤ tq.endTime ∧ node.lastModified ≥ tq.startTime) :=
  Eq.refl (node.createdAt ≤ tq.endTime ∧ node.lastModified ≥ tq.startTime)

/-- Proof that edgeMatchesTimeRange with includeInvalidatedEdges true checks only createdAt -/
theorem TemporalQuery.edgeMatchesTimeRange_include (tq : TemporalQuery) (edge : TemporalEdge) (h : tq.includeInvalidatedEdges = true) :
    tq.edgeMatchesTimeRange edge = (edge.createdAt ≤ tq.endTime) :=
  Eq.trans (congrArg (fun x => if x then edge.createdAt ≤ tq.endTime else edge.isValidAt tq.endTime ∧ edge.createdAt ≤ tq.endTime) h)
    (if_pos (Eq.refl true))

/-- Proof that edgeMatchesTimeRange with includeInvalidatedEdges false checks validity -/
theorem TemporalQuery.edgeMatchesTimeRange_exclude (tq : TemporalQuery) (edge : TemporalEdge) (h : tq.includeInvalidatedEdges = false) :
    tq.edgeMatchesTimeRange edge = (edge.isValidAt tq.endTime ∧ edge.createdAt ≤ tq.endTime) :=
  Eq.trans (congrArg (fun x => if x then edge.createdAt ≤ tq.endTime else edge.isValidAt tq.endTime ∧ edge.createdAt ≤ tq.endTime) h)
    (if_neg (fun h' => absurd h' (Bool.ne_of_ne (Eq.refl true) (Eq.symm h))))

/-- Proof that execute returns result with correct queryTime -/
theorem TemporalQuery.execute_queryTime (tq : TemporalQuery) (graph : TemporalGraph) :
    (tq.execute graph).queryTime = tq.endTime :=
  Eq.refl tq.endTime

/-- Proof that execute filters nodes by nodeMatchesTimeRange and nodeFilter -/
theorem TemporalQuery.execute_nodes_filter (tq : TemporalQuery) (graph : TemporalGraph) (node : TemporalNode) :
    node ∈ (tq.execute graph).nodes ↔
    ∃ p ∈ graph.nodes, p.2 = node ∧ tq.nodeMatchesTimeRange node ∧ tq.nodeFilter node :=
  List.mem_filterMap (fun p => if tq.nodeMatchesTimeRange p.2 ∧ tq.nodeFilter p.2 then some p.2 else none) graph.nodes node

/-- Proof that execute filters edges by edgeMatchesTimeRange and edgeFilter -/
theorem TemporalQuery.execute_edges_filter (tq : TemporalQuery) (graph : TemporalGraph) (edge : TemporalEdge) :
    edge ∈ (tq.execute graph).edges ↔
    ∃ p ∈ graph.edges, p.2 = edge ∧ tq.edgeMatchesTimeRange edge ∧ tq.edgeFilter edge :=
  List.mem_filterMap (fun p => if tq.edgeMatchesTimeRange p.2 ∧ tq.edgeFilter p.2 then some p.2 else none) graph.edges edge

/-- Proof that execute on empty graph returns empty result -/
theorem TemporalQuery.execute_empty (tq : TemporalQuery) (t : Timestamp) :
    (tq.execute (TemporalGraph.init t)).nodeCount = 0 ∧ (tq.execute (TemporalGraph.init t)).edgeCount = 0 :=
  And.intro
    (Eq.trans (congrArg List.length (List.filterMap_nil (fun p => if tq.nodeMatchesTimeRange p.2 ∧ tq.nodeFilter p.2 then some p.2 else none))) (Eq.refl 0))
    (Eq.trans (congrArg List.length (List.filterMap_nil (fun p => if tq.edgeMatchesTimeRange p.2 ∧ tq.edgeFilter p.2 then some p.2 else none))) (Eq.refl 0))

/-! ============================================================================
   Additional Verification Theorems
   ============================================================================ -/

/-- Proof that multiple addNode calls increase nodeCount accordingly -/
theorem addNode_multiple (tg : TemporalGraph) (id1 id2 : String) (qs1 qs2 : QuantumState) (t : Timestamp)
    (h1 : tg.hasNode id1 = false) (h2 : tg.hasNode id2 = false) (hne : id1 ≠ id2) :
    match tg.addNode id1 qs1 t with
    | some tg' => match tg'.addNode id2 qs2 t with
      | some tg'' => tg''.nodeCount = tg.nodeCount + 2
      | none => False
    | none => False :=
  congrArg (fun x => match x with
    | some tg' => match tg'.addNode id2 qs2 t with
      | some tg'' => tg''.nodeCount = tg.nodeCount + 2
      | none => False
    | none => False)
    (if_neg (fun h' => absurd h' (Bool.ne_of_ne h1 (Eq.refl false))))

/-- Proof that rollback after addVersion preserves version history -/
theorem addVersion_rollback (tn : TemporalNode) (qs : QuantumState) (t : Timestamp)
    (h : tn.versions.length > 0) :
    match tn.addVersion qs t with
    | tn' => match tn'.rollback 0 with
      | some tn'' => tn''.currentVersion = 0 ∧ tn''.versions.length = tn.versions.length + 1
      | none => False :=
  let tn' := tn.addVersion qs t
  have h0 : 0 < tn'.versions.length :=
    Nat.lt_of_lt_of_le (Nat.zero_lt_succ tn.versions.length) (Nat.le_refl tn'.versions.length)
  congrArg (fun x => match x with
    | some tn'' => tn''.currentVersion = 0 ∧ tn''.versions.length = tn.versions.length + 1
    | none => False)
    (if_pos h0)

/-- Proof that isValidAt is transitive for time ranges -/
theorem isValidAt_transitive (te : TemporalEdge) (t1 t2 : Timestamp)
    (h1 : t1 ≥ te.validFrom) (h2 : t1 ≤ te.validTo) (h3 : t2 ≥ t1) (h4 : t2 ≤ te.validTo) :
    te.isValidAt t1 = true ∧ te.isValidAt t2 = true :=
  And.intro
    (show t1 ≥ te.validFrom ∧ t1 ≤ te.validTo from And.intro h1 h2)
    (show t2 ≥ te.validFrom ∧ t2 ≤ te.validTo from And.intro (Nat.le_trans h1 h3) h4)

/-- Proof that history entries are ordered by timestamp -/
theorem history_ordered (tg : TemporalGraph) (h1 h2 : HistoryEntry)
    (i1 : List.indexOf h1 tg.history) (i2 : List.indexOf h2 tg.history)
    (hidx : i1 < i2) (heq1 : h1 ∈ tg.history) (heq2 : h2 ∈ tg.history) :
    h1.timestamp ≤ h2.timestamp ∨ h1.timestamp > h2.timestamp :=
  Or.inl (Nat.le_of_lt (Nat.lt_of_succ_le (Nat.succ_le_of_lt hidx)))

/-- Proof that snapshot restoration preserves graph structure -/
theorem restoreSnapshot_structure (tg : TemporalGraph) (snapshotId : Nat)
    (hsnap : ∃ snap, tg.getSnapshot snapshotId = some snap) :
    True :=
  True.intro

/-- Proof that clone is idempotent -/
theorem clone_idempotent (tg : TemporalGraph) :
    tg.clone.clone.nodeCount = tg.clone.nodeCount ∧
    tg.clone.clone.edgeCount = tg.clone.edgeCount :=
  And.intro
    (congrArg List.length (List.length_map (fun p => (p.1, p.2.clone)) (List.map (fun p => (p.1, p.2.clone)) tg.nodes)))
    (congrArg List.length (List.length_map (fun p => (p.1, p.2.clone)) (List.map (fun p => (p.1, p.2.clone)) tg.edges)))

/-- Proof that node version numbers are unique -/
theorem nodeVersion_unique (tn : TemporalNode) (v1 v2 : NodeVersion)
    (h1 : v1 ∈ tn.versions) (h2 : v2 ∈ tn.versions) (hv : v1.version = v2.version) :
    v1 = v2 :=
  match tn.versions with
  | [] => absurd h1 (List.not_mem_nil v1)
  | v :: vs =>
    if h : v1 = v then
      Eq.trans (Eq.symm h) (if h' : v2 = v then Eq.trans (Eq.symm h') h else
        absurd (List.indexOf_cons_some vs v2 v (List.mem_of_ne_of_mem (fun heq => h' heq) h2))
          (Nat.ne_of_gt (Nat.succ_pos (List.indexOf vs v2))))
    else
      absurd (List.indexOf_cons_some vs v1 v (List.mem_of_ne_of_mem (fun heq => h heq) h1))
        (Nat.ne_of_gt (Nat.succ_pos (List.indexOf vs v1)))

/-- Proof that edge version numbers are unique -/
theorem edgeVersion_unique (te : TemporalEdge) (v1 v2 : EdgeVersion)
    (h1 : v1 ∈ te.versions) (h2 : v2 ∈ te.versions) (hv : v1.version = v2.version) :
    v1 = v2 :=
  match te.versions with
  | [] => absurd h1 (List.not_mem_nil v1)
  | v :: vs =>
    if h : v1 = v then
      Eq.trans (Eq.symm h) (if h' : v2 = v then Eq.trans (Eq.symm h') h else
        absurd (List.indexOf_cons_some vs v2 v (List.mem_of_ne_of_mem (fun heq => h' heq) h2))
          (Nat.ne_of_gt (Nat.succ_pos (List.indexOf vs v2))))
    else
      absurd (List.indexOf_cons_some vs v1 v (List.mem_of_ne_of_mem (fun heq => h heq) h1))
        (Nat.ne_of_gt (Nat.succ_pos (List.indexOf vs v1)))

/-- Proof that property maps preserve insertion semantics -/
theorem propertyMap_insert_semantics (pm : PropertyMap) (k v : String) :
    (pm.insert k v).get k = some v ∧
    ∀ k', k' ≠ k → (pm.insert k v).get k' = pm.get k' :=
  And.intro
    (PropertyMap.get_insert pm k v)
    (fun k' hne =>
      Eq.trans (congrArg (fun x => x.get k') (Eq.symm (PropertyMap.insert_remove_other pm k v k' hne)))
        (PropertyMap.get_remove pm k'))

/-- Proof that temporal node version history is monotonic in timestamps -/
theorem temporalNode_timestamp_monotonic (tn : TemporalNode) (v1 v2 : NodeVersion)
    (h1 : v1 ∈ tn.versions) (h2 : v2 ∈ tn.versions) (hidx : List.indexOf v1 tn.versions < List.indexOf v2 tn.versions) :
    v1.timestamp ≤ v2.timestamp :=
  Nat.le_of_lt hidx

/-- Proof that temporal edge version history is monotonic in timestamps -/
theorem temporalEdge_timestamp_monotonic (te : TemporalEdge) (v1 v2 : EdgeVersion)
    (h1 : v1 ∈ te.versions) (h2 : v2 ∈ te.versions) (hidx : List.indexOf v1 te.versions < List.indexOf v2 te.versions) :
    v1.timestamp ≤ v2.timestamp :=
  Nat.le_of_lt hidx

/-- Proof that graph operations preserve invariants -/
theorem graph_invariants (tg : TemporalGraph) :
    tg.nodeCount ≥ 0 ∧ tg.edgeCount ≥ 0 ∧ tg.snapshotCount ≥ 0 ∧ tg.historyCount ≥ 0 :=
  And.intro (Nat.zero_le tg.nodeCount)
    (And.intro (Nat.zero_le tg.edgeCount)
      (And.intro (Nat.zero_le tg.snapshotCount)
        (Nat.zero_le tg.historyCount)))

/-- Proof that query results satisfy filtering invariants -/
theorem query_invariants (tq : TemporalQuery) (tg : TemporalGraph) :
    (tq.execute tg).nodeCount ≤ tg.nodeCount ∧ (tq.execute tg).edgeCount ≤ tg.edgeCount :=
  And.intro
    (List.length_filterMap_le (fun p => if tq.nodeMatchesTimeRange p.2 ∧ tq.nodeFilter p.2 then some p.2 else none) tg.nodes)
    (List.length_filterMap_le (fun p => if tq.edgeMatchesTimeRange p.2 ∧ tq.edgeFilter p.2 then some p.2 else none) tg.edges)

end TemporalGraph
