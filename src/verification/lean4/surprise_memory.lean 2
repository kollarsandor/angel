import Init.Data.Nat.Basic
import Init.Data.List.Basic
import Init.Data.Array.Basic
import Init.Data.Option.Basic

namespace SurpriseMemory

def HASH_SIZE : Nat := 16
def HASH_BITS : Nat := HASH_SIZE * 8
def MAX_INPUT_SIZE : Nat := 100 * 1024 * 1024
def JACCARD_SAMPLE_SIZE : Nat := 1000
def MAX_ENTANGLEMENT_PAIRS : Nat := 100

def RETENTION_AGE_WEIGHT_NUM : Nat := 3
def RETENTION_AGE_WEIGHT_DEN : Nat := 10
def RETENTION_FREQUENCY_WEIGHT_NUM : Nat := 2
def RETENTION_FREQUENCY_WEIGHT_DEN : Nat := 10
def RETENTION_BASE_WEIGHT_NUM : Nat := 5
def RETENTION_BASE_WEIGHT_DEN : Nat := 10
def DEFAULT_SURPRISE_THRESHOLD_NUM : Nat := 3
def DEFAULT_SURPRISE_THRESHOLD_DEN : Nat := 10
def TEMPORAL_NOVELTY_WINDOW_MS : Nat := 24 * 60 * 60 * 1000
def BIGRAM_SPACE : Nat := 256 * 256
def HASH_WORD_COUNT : Nat := 4

structure Rational where
  num : Int
  den : Nat
  den_pos : den > 0
deriving Repr

def Rational.mk' (n : Int) (d : Nat) (h : d > 0) : Rational := { num := n, den := d, den_pos := h }

def Rational.zero : Rational := { num := 0, den := 1, den_pos := Nat.zero_lt_succ 0 }

def Rational.one : Rational := { num := 1, den := 1, den_pos := Nat.zero_lt_succ 0 }

def Rational.fromNat (n : Nat) : Rational := { num := Int.ofNat n, den := 1, den_pos := Nat.zero_lt_succ 0 }

def Rational.add (a b : Rational) : Rational :=
  { num := a.num * Int.ofNat b.den + b.num * Int.ofNat a.den
  , den := a.den * b.den
  , den_pos := Nat.mul_pos a.den_pos b.den_pos }

def Rational.sub (a b : Rational) : Rational :=
  { num := a.num * Int.ofNat b.den - b.num * Int.ofNat a.den
  , den := a.den * b.den
  , den_pos := Nat.mul_pos a.den_pos b.den_pos }

def Rational.mul (a b : Rational) : Rational :=
  { num := a.num * b.num
  , den := a.den * b.den
  , den_pos := Nat.mul_pos a.den_pos b.den_pos }

def Rational.le (a b : Rational) : Prop :=
  a.num * Int.ofNat b.den ≤ b.num * Int.ofNat a.den

def Rational.lt (a b : Rational) : Prop :=
  a.num * Int.ofNat b.den < b.num * Int.ofNat a.den

def Rational.gt (a b : Rational) : Prop :=
  Rational.lt b a

def Rational.isNonneg (r : Rational) : Prop := r.num ≥ 0

instance : BEq Rational where
  beq a b := (a.num * Int.ofNat b.den) == (b.num * Int.ofNat a.den)

def Rational.clamp (lo hi x : Rational) : Rational :=
  if decide (Rational.lt x lo) then lo
  else if decide (Rational.gt x hi) then hi
  else x

def BlockId := Fin HASH_SIZE → UInt8

instance : BEq BlockId where
  beq a b :=
    let rec go (i : Nat) : Bool :=
      if h : i < HASH_SIZE then
        if a ⟨i, h⟩ == b ⟨i, h⟩ then go (i + 1) else false
      else true
    go 0

noncomputable instance : DecidableEq BlockId := Classical.decEq _

instance : Hashable BlockId where
  hash a :=
    let rec go (i : Nat) (acc : UInt64) : UInt64 :=
      if h : i < HASH_SIZE then
        go (i + 1) (acc * 33 + (a ⟨i, h⟩).toUInt64)
      else acc
    go 0 5381

def zeroBlockId : BlockId := fun _ => 0
def allFFBlockId : BlockId := fun _ => 0xFF
def constBlockId (v : UInt8) : BlockId := fun _ => v

structure SurpriseMetrics where
  jaccard_dissimilarity : Rational
  content_hash_distance : Rational
  temporal_novelty : Rational
  combined_surprise : Rational
deriving Repr

def SurpriseMetrics.init (jaccard hash_dist temporal : Rational) : SurpriseMetrics :=
  let cj := Rational.clamp Rational.zero Rational.one jaccard
  let ch := Rational.clamp Rational.zero Rational.one hash_dist
  let ct := Rational.clamp Rational.zero Rational.one temporal
  let sum := Rational.add (Rational.add cj ch) ct
  let combined := { num := sum.num, den := sum.den * 3, den_pos := Nat.mul_pos sum.den_pos (Nat.zero_lt_succ 2) }
  { jaccard_dissimilarity := cj
  , content_hash_distance := ch
  , temporal_novelty := ct
  , combined_surprise := combined }

def SurpriseMetrics.exceedsThreshold (self : SurpriseMetrics) (threshold : Rational) : Bool :=
  decide (Rational.gt self.combined_surprise threshold)

structure SurpriseRecord where
  block_id : BlockId
  surprise_score : Rational
  creation_time : Int
  last_access_time : Int
  retention_priority : Rational
  access_frequency : Nat
deriving Repr

def retentionBase : Rational :=
  Rational.mk' (Int.ofNat RETENTION_BASE_WEIGHT_NUM) RETENTION_BASE_WEIGHT_DEN (Nat.zero_lt_succ 9)

def retentionAgeWeight : Rational :=
  Rational.mk' (Int.ofNat RETENTION_AGE_WEIGHT_NUM) RETENTION_AGE_WEIGHT_DEN (Nat.zero_lt_succ 9)

def retentionFrequencyWeight : Rational :=
  Rational.mk' (Int.ofNat RETENTION_FREQUENCY_WEIGHT_NUM) RETENTION_FREQUENCY_WEIGHT_DEN (Nat.zero_lt_succ 9)

def recencyFactorFromAgeMs (age_ms : Nat) : Rational :=
  Rational.mk' 1 (age_ms + 1) (Nat.succ_pos age_ms)

def frequencyFactorFromAccesses (n : Nat) : Rational :=
  let denom := n + 8
  Rational.mk' (Int.ofNat n) denom (Nat.succ_pos (n + 7))

def SurpriseRecord.init (block_id : BlockId) (score : Rational) (now : Int) : SurpriseRecord :=
  { block_id := block_id
  , surprise_score := score
  , creation_time := now
  , last_access_time := now
  , retention_priority := score
  , access_frequency := 1 }

def SurpriseRecord.updateRetention (self : SurpriseRecord) (now : Int) : SurpriseRecord :=
  let age_ns := now - self.last_access_time
  let nonneg_age_ns := if age_ns < 0 then 0 else age_ns.toNat
  let age_ms := nonneg_age_ns / 1000000
  let age_component := Rational.mul retentionAgeWeight (recencyFactorFromAgeMs age_ms)
  let freq_component := Rational.mul retentionFrequencyWeight (frequencyFactorFromAccesses self.access_frequency)
  let weight_sum := Rational.add (Rational.add retentionBase age_component) freq_component
  let new_priority := Rational.clamp Rational.zero Rational.one (Rational.mul self.surprise_score weight_sum)
  { self with
    retention_priority := new_priority
  , last_access_time := now }

def SurpriseRecord.recordAccess (self : SurpriseRecord) (now : Int) : SurpriseRecord :=
  let updated := { self with access_frequency := self.access_frequency + 1, last_access_time := now }
  let age_component := Rational.mul retentionAgeWeight (recencyFactorFromAgeMs 0)
  let freq_component := Rational.mul retentionFrequencyWeight (frequencyFactorFromAccesses updated.access_frequency)
  let weight_sum := Rational.add (Rational.add retentionBase age_component) freq_component
  let new_priority := Rational.clamp Rational.zero Rational.one (Rational.mul updated.surprise_score weight_sum)
  { updated with retention_priority := new_priority }

def SurpriseRecord.getRetentionPriority (self : SurpriseRecord) : Rational :=
  self.retention_priority

def SurpriseRecord.getAccessFrequency (self : SurpriseRecord) : Nat :=
  self.access_frequency

structure SurpriseMemoryStatistics where
  total_blocks : Nat
  high_surprise_blocks : Nat
  low_surprise_blocks : Nat
  average_surprise_num : Int
  average_surprise_den : Nat
  surprise_threshold : Rational
  evictions_due_to_low_surprise : Nat
  novel_block_allocations : Nat
  total_surprise_sum : Rational
deriving Repr

def SurpriseMemoryStatistics.init (threshold : Rational) : SurpriseMemoryStatistics :=
  { total_blocks := 0
  , high_surprise_blocks := 0
  , low_surprise_blocks := 0
  , average_surprise_num := 0
  , average_surprise_den := 1
  , surprise_threshold := threshold
  , evictions_due_to_low_surprise := 0
  , novel_block_allocations := 0
  , total_surprise_sum := Rational.zero }

def SurpriseMemoryStatistics.recalculateAverage (self : SurpriseMemoryStatistics) : SurpriseMemoryStatistics :=
  if self.total_blocks > 0 then
    { self with
      average_surprise_num := self.total_surprise_sum.num
    , average_surprise_den := self.total_surprise_sum.den * self.total_blocks }
  else
    { self with
      average_surprise_num := 0
    , average_surprise_den := 1
    , total_surprise_sum := Rational.zero }

def SurpriseMemoryStatistics.addBlock (self : SurpriseMemoryStatistics) (surprise_score : Rational) (threshold : Rational) : SurpriseMemoryStatistics :=
  let s1 := { self with
    total_blocks := self.total_blocks + 1
  , total_surprise_sum := Rational.add self.total_surprise_sum surprise_score }
  let s2 := if decide (Rational.gt surprise_score threshold) then
    { s1 with high_surprise_blocks := s1.high_surprise_blocks + 1 }
  else
    { s1 with low_surprise_blocks := s1.low_surprise_blocks + 1 }
  s2.recalculateAverage

def SurpriseMemoryStatistics.removeBlock (self : SurpriseMemoryStatistics) (surprise_score : Rational) (threshold : Rational) : SurpriseMemoryStatistics :=
  if self.total_blocks > 0 then
    let new_sum := Rational.sub self.total_surprise_sum surprise_score
    let clamped_sum := if decide (Rational.lt new_sum Rational.zero) then Rational.zero else new_sum
    let s1 := { self with total_blocks := self.total_blocks - 1, total_surprise_sum := clamped_sum }
    let s2 := if decide (Rational.gt surprise_score threshold) then
      { s1 with high_surprise_blocks := if s1.high_surprise_blocks > 0 then s1.high_surprise_blocks - 1 else 0 }
    else
      { s1 with low_surprise_blocks := if s1.low_surprise_blocks > 0 then s1.low_surprise_blocks - 1 else 0 }
    s2.recalculateAverage
  else
    self

structure CandidateItem where
  block_id : BlockId
  priority : Rational
deriving Repr

def popCount8 (b : UInt8) : Nat :=
  let b0 := b.toNat
  let rec count (bit : Nat) (acc : Nat) : Nat :=
    if bit < 8 then
      count (bit + 1) (acc + ((b0 >>> bit) &&& 1))
    else acc
  count 0 0

def computeHashDistance (hash_a hash_b : BlockId) : Rational :=
  let rec go (i : Nat) (dist : Nat) : Nat :=
    if h : i < HASH_SIZE then
      go (i + 1) (dist + popCount8 (hash_a ⟨i, h⟩ ^^^ hash_b ⟨i, h⟩))
    else dist
  let hamming := go 0 0
  { num := Int.ofNat hamming, den := HASH_BITS, den_pos := Nat.zero_lt_succ 127 }

def u32rotr (x : UInt32) (n : Nat) : UInt32 :=
  let shift := n % 32
  if shift = 0 then x
  else (x >>> UInt32.ofNat shift) ||| (x <<< UInt32.ofNat (32 - shift))

def shaCh (x y z : UInt32) : UInt32 := (x &&& y) ^^^ ((~~~x) &&& z)

def shaMaj (x y z : UInt32) : UInt32 := (x &&& y) ^^^ (x &&& z) ^^^ (y &&& z)

def shaBigSigma0 (x : UInt32) : UInt32 := u32rotr x 2 ^^^ u32rotr x 13 ^^^ u32rotr x 22

def shaBigSigma1 (x : UInt32) : UInt32 := u32rotr x 6 ^^^ u32rotr x 11 ^^^ u32rotr x 25

def shaSmallSigma0 (x : UInt32) : UInt32 := u32rotr x 7 ^^^ u32rotr x 18 ^^^ (x >>> UInt32.ofNat 3)

def shaSmallSigma1 (x : UInt32) : UInt32 := u32rotr x 17 ^^^ u32rotr x 19 ^^^ (x >>> UInt32.ofNat 10)

def sha256Init : Array UInt32 :=
  #[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

def sha256K : Array UInt32 :=
  #[0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

def beWord32 (b0 b1 b2 b3 : UInt8) : UInt32 :=
  (UInt32.ofNat b0.toNat <<< 24) ||| (UInt32.ofNat b1.toNat <<< 16) ||| (UInt32.ofNat b2.toNat <<< 8) ||| UInt32.ofNat b3.toNat

def word32ToBytes (x : UInt32) : List UInt8 :=
  [((x >>> UInt32.ofNat 24) &&& 0xFF).toUInt8,
   ((x >>> UInt32.ofNat 16) &&& 0xFF).toUInt8,
   ((x >>> UInt32.ofNat 8) &&& 0xFF).toUInt8,
   (x &&& 0xFF).toUInt8]

def word64ToBytes (x : UInt64) : List UInt8 :=
  [((x >>> UInt64.ofNat 56) &&& 0xFF).toUInt8,
   ((x >>> UInt64.ofNat 48) &&& 0xFF).toUInt8,
   ((x >>> UInt64.ofNat 40) &&& 0xFF).toUInt8,
   ((x >>> UInt64.ofNat 32) &&& 0xFF).toUInt8,
   ((x >>> UInt64.ofNat 24) &&& 0xFF).toUInt8,
   ((x >>> UInt64.ofNat 16) &&& 0xFF).toUInt8,
   ((x >>> UInt64.ofNat 8) &&& 0xFF).toUInt8,
   (x &&& 0xFF).toUInt8]

def padSha256 (data : List UInt8) : Array UInt8 :=
  let bitLen : UInt64 := UInt64.ofNat data.length * 8
  let withOne := data ++ [0x80]
  let modLen := withOne.length % 64
  let zeroPad := if modLen ≤ 56 then 56 - modLen else 56 + (64 - modLen)
  let padded := withOne ++ List.replicate zeroPad 0 ++ word64ToBytes bitLen
  padded.toArray

def sha256Schedule (chunk : Array UInt8) : Array UInt32 := Id.run do
  let mut w : Array UInt32 := Array.replicate 64 0
  for i in [0:16] do
    let base := i * 4
    let wv := beWord32 (chunk.getD base 0) (chunk.getD (base + 1) 0) (chunk.getD (base + 2) 0) (chunk.getD (base + 3) 0)
    w := w.set! i wv
  for i in [16:64] do
    let s0 := shaSmallSigma0 (w.get! (i - 15))
    let s1 := shaSmallSigma1 (w.get! (i - 2))
    let next := w.get! (i - 16) + s0 + w.get! (i - 7) + s1
    w := w.set! i next
  pure w

def sha256Compress (state : Array UInt32) (chunk : Array UInt8) : Array UInt32 := Id.run do
  let w := sha256Schedule chunk
  let mut a := state.get! 0
  let mut b := state.get! 1
  let mut c := state.get! 2
  let mut d := state.get! 3
  let mut e := state.get! 4
  let mut f := state.get! 5
  let mut g := state.get! 6
  let mut h := state.get! 7
  for i in [0:64] do
    let t1 := h + shaBigSigma1 e + shaCh e f g + sha256K.get! i + w.get! i
    let t2 := shaBigSigma0 a + shaMaj a b c
    h := g
    g := f
    f := e
    e := d + t1
    d := c
    c := b
    b := a
    a := t1 + t2
  let mut out := state
  out := out.set! 0 (out.get! 0 + a)
  out := out.set! 1 (out.get! 1 + b)
  out := out.set! 2 (out.get! 2 + c)
  out := out.set! 3 (out.get! 3 + d)
  out := out.set! 4 (out.get! 4 + e)
  out := out.set! 5 (out.get! 5 + f)
  out := out.set! 6 (out.get! 6 + g)
  out := out.set! 7 (out.get! 7 + h)
  pure out

def sha256Bytes (data : List UInt8) : List UInt8 :=
  let padded := padSha256 data
  let chunkCount := padded.size / 64
  let finalState := Id.run do
    let mut state := sha256Init
    for i in [0:chunkCount] do
      let chunk := (List.range 64).map (fun j => padded.getD (i * 64 + j) 0) |>.toArray
      state := sha256Compress state chunk
    pure state
  finalState.toList.bind word32ToBytes

def computeContentHash (data : List UInt8) : BlockId :=
  let digest := sha256Bytes data
  fun (i : Fin HASH_SIZE) => digest.getD i.val 0

def bigramKey (a b : UInt8) : Nat := a.toNat * 256 + b.toNat

def dedupNatList (xs : List Nat) : List Nat :=
  xs.eraseDups

def sampledBigramKeys (data : List UInt8) : List Nat :=
  let clipped := data.take MAX_INPUT_SIZE
  let rec pairs (remaining : List UInt8) (acc : List Nat) : List Nat :=
    match remaining with
    | a :: b :: tl => pairs (b :: tl) (bigramKey a b :: acc)
    | _ => acc.reverse
  let raw := pairs clipped []
  let sampleCount := min raw.length JACCARD_SAMPLE_SIZE
  dedupNatList (raw.take sampleCount)

def computeJaccardDistanceNat (data_a data_b : List UInt8) : Rational :=
  let set_a := sampledBigramKeys data_a
  let set_b := sampledBigramKeys data_b
  let intersection_count := (set_a.filter (fun x => set_b.contains x)).length
  let union_count := set_a.length + set_b.length - intersection_count
  if union_count == 0 then Rational.zero
  else
    let similarity := Rational.mk' (Int.ofNat intersection_count) union_count (Nat.succ_pos _)
    Rational.sub Rational.one similarity

inductive StoreError where
  | inputTooLarge
  | storageError
  | notFound
deriving Repr, BEq

structure StorageBlock where
  block_id : BlockId
  data : List UInt8
  content_hash : BlockId
deriving Repr

structure StorageState where
  blocks : List StorageBlock
  capacity : Nat
deriving Repr

def StorageState.empty (capacity : Nat) : StorageState :=
  { blocks := [], capacity := capacity }

def StorageState.count (st : StorageState) : Nat :=
  st.blocks.length

def StorageState.containsBlock (st : StorageState) (bid : BlockId) : Bool :=
  st.blocks.any (fun b => b.block_id == bid)

def StorageState.findBlock (st : StorageState) (bid : BlockId) : Option StorageBlock :=
  st.blocks.find? (fun b => b.block_id == bid)

def StorageState.findByContent (st : StorageState) (data : List UInt8) : Option BlockId :=
  (st.blocks.find? (fun b => b.data == data)).map (fun b => b.block_id)

def StorageState.store (st : StorageState) (data : List UInt8) : StorageState × BlockId :=
  match st.findByContent data with
  | some bid => (st, bid)
  | none =>
      let bid := computeContentHash data
      let blk : StorageBlock := { block_id := bid, data := data, content_hash := bid }
      let baseBlocks := if st.capacity = 0 then [] else blk :: st.blocks
      let trimmed := baseBlocks.take st.capacity
      ({ st with blocks := trimmed }, bid)

def StorageState.removeBlock (st : StorageState) (bid : BlockId) : StorageState :=
  { st with blocks := st.blocks.filter (fun b => !(b.block_id == bid)) }

structure ManagerState where
  storage : StorageState
  surprise_records : List (BlockId × SurpriseRecord)
  surprise_threshold : Rational
  statistics : SurpriseMemoryStatistics
deriving Repr

def defaultThreshold : Rational :=
  Rational.mk' (Int.ofNat DEFAULT_SURPRISE_THRESHOLD_NUM) DEFAULT_SURPRISE_THRESHOLD_DEN (Nat.zero_lt_succ 9)

def ManagerState.init (capacity : Nat) : ManagerState :=
  { storage := StorageState.empty capacity
  , surprise_records := []
  , surprise_threshold := defaultThreshold
  , statistics := SurpriseMemoryStatistics.init defaultThreshold }

def ManagerState.findRecord (st : ManagerState) (bid : BlockId) : Option SurpriseRecord :=
  match st.surprise_records.find? (fun p => p.1 == bid) with
  | some (_, r) => some r
  | none => none

def ManagerState.containsRecord (st : ManagerState) (bid : BlockId) : Bool :=
  st.surprise_records.any (fun p => p.1 == bid)

def ManagerState.recordCount (st : ManagerState) : Nat :=
  st.surprise_records.length

def ManagerState.putRecord (st : ManagerState) (bid : BlockId) (rec : SurpriseRecord) : ManagerState :=
  let filtered := st.surprise_records.filter (fun p => !(p.1 == bid))
  { st with surprise_records := (bid, rec) :: filtered }

def ManagerState.removeRecord (st : ManagerState) (bid : BlockId) : ManagerState :=
  { st with surprise_records := st.surprise_records.filter (fun p => !(p.1 == bid)) }

def ManagerState.refreshRetention (st : ManagerState) (now : Int) : ManagerState :=
  { st with surprise_records := st.surprise_records.map (fun (bid, rec) => (bid, rec.updateRetention now)) }

def ManagerState.recomputeStatistics (st : ManagerState) : ManagerState :=
  let stats := st.surprise_records.foldl (fun acc (_, rec) =>
    let acc1 := acc.addBlock rec.surprise_score st.surprise_threshold
    if decide (Rational.gt rec.surprise_score st.surprise_threshold) then
      { acc1 with novel_block_allocations := acc1.novel_block_allocations + 1 }
    else acc1
  ) (SurpriseMemoryStatistics.init st.surprise_threshold)
  { st with statistics := stats }

def ManagerState.latestAccessTime (st : ManagerState) : Option Int :=
  st.surprise_records.foldl (fun acc (_, rec) =>
    match acc with
    | none => some rec.last_access_time
    | some current => some (if rec.last_access_time > current then rec.last_access_time else current)
  ) none

def temporalNoveltyForState (st : ManagerState) (now : Int) : Rational :=
  match st.latestAccessTime with
  | none => Rational.one
  | some last =>
      let age := if now > last then now - last else 0
      let delta_ms := if age < 0 then 0 else age.toNat / 1000000
      let clipped := min delta_ms TEMPORAL_NOVELTY_WINDOW_MS
      Rational.mk' (Int.ofNat clipped) TEMPORAL_NOVELTY_WINDOW_MS (Nat.succ_pos _)

def ManagerState.setSurpriseThreshold (st : ManagerState) (threshold : Rational) : ManagerState :=
  let clamped := Rational.clamp Rational.zero Rational.one threshold
  (ManagerState.recomputeStatistics { st with surprise_threshold := clamped, statistics := { st.statistics with surprise_threshold := clamped } })

def computeSurpriseForState (st : ManagerState) (new_data : List UInt8) (now : Int) : Except StoreError SurpriseMetrics :=
  if new_data.length > MAX_INPUT_SIZE then
    Except.error StoreError.inputTooLarge
  else
    let new_hash := computeContentHash new_data
    let existing_bid := st.storage.findByContent new_data
    let sampledBlocks := st.storage.blocks.filter (fun blk =>
      match existing_bid with
      | some bid => !(blk.block_id == bid)
      | none => true)
    let block_count := sampledBlocks.length
    if block_count == 0 then
      Except.ok (SurpriseMetrics.init Rational.one Rational.one Rational.one)
    else
      let max_samples := min block_count JACCARD_SAMPLE_SIZE
      let sampled := sampledBlocks.take max_samples
      let result := sampled.foldl (fun (acc : Rational × Rational) blk =>
        let jd := computeJaccardDistanceNat new_data blk.data
        let hd := computeHashDistance new_hash blk.content_hash
        let new_j := if decide (Rational.lt jd acc.1) then jd else acc.1
        let new_h := if decide (Rational.lt hd acc.2) then hd else acc.2
        (new_j, new_h)
      ) (Rational.one, Rational.one)
      let temporal_novelty := temporalNoveltyForState st now
      Except.ok (SurpriseMetrics.init result.1 result.2 temporal_novelty)

def storeWithSurprise (st : ManagerState) (data : List UInt8) (now : Int) : Except StoreError (ManagerState × BlockId) :=
  if data.length > MAX_INPUT_SIZE then
    Except.error StoreError.inputTooLarge
  else
    match st.storage.findByContent data with
    | some bid =>
        match st.findRecord bid with
        | some rec =>
            let updated_rec := rec.recordAccess now
            let st' := (st.putRecord bid updated_rec).recomputeStatistics
            Except.ok (st', bid)
        | none =>
            match computeSurpriseForState st data now with
            | Except.error e => Except.error e
            | Except.ok surprise =>
                let new_rec := SurpriseRecord.init bid surprise.combined_surprise now
                let st' := (st.putRecord bid new_rec).recomputeStatistics
                Except.ok (st', bid)
    | none =>
        match computeSurpriseForState st data now with
        | Except.error e => Except.error e
        | Except.ok surprise =>
            let (storage', bid) := st.storage.store data
            let new_rec := SurpriseRecord.init bid surprise.combined_surprise now
            let st' := { st with storage := storage' }
            let st'' := (st'.putRecord bid new_rec).recomputeStatistics
            Except.ok (st'', bid)

def partialSortCandidates (items : List CandidateItem) (k : Nat) : List CandidateItem :=
  let sorted := items.mergeSort (fun a b => decide (Rational.lt a.priority b.priority))
  sorted.take k ++ (sorted.drop k)

def evictLowSurpriseBlocks (st : ManagerState) (target_capacity : Nat) : ManagerState × Nat :=
  let refreshed := st.refreshRetention 0
  let current_size := refreshed.storage.count
  if current_size ≤ target_capacity then (refreshed, 0)
  else
    let to_evict := current_size - target_capacity
    let candidates : List CandidateItem := refreshed.storage.blocks.map (fun blk =>
      let priority := match refreshed.findRecord blk.block_id with
        | some rec => rec.retention_priority
        | none => Rational.zero
      { block_id := blk.block_id, priority := priority })
    if candidates.length == 0 then (refreshed, 0)
    else
      let sorted := partialSortCandidates candidates to_evict
      let k := min to_evict sorted.length
      let to_remove := sorted.take k
      let result := to_remove.foldl (fun (acc : ManagerState × Nat) cand =>
        let (s, count) := acc
        if s.storage.containsBlock cand.block_id then
          let score := match s.findRecord cand.block_id with
            | some rec => rec.surprise_score
            | none => Rational.zero
          let s1 := { s with
            storage := s.storage.removeBlock cand.block_id
          , statistics := s.statistics.removeBlock score s.surprise_threshold }
          let s2 := s1.removeRecord cand.block_id
          let s3 := { s2 with statistics := { s2.statistics with evictions_due_to_low_surprise := s2.statistics.evictions_due_to_low_surprise + 1 } }
          (s3, count + 1)
        else (s, count)
      ) (refreshed, 0)
      ((result.1).recomputeStatistics, result.2)

theorem rational_zero_num : Rational.zero.num = 0 := Eq.refl _
theorem rational_zero_den : Rational.zero.den = 1 := Eq.refl _
theorem rational_one_num : Rational.one.num = 1 := Eq.refl _
theorem rational_one_den : Rational.one.den = 1 := Eq.refl _

theorem rational_add_zero_left (r : Rational) :
    (Rational.add Rational.zero r).num = r.num :=
  show 0 * Int.ofNat r.den + r.num * Int.ofNat 1 = r.num from
    have h1 : (0 : Int) * Int.ofNat r.den = 0 := Int.zero_mul _
    have h2 : r.num * Int.ofNat 1 = r.num := Int.mul_one r.num
    have h3 : (0 : Int) + r.num = r.num := Int.zero_add r.num
    Eq.trans (congrArg (· + r.num * Int.ofNat 1) h1) (Eq.trans (congrArg (0 + ·) h2) h3)

theorem rational_add_zero_right (r : Rational) :
    (Rational.add r Rational.zero).num = r.num :=
  show r.num * Int.ofNat 1 + 0 * Int.ofNat r.den = r.num from
    have h1 : r.num * Int.ofNat 1 = r.num := Int.mul_one r.num
    have h2 : (0 : Int) * Int.ofNat r.den = 0 := Int.zero_mul _
    have h3 : r.num + (0 : Int) = r.num := Int.add_zero r.num
    Eq.trans (congrArg (r.num * Int.ofNat 1 + ·) h2) (Eq.trans (congrArg (· + 0) h1) h3)

theorem rational_mul_one_num (r : Rational) :
    (Rational.mul r Rational.one).num = r.num :=
  show r.num * 1 = r.num from Int.mul_one r.num

theorem rational_mul_one_den (r : Rational) :
    (Rational.mul r Rational.one).den = r.den :=
  show r.den * 1 = r.den from Nat.mul_one r.den

theorem rational_mul_zero_num (r : Rational) :
    (Rational.mul r Rational.zero).num = 0 :=
  show r.num * 0 = 0 from Int.mul_zero r.num

theorem rational_add_comm_num (a b : Rational) :
    (Rational.add a b).num = (Rational.add b a).num :=
  show a.num * Int.ofNat b.den + b.num * Int.ofNat a.den =
       b.num * Int.ofNat a.den + a.num * Int.ofNat b.den from
    Int.add_comm (a.num * Int.ofNat b.den) (b.num * Int.ofNat a.den)

theorem rational_add_comm_den (a b : Rational) :
    (Rational.add a b).den = (Rational.add b a).den :=
  show a.den * b.den = b.den * a.den from Nat.mul_comm a.den b.den

theorem rational_mul_comm_num (a b : Rational) :
    (Rational.mul a b).num = (Rational.mul b a).num :=
  show a.num * b.num = b.num * a.num from Int.mul_comm a.num b.num

theorem rational_mul_comm_den (a b : Rational) :
    (Rational.mul a b).den = (Rational.mul b a).den :=
  show a.den * b.den = b.den * a.den from Nat.mul_comm a.den b.den

theorem rational_clamp_identity (r : Rational)
    (h_lo : Rational.le Rational.zero r)
    (h_hi : Rational.le r Rational.one) :
    ¬decide (Rational.lt r Rational.zero) = true →
    ¬decide (Rational.gt r Rational.one) = true →
    Rational.clamp Rational.zero Rational.one r = r :=
  fun h_nlt h_ngt =>
    show (if decide (Rational.lt r Rational.zero) then Rational.zero
          else if decide (Rational.gt r Rational.one) then Rational.one
          else r) = r from
      have h1 : decide (Rational.lt r Rational.zero) = false :=
        match decide (Rational.lt r Rational.zero) with
        | true => absurd (Eq.refl true) h_nlt
        | false => Eq.refl false
      have h2 : decide (Rational.gt r Rational.one) = false :=
        match decide (Rational.gt r Rational.one) with
        | true => absurd (Eq.refl true) h_ngt
        | false => Eq.refl false
      show (if decide (Rational.lt r Rational.zero) then Rational.zero
            else if decide (Rational.gt r Rational.one) then Rational.one
            else r) = r from
        Eq.mpr (congrArg (fun b => (if b then Rational.zero else if decide (Rational.gt r Rational.one) then Rational.one else r) = r) h1 ▸ Eq.refl _)
          (Eq.mpr (congrArg (fun b => (if b then Rational.one else r) = r) h2 ▸ Eq.refl _)
            (Eq.refl r))

theorem rational_clamp_low (r lo hi : Rational)
    (h : decide (Rational.lt r lo) = true) :
    Rational.clamp lo hi r = lo :=
  show (if decide (Rational.lt r lo) then lo
        else if decide (Rational.gt r hi) then hi
        else r) = lo from
    Eq.mpr (congrArg (fun b => (if b then lo else if decide (Rational.gt r hi) then hi else r) = lo) h ▸ Eq.refl _)
      (Eq.refl lo)

theorem rational_clamp_high (r lo hi : Rational)
    (h_nlt : decide (Rational.lt r lo) = false)
    (h_gt : decide (Rational.gt r hi) = true) :
    Rational.clamp lo hi r = hi :=
  show (if decide (Rational.lt r lo) then lo
        else if decide (Rational.gt r hi) then hi
        else r) = hi from
    Eq.mpr (congrArg (fun b => (if b then lo else if decide (Rational.gt r hi) then hi else r) = hi) h_nlt ▸ Eq.refl _)
      (Eq.mpr (congrArg (fun b => (if b then hi else r) = hi) h_gt ▸ Eq.refl _)
        (Eq.refl hi))

theorem surpriseMetrics_init_jaccard (j h t : Rational) :
    (SurpriseMetrics.init j h t).jaccard_dissimilarity = Rational.clamp Rational.zero Rational.one j :=
  Eq.refl _

theorem surpriseMetrics_init_hash_dist (j h t : Rational) :
    (SurpriseMetrics.init j h t).content_hash_distance = Rational.clamp Rational.zero Rational.one h :=
  Eq.refl _

theorem surpriseMetrics_init_temporal (j h t : Rational) :
    (SurpriseMetrics.init j h t).temporal_novelty = Rational.clamp Rational.zero Rational.one t :=
  Eq.refl _

theorem surpriseMetrics_init_combined_num (j h t : Rational) :
    (SurpriseMetrics.init j h t).combined_surprise.num =
    (Rational.add (Rational.add (Rational.clamp Rational.zero Rational.one j) (Rational.clamp Rational.zero Rational.one h)) (Rational.clamp Rational.zero Rational.one t)).num :=
  Eq.refl _

theorem surpriseMetrics_init_combined_den (j h t : Rational) :
    (SurpriseMetrics.init j h t).combined_surprise.den =
    (Rational.add (Rational.add (Rational.clamp Rational.zero Rational.one j) (Rational.clamp Rational.zero Rational.one h)) (Rational.clamp Rational.zero Rational.one t)).den * 3 :=
  Eq.refl _

theorem surpriseMetrics_exceedsThreshold_spec (m : SurpriseMetrics) (threshold : Rational) :
    m.exceedsThreshold threshold = decide (Rational.gt m.combined_surprise threshold) :=
  Eq.refl _

theorem surpriseRecord_init_block_id (bid : BlockId) (score : Rational) (now : Int) :
    (SurpriseRecord.init bid score now).block_id = bid :=
  Eq.refl _

theorem surpriseRecord_init_score (bid : BlockId) (score : Rational) (now : Int) :
    (SurpriseRecord.init bid score now).surprise_score = score :=
  Eq.refl _

theorem surpriseRecord_init_creation_time (bid : BlockId) (score : Rational) (now : Int) :
    (SurpriseRecord.init bid score now).creation_time = now :=
  Eq.refl _

theorem surpriseRecord_init_last_access (bid : BlockId) (score : Rational) (now : Int) :
    (SurpriseRecord.init bid score now).last_access_time = now :=
  Eq.refl _

theorem surpriseRecord_init_retention (bid : BlockId) (score : Rational) (now : Int) :
    (SurpriseRecord.init bid score now).retention_priority = score :=
  Eq.refl _

theorem surpriseRecord_init_frequency (bid : BlockId) (score : Rational) (now : Int) :
    (SurpriseRecord.init bid score now).access_frequency = 1 :=
  Eq.refl _

theorem surpriseRecord_init_all_fields (bid : BlockId) (score : Rational) (now : Int) :
    let r := SurpriseRecord.init bid score now
    r.block_id = bid ∧
    r.surprise_score = score ∧
    r.creation_time = now ∧
    r.last_access_time = now ∧
    r.retention_priority = score ∧
    r.access_frequency = 1 :=
  And.intro (Eq.refl _)
    (And.intro (Eq.refl _)
      (And.intro (Eq.refl _)
        (And.intro (Eq.refl _)
          (And.intro (Eq.refl _) (Eq.refl _)))))

theorem surpriseRecord_recordAccess_frequency (r : SurpriseRecord) (now : Int) :
    (r.recordAccess now).access_frequency = r.access_frequency + 1 :=
  Eq.refl _

theorem surpriseRecord_recordAccess_time (r : SurpriseRecord) (now : Int) :
    (r.recordAccess now).last_access_time = now :=
  Eq.refl _

theorem surpriseRecord_recordAccess_preserves_block_id (r : SurpriseRecord) (now : Int) :
    (r.recordAccess now).block_id = r.block_id :=
  Eq.refl _

theorem surpriseRecord_recordAccess_preserves_score (r : SurpriseRecord) (now : Int) :
    (r.recordAccess now).surprise_score = r.surprise_score :=
  Eq.refl _

theorem surpriseRecord_recordAccess_preserves_creation (r : SurpriseRecord) (now : Int) :
    (r.recordAccess now).creation_time = r.creation_time :=
  Eq.refl _

theorem surpriseRecord_getRetentionPriority_eq (r : SurpriseRecord) :
    r.getRetentionPriority = r.retention_priority :=
  Eq.refl _

theorem surpriseRecord_getAccessFrequency_eq (r : SurpriseRecord) :
    r.getAccessFrequency = r.access_frequency :=
  Eq.refl _

theorem surpriseRecord_after_one_access (bid : BlockId) (score : Rational) (t0 t1 : Int) :
    let r := SurpriseRecord.init bid score t0
    (r.recordAccess t1).access_frequency = 2 :=
  Eq.refl _

theorem surpriseRecord_after_two_accesses (bid : BlockId) (score : Rational) (t0 t1 t2 : Int) :
    let r := SurpriseRecord.init bid score t0
    let r' := r.recordAccess t1
    (r'.recordAccess t2).access_frequency = 3 :=
  Eq.refl _

theorem surpriseRecord_after_three_accesses (bid : BlockId) (score : Rational) (t0 t1 t2 t3 : Int) :
    let r := SurpriseRecord.init bid score t0
    ((r.recordAccess t1).recordAccess t2).recordAccess t3 |>.access_frequency = 4 :=
  Eq.refl _

theorem surpriseRecord_access_monotone (r : SurpriseRecord) (now : Int) :
    (r.recordAccess now).access_frequency > r.access_frequency :=
  Nat.lt_succ_of_le (Nat.le_refl _)

theorem statistics_init_total_blocks (threshold : Rational) :
    (SurpriseMemoryStatistics.init threshold).total_blocks = 0 :=
  Eq.refl _

theorem statistics_init_high_blocks (threshold : Rational) :
    (SurpriseMemoryStatistics.init threshold).high_surprise_blocks = 0 :=
  Eq.refl _

theorem statistics_init_low_blocks (threshold : Rational) :
    (SurpriseMemoryStatistics.init threshold).low_surprise_blocks = 0 :=
  Eq.refl _

theorem statistics_init_evictions (threshold : Rational) :
    (SurpriseMemoryStatistics.init threshold).evictions_due_to_low_surprise = 0 :=
  Eq.refl _

theorem statistics_init_novel (threshold : Rational) :
    (SurpriseMemoryStatistics.init threshold).novel_block_allocations = 0 :=
  Eq.refl _

theorem statistics_init_sum_zero (threshold : Rational) :
    (SurpriseMemoryStatistics.init threshold).total_surprise_sum = Rational.zero :=
  Eq.refl _

theorem statistics_init_all_fields (threshold : Rational) :
    let s := SurpriseMemoryStatistics.init threshold
    s.total_blocks = 0 ∧
    s.high_surprise_blocks = 0 ∧
    s.low_surprise_blocks = 0 ∧
    s.evictions_due_to_low_surprise = 0 ∧
    s.novel_block_allocations = 0 ∧
    s.total_surprise_sum = Rational.zero :=
  And.intro (Eq.refl _)
    (And.intro (Eq.refl _)
      (And.intro (Eq.refl _)
        (And.intro (Eq.refl _)
          (And.intro (Eq.refl _) (Eq.refl _)))))

theorem statistics_addBlock_total (s : SurpriseMemoryStatistics) (score threshold : Rational) :
    (s.addBlock score threshold).total_blocks = s.total_blocks + 1 :=
  show (SurpriseMemoryStatistics.recalculateAverage
    (if decide (Rational.gt score threshold) then
      { { s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score } with
        high_surprise_blocks := ({ s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score }).high_surprise_blocks + 1 }
    else
      { { s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score } with
        low_surprise_blocks := ({ s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score }).low_surprise_blocks + 1 })).total_blocks = s.total_blocks + 1 from
    match decide (Rational.gt score threshold) with
    | true =>
      show (SurpriseMemoryStatistics.recalculateAverage
        { { s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score } with
          high_surprise_blocks := s.high_surprise_blocks + 1 }).total_blocks = s.total_blocks + 1 from
        show (if s.total_blocks + 1 > 0 then
              { { { s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score } with high_surprise_blocks := s.high_surprise_blocks + 1 } with
                average_surprise_num := (Rational.add s.total_surprise_sum score).num
              , average_surprise_den := (Rational.add s.total_surprise_sum score).den * (s.total_blocks + 1) }
            else _).total_blocks = s.total_blocks + 1 from
          Eq.refl _
    | false =>
      show (SurpriseMemoryStatistics.recalculateAverage
        { { s with total_blocks := s.total_blocks + 1, total_surprise_sum := Rational.add s.total_surprise_sum score } with
          low_surprise_blocks := s.low_surprise_blocks + 1 }).total_blocks = s.total_blocks + 1 from
        Eq.refl _

theorem statistics_addBlock_high (s : SurpriseMemoryStatistics) (score threshold : Rational)
    (h : decide (Rational.gt score threshold) = true) :
    (s.addBlock score threshold).high_surprise_blocks = s.high_surprise_blocks + 1 :=
  show (SurpriseMemoryStatistics.recalculateAverage
    (if decide (Rational.gt score threshold) then _ else _)).high_surprise_blocks = s.high_surprise_blocks + 1 from
    Eq.mpr (congrArg (fun b => (SurpriseMemoryStatistics.recalculateAverage (if b then _ else _)).high_surprise_blocks = s.high_surprise_blocks + 1) h ▸ Eq.refl _)
      (Eq.refl _)

theorem statistics_addBlock_low (s : SurpriseMemoryStatistics) (score threshold : Rational)
    (h : decide (Rational.gt score threshold) = false) :
    (s.addBlock score threshold).low_surprise_blocks = s.low_surprise_blocks + 1 :=
  show (SurpriseMemoryStatistics.recalculateAverage
    (if decide (Rational.gt score threshold) then _ else _)).low_surprise_blocks = s.low_surprise_blocks + 1 from
    Eq.mpr (congrArg (fun b => (SurpriseMemoryStatistics.recalculateAverage (if b then _ else _)).low_surprise_blocks = s.low_surprise_blocks + 1) h ▸ Eq.refl _)
      (Eq.refl _)

theorem statistics_addBlock_preserves_evictions (s : SurpriseMemoryStatistics) (score threshold : Rational) :
    (s.addBlock score threshold).evictions_due_to_low_surprise = s.evictions_due_to_low_surprise :=
  match decide (Rational.gt score threshold) with
  | true => Eq.refl _
  | false => Eq.refl _

theorem statistics_addBlock_preserves_novel (s : SurpriseMemoryStatistics) (score threshold : Rational) :
    (s.addBlock score threshold).novel_block_allocations = s.novel_block_allocations :=
  match decide (Rational.gt score threshold) with
  | true => Eq.refl _
  | false => Eq.refl _

theorem statistics_addBlock_sum (s : SurpriseMemoryStatistics) (score threshold : Rational) :
    (s.addBlock score threshold).total_surprise_sum = Rational.add s.total_surprise_sum score :=
  match decide (Rational.gt score threshold) with
  | true => Eq.refl _
  | false => Eq.refl _

theorem statistics_removeBlock_when_empty (s : SurpriseMemoryStatistics) (score threshold : Rational)
    (h : s.total_blocks = 0) :
    s.removeBlock score threshold = s :=
  show (if s.total_blocks > 0 then _ else s) = s from
    have h_not_pos : (s.total_blocks > 0) = false :=
      Eq.subst (motive := fun n => (n > 0) = false) (Eq.symm h) (Eq.refl _)
    Eq.mpr (congrArg (fun b => (if b then _ else s) = s) h_not_pos ▸ Eq.refl _) (Eq.refl s)

theorem statistics_removeBlock_total (s : SurpriseMemoryStatistics) (score threshold : Rational)
    (h : s.total_blocks > 0) :
    (s.removeBlock score threshold).total_blocks = s.total_blocks - 1 :=
  show (if s.total_blocks > 0 then _ else s).total_blocks = s.total_blocks - 1 from
    have h_pos : (s.total_blocks > 0) = true := Nat.ble_eq s.total_blocks 0 ▸ h ▸ Eq.refl _
    match decide (Rational.lt (Rational.sub s.total_surprise_sum score) Rational.zero),
          decide (Rational.gt score threshold) with
    | true, true => Eq.refl _
    | true, false => Eq.refl _
    | false, true => Eq.refl _
    | false, false => Eq.refl _

theorem statistics_add_then_remove_total (threshold score : Rational) :
    let s0 := SurpriseMemoryStatistics.init threshold
    let s1 := s0.addBlock score threshold
    (s1.removeBlock score threshold).total_blocks = 0 :=
  show ((SurpriseMemoryStatistics.init threshold).addBlock score threshold).removeBlock score threshold |>.total_blocks = 0 from
    have h1 : ((SurpriseMemoryStatistics.init threshold).addBlock score threshold).total_blocks = 1 :=
      statistics_addBlock_total (SurpriseMemoryStatistics.init threshold) score threshold
    match decide (Rational.gt score threshold),
          decide (Rational.lt (Rational.sub (Rational.add Rational.zero score) score) Rational.zero) with
    | true, true => Eq.refl _
    | true, false => Eq.refl _
    | false, true => Eq.refl _
    | false, false => Eq.refl _

theorem statistics_two_adds_total (threshold s1 s2 : Rational) :
    let st0 := SurpriseMemoryStatistics.init threshold
    let st1 := st0.addBlock s1 threshold
    (st1.addBlock s2 threshold).total_blocks = 2 :=
  show ((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).addBlock s2 threshold |>.total_blocks = 2 from
    have step1 : ((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).total_blocks = 1 :=
      statistics_addBlock_total _ _ _
    have step2 : (((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).addBlock s2 threshold).total_blocks =
                  ((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).total_blocks + 1 :=
      statistics_addBlock_total _ _ _
    Eq.trans step2 (congrArg (· + 1) step1)

theorem statistics_three_adds_total (threshold s1 s2 s3 : Rational) :
    let st0 := SurpriseMemoryStatistics.init threshold
    ((st0.addBlock s1 threshold).addBlock s2 threshold).addBlock s3 threshold |>.total_blocks = 3 :=
  have h2 : ((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).addBlock s2 threshold |>.total_blocks = 2 :=
    statistics_two_adds_total threshold s1 s2
  have h3 : (((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).addBlock s2 threshold).addBlock s3 threshold |>.total_blocks =
    ((SurpriseMemoryStatistics.init threshold).addBlock s1 threshold).addBlock s2 threshold |>.total_blocks + 1 :=
    statistics_addBlock_total _ _ _
  Eq.trans h3 (congrArg (· + 1) h2)

structure StatisticsInvariant (s : SurpriseMemoryStatistics) : Prop where
  partition_le : s.high_surprise_blocks + s.low_surprise_blocks ≤ s.total_blocks
  blocks_consistent : s.total_blocks = 0 → s.high_surprise_blocks = 0 ∧ s.low_surprise_blocks = 0

theorem statistics_invariant_init (threshold : Rational) :
    StatisticsInvariant (SurpriseMemoryStatistics.init threshold) :=
  { partition_le := Nat.zero_le 0
  , blocks_consistent := fun _ => And.intro (Eq.refl _) (Eq.refl _) }

theorem statistics_invariant_addBlock (s : SurpriseMemoryStatistics) (score threshold : Rational)
    (inv : StatisticsInvariant s) :
    StatisticsInvariant (s.addBlock score threshold) :=
  { partition_le :=
      match h : decide (Rational.gt score threshold) with
      | true =>
        show (s.high_surprise_blocks + 1) + s.low_surprise_blocks ≤ s.total_blocks + 1 from
          have h1 : s.high_surprise_blocks + s.low_surprise_blocks ≤ s.total_blocks := inv.partition_le
          have h2 : (s.high_surprise_blocks + 1) + s.low_surprise_blocks =
                     (s.high_surprise_blocks + s.low_surprise_blocks) + 1 :=
            Eq.trans (Nat.add_right_comm s.high_surprise_blocks 1 s.low_surprise_blocks) (Eq.refl _)
          Eq.subst (Eq.symm h2) (Nat.succ_le_succ h1)
      | false =>
        show s.high_surprise_blocks + (s.low_surprise_blocks + 1) ≤ s.total_blocks + 1 from
          have h1 : s.high_surprise_blocks + s.low_surprise_blocks ≤ s.total_blocks := inv.partition_le
          have h2 : s.high_surprise_blocks + (s.low_surprise_blocks + 1) =
                     (s.high_surprise_blocks + s.low_surprise_blocks) + 1 :=
            Nat.add_assoc s.high_surprise_blocks s.low_surprise_blocks 1
          Eq.subst (Eq.symm h2) (Nat.succ_le_succ h1)
  , blocks_consistent := fun h_zero =>
      absurd h_zero (Nat.not_eq_zero_of_lt (Nat.zero_lt_succ s.total_blocks)) }

def batchAddBlocks (st : SurpriseMemoryStatistics) (scores : List Rational) (threshold : Rational) : SurpriseMemoryStatistics :=
  scores.foldl (fun acc score => acc.addBlock score threshold) st

theorem batchAddBlocks_empty (st : SurpriseMemoryStatistics) (threshold : Rational) :
    batchAddBlocks st [] threshold = st :=
  Eq.refl _

theorem batchAddBlocks_singleton (st : SurpriseMemoryStatistics) (score threshold : Rational) :
    batchAddBlocks st [score] threshold = st.addBlock score threshold :=
  Eq.refl _

theorem batchAddBlocks_total_count (st : SurpriseMemoryStatistics) (scores : List Rational) (threshold : Rational) :
    (batchAddBlocks st scores threshold).total_blocks = st.total_blocks + scores.length :=
  List.recOn scores
    (show st.total_blocks = st.total_blocks + 0 from Eq.symm (Nat.add_zero st.total_blocks))
    (fun score rest ih =>
      show (batchAddBlocks (st.addBlock score threshold) rest threshold).total_blocks =
           st.total_blocks + (rest.length + 1) from
        have ih' : (batchAddBlocks (st.addBlock score threshold) rest threshold).total_blocks =
                    (st.addBlock score threshold).total_blocks + rest.length :=
          List.recOn rest
            (show (st.addBlock score threshold).total_blocks =
                  (st.addBlock score threshold).total_blocks + 0 from
              Eq.symm (Nat.add_zero _))
            (fun s2 rest2 ih2 =>
              show (batchAddBlocks ((st.addBlock score threshold).addBlock s2 threshold) rest2 threshold).total_blocks =
                   (st.addBlock score threshold).total_blocks + (rest2.length + 1) from
                have base_eq : ((st.addBlock score threshold).addBlock s2 threshold).total_blocks =
                               (st.addBlock score threshold).total_blocks + 1 :=
                  statistics_addBlock_total _ _ _
                have inner_ih : (batchAddBlocks ((st.addBlock score threshold).addBlock s2 threshold) rest2 threshold).total_blocks =
                                ((st.addBlock score threshold).addBlock s2 threshold).total_blocks + rest2.length :=
                  List.recOn rest2
                    (Eq.symm (Nat.add_zero _))
                    (fun _ _ _ =>
                      Eq.trans
                        (statistics_addBlock_total _ _ _)
                        (congrArg (· + 1) (Eq.symm (Nat.add_zero _) ▸ statistics_addBlock_total _ _ _) ▸ Eq.refl _))
                Eq.trans inner_ih (Eq.trans (congrArg (· + rest2.length) base_eq) (Eq.symm (Nat.add_assoc _ 1 _) ▸ Nat.add_right_comm _ 1 _)))
        have h_add_total : (st.addBlock score threshold).total_blocks = st.total_blocks + 1 :=
          statistics_addBlock_total st score threshold
        Eq.trans ih' (Eq.trans (congrArg (· + rest.length) h_add_total)
          (Eq.trans (Nat.add_assoc st.total_blocks 1 rest.length)
            (congrArg (st.total_blocks + ·) (Nat.add_comm 1 rest.length)))))

theorem storageState_empty_count (cap : Nat) :
    (StorageState.empty cap).count = 0 :=
  Eq.refl _

theorem storageState_empty_capacity (cap : Nat) :
    (StorageState.empty cap).capacity = cap :=
  Eq.refl _

theorem storageState_store_count (st : StorageState) (data : List UInt8) :
    (st.store data).1.count = st.count + 1 :=
  Eq.refl _

theorem storageState_store_bid (st : StorageState) (data : List UInt8) :
    (st.store data).2 = computeContentHash data :=
  Eq.refl _

theorem storageState_store_preserves_capacity (st : StorageState) (data : List UInt8) :
    (st.store data).1.capacity = st.capacity :=
  Eq.refl _

theorem storageState_store_block_is_head (st : StorageState) (data : List UInt8) :
    (st.store data).1.blocks = { block_id := computeContentHash data, data := data, content_hash := computeContentHash data } :: st.blocks :=
  Eq.refl _

theorem storageState_two_stores_count (st : StorageState) (d1 d2 : List UInt8) :
    ((st.store d1).1.store d2).1.count = st.count + 2 :=
  show ((st.store d1).1.store d2).1.blocks.length = st.blocks.length + 2 from
    have h1 : (st.store d1).1.count = st.count + 1 := storageState_store_count st d1
    have h2 : ((st.store d1).1.store d2).1.count = (st.store d1).1.count + 1 :=
      storageState_store_count (st.store d1).1 d2
    Eq.trans h2 (congrArg (· + 1) h1)

theorem storageState_three_stores_count (st : StorageState) (d1 d2 d3 : List UInt8) :
    (((st.store d1).1.store d2).1.store d3).1.count = st.count + 3 :=
  have h2 : ((st.store d1).1.store d2).1.count = st.count + 2 :=
    storageState_two_stores_count st d1 d2
  have h3 : (((st.store d1).1.store d2).1.store d3).1.count =
             ((st.store d1).1.store d2).1.count + 1 :=
    storageState_store_count _ d3
  Eq.trans h3 (congrArg (· + 1) h2)

theorem computeContentHash_deterministic (data : List UInt8) :
    computeContentHash data = computeContentHash data :=
  Eq.refl _

theorem computeHashDistance_self (h : BlockId) :
    let d := computeHashDistance h h
    d.num = 0 :=
  show (computeHashDistance h h).num = 0 from
    show Int.ofNat (
      let rec go (i : Nat) (dist : Nat) : Nat :=
        if hp : i < HASH_SIZE then go (i + 1) (dist + popCount8 (h ⟨i, hp⟩ ^^^ h ⟨i, hp⟩)) else dist
      go 0 0) = 0 from
    Eq.refl _

theorem computeHashDistance_den (a b : BlockId) :
    (computeHashDistance a b).den = HASH_BITS :=
  Eq.refl _

theorem managerState_init_storage_empty (cap : Nat) :
    (ManagerState.init cap).storage.count = 0 :=
  Eq.refl _

theorem managerState_init_no_records (cap : Nat) :
    (ManagerState.init cap).recordCount = 0 :=
  Eq.refl _

theorem managerState_init_threshold (cap : Nat) :
    (ManagerState.init cap).surprise_threshold = defaultThreshold :=
  Eq.refl _

theorem managerState_init_stats_zero (cap : Nat) :
    (ManagerState.init cap).statistics.total_blocks = 0 :=
  Eq.refl _

theorem managerState_init_all (cap : Nat) :
    let m := ManagerState.init cap
    m.storage.count = 0 ∧
    m.recordCount = 0 ∧
    m.surprise_threshold = defaultThreshold ∧
    m.statistics.total_blocks = 0 ∧
    m.statistics.high_surprise_blocks = 0 ∧
    m.statistics.low_surprise_blocks = 0 :=
  And.intro (Eq.refl _)
    (And.intro (Eq.refl _)
      (And.intro (Eq.refl _)
        (And.intro (Eq.refl _)
          (And.intro (Eq.refl _) (Eq.refl _)))))

theorem managerState_setSurpriseThreshold_stats (st : ManagerState) (t : Rational) :
    (st.setSurpriseThreshold t).statistics.surprise_threshold =
    Rational.clamp Rational.zero Rational.one t :=
  let clamped := Rational.clamp Rational.zero Rational.one t
  let s := { st with surprise_threshold := clamped, statistics := { st.statistics with surprise_threshold := clamped } }
  Eq.refl _

theorem managerState_setSurpriseThreshold_threshold (st : ManagerState) (t : Rational) :
    (st.setSurpriseThreshold t).surprise_threshold =
    Rational.clamp Rational.zero Rational.one t :=
  Eq.refl _

theorem managerState_setSurpriseThreshold_preserves_storage (st : ManagerState) (t : Rational) :
    (st.setSurpriseThreshold t).storage = st.storage :=
  let clamped := Rational.clamp Rational.zero Rational.one t
  let s := { st with surprise_threshold := clamped, statistics := { st.statistics with surprise_threshold := clamped } }
  Eq.refl _

theorem managerState_setSurpriseThreshold_preserves_records (st : ManagerState) (t : Rational) :
    (st.setSurpriseThreshold t).surprise_records = st.surprise_records :=
  let clamped := Rational.clamp Rational.zero Rational.one t
  let s := { st with surprise_threshold := clamped, statistics := { st.statistics with surprise_threshold := clamped } }
  Eq.refl _

theorem managerState_putRecord_increases_or_preserves (st : ManagerState) (bid : BlockId) (rec : SurpriseRecord) :
    (st.putRecord bid rec).recordCount ≥ 1 :=
  show ((bid, rec) :: st.surprise_records.filter (fun p => !(p.1 == bid))).length ≥ 1 from
    Nat.succ_le_succ (Nat.zero_le _)

theorem managerState_putRecord_contains (st : ManagerState) (bid : BlockId) (rec : SurpriseRecord) :
    (st.putRecord bid rec).containsRecord bid = true :=
  show List.any ((bid, rec) :: st.surprise_records.filter (fun p => !(p.1 == bid))) (fun p => p.1 == bid) = true from
    Eq.refl _

theorem managerState_removeRecord_count_le (st : ManagerState) (bid : BlockId) :
    (st.removeRecord bid).recordCount ≤ st.recordCount :=
  show (st.surprise_records.filter (fun p => !(p.1 == bid))).length ≤ st.surprise_records.length from
    List.length_filter_le _ _

theorem computeSurprise_empty_storage (data : List UInt8) (h : data.length ≤ MAX_INPUT_SIZE) :
    computeSurpriseForState (ManagerState.init 1024) data 0 =
    Except.ok (SurpriseMetrics.init Rational.one Rational.one Rational.one) :=
  show (if data.length > MAX_INPUT_SIZE then Except.error StoreError.inputTooLarge else _) =
       Except.ok (SurpriseMetrics.init Rational.one Rational.one Rational.one) from
    have h1 : (data.length > MAX_INPUT_SIZE) = false := Nat.not_lt_of_le h
    Eq.refl _

theorem computeSurprise_inputTooLarge (st : ManagerState) (data : List UInt8)
    (h : data.length > MAX_INPUT_SIZE) :
    computeSurpriseForState st data 0 = Except.error StoreError.inputTooLarge :=
  show (if data.length > MAX_INPUT_SIZE then Except.error StoreError.inputTooLarge else _) =
       Except.error StoreError.inputTooLarge from
    have : (data.length > MAX_INPUT_SIZE) = true := h
    Eq.refl _

theorem storeWithSurprise_inputTooLarge (st : ManagerState) (data : List UInt8) (now : Int)
    (h : data.length > MAX_INPUT_SIZE) :
    storeWithSurprise st data now = Except.error StoreError.inputTooLarge :=
  show (if data.length > MAX_INPUT_SIZE then Except.error StoreError.inputTooLarge else _) =
       Except.error StoreError.inputTooLarge from
    have : (data.length > MAX_INPUT_SIZE) = true := h
    Eq.refl _

theorem evictLowSurprise_noop_under_capacity (st : ManagerState) (target : Nat)
    (h : st.storage.count ≤ target) :
    evictLowSurpriseBlocks st target = (st, 0) :=
  show (if st.storage.count ≤ target then (st, 0) else _) = (st, 0) from
    have : (st.storage.count ≤ target) = true := Nat.ble_eq_true_of_le h
    Eq.refl _

theorem evictLowSurprise_preserves_state_under_capacity (st : ManagerState) (target : Nat)
    (h : st.storage.count ≤ target) :
    (evictLowSurpriseBlocks st target).1 = st :=
  congrArg Prod.fst (evictLowSurprise_noop_under_capacity st target h)

theorem evictLowSurprise_zero_evictions_under_capacity (st : ManagerState) (target : Nat)
    (h : st.storage.count ≤ target) :
    (evictLowSurpriseBlocks st target).2 = 0 :=
  congrArg Prod.snd (evictLowSurprise_noop_under_capacity st target h)

private theorem Nat.ble_eq_true_of_le {a b : Nat} (h : a ≤ b) : (a ≤ b) = true :=
  propext ⟨fun _ => Eq.refl _, fun _ => h⟩

private theorem Nat.not_eq_zero_of_lt {a : Nat} (h : 0 < Nat.succ a) : Nat.succ a ≠ 0 :=
  fun h_eq => Nat.noConfusion h_eq

inductive MemoryBlockState where
  | active
  | evicted
  | entangled
  | archived
deriving Repr, BEq

structure MemoryBlock where
  block_id : BlockId
  state : MemoryBlockState
  data : List UInt8
  surprise_score : Rational
  creation_time : Int
deriving Repr

def MemoryBlock.isActive (b : MemoryBlock) : Bool :=
  b.state == MemoryBlockState.active

def MemoryBlock.isEvicted (b : MemoryBlock) : Bool :=
  b.state == MemoryBlockState.evicted

def MemoryBlock.isEntangled (b : MemoryBlock) : Bool :=
  b.state == MemoryBlockState.entangled

def MemoryBlock.isArchived (b : MemoryBlock) : Bool :=
  b.state == MemoryBlockState.archived

theorem memoryBlock_active_not_evicted (b : MemoryBlock) (h : b.state = MemoryBlockState.active) :
    b.isEvicted = false :=
  show (b.state == MemoryBlockState.evicted) = false from
    Eq.subst (motive := fun s => (s == MemoryBlockState.evicted) = false) (Eq.symm h) (Eq.refl _)

theorem memoryBlock_active_not_entangled (b : MemoryBlock) (h : b.state = MemoryBlockState.active) :
    b.isEntangled = false :=
  show (b.state == MemoryBlockState.entangled) = false from
    Eq.subst (motive := fun s => (s == MemoryBlockState.entangled) = false) (Eq.symm h) (Eq.refl _)

theorem memoryBlock_active_not_archived (b : MemoryBlock) (h : b.state = MemoryBlockState.active) :
    b.isArchived = false :=
  show (b.state == MemoryBlockState.archived) = false from
    Eq.subst (motive := fun s => (s == MemoryBlockState.archived) = false) (Eq.symm h) (Eq.refl _)

theorem memoryBlock_evicted_not_active (b : MemoryBlock) (h : b.state = MemoryBlockState.evicted) :
    b.isActive = false :=
  show (b.state == MemoryBlockState.active) = false from
    Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _)

theorem memoryBlock_state_exclusive (b : MemoryBlock) :
    (b.isActive = true → b.isEvicted = false) ∧
    (b.isActive = true → b.isEntangled = false) ∧
    (b.isActive = true → b.isArchived = false) ∧
    (b.isEvicted = true → b.isActive = false) :=
  And.intro
    (fun ha =>
      match h : b.state with
      | MemoryBlockState.active => memoryBlock_active_not_evicted b h
      | MemoryBlockState.evicted =>
        absurd (show b.isActive = true from ha)
          (show ¬(b.isActive = true) from
            fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (memoryBlock_evicted_not_active b h)) h2))
      | MemoryBlockState.entangled =>
        absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isActive = false from
          Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _))) h2))
      | MemoryBlockState.archived =>
        absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isActive = false from
          Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _))) h2)))
    (And.intro
      (fun ha =>
        match h : b.state with
        | MemoryBlockState.active => memoryBlock_active_not_entangled b h
        | MemoryBlockState.evicted =>
          absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (memoryBlock_evicted_not_active b h)) h2))
        | MemoryBlockState.entangled =>
          absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isActive = false from
            Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _))) h2))
        | MemoryBlockState.archived =>
          absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isActive = false from
            Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _))) h2)))
      (And.intro
        (fun ha =>
          match h : b.state with
          | MemoryBlockState.active => memoryBlock_active_not_archived b h
          | MemoryBlockState.evicted =>
            absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (memoryBlock_evicted_not_active b h)) h2))
          | MemoryBlockState.entangled =>
            absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isActive = false from
              Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _))) h2))
          | MemoryBlockState.archived =>
            absurd ha (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isActive = false from
              Eq.subst (motive := fun s => (s == MemoryBlockState.active) = false) (Eq.symm h) (Eq.refl _))) h2)))
        (fun he =>
          match h : b.state with
          | MemoryBlockState.active =>
            absurd he (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (memoryBlock_active_not_evicted b h)) h2))
          | MemoryBlockState.evicted => memoryBlock_evicted_not_active b h
          | MemoryBlockState.entangled =>
            absurd he (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isEvicted = false from
              Eq.subst (motive := fun s => (s == MemoryBlockState.evicted) = false) (Eq.symm h) (Eq.refl _))) h2))
          | MemoryBlockState.archived =>
            absurd he (fun h2 => Bool.noConfusion (Eq.trans (Eq.symm (show b.isEvicted = false from
              Eq.subst (motive := fun s => (s == MemoryBlockState.evicted) = false) (Eq.symm h) (Eq.refl _))) h2)))))

def transitionState (current target : MemoryBlockState) : Option MemoryBlockState :=
  match current, target with
  | MemoryBlockState.active, MemoryBlockState.evicted => some MemoryBlockState.evicted
  | MemoryBlockState.active, MemoryBlockState.entangled => some MemoryBlockState.entangled
  | MemoryBlockState.active, MemoryBlockState.archived => some MemoryBlockState.archived
  | MemoryBlockState.entangled, MemoryBlockState.active => some MemoryBlockState.active
  | MemoryBlockState.archived, MemoryBlockState.active => some MemoryBlockState.active
  | _, _ => none

theorem transitionState_evicted_terminal :
    ∀ (target : MemoryBlockState), transitionState MemoryBlockState.evicted target = none :=
  fun target =>
    match target with
    | MemoryBlockState.active => Eq.refl _
    | MemoryBlockState.evicted => Eq.refl _
    | MemoryBlockState.entangled => Eq.refl _
    | MemoryBlockState.archived => Eq.refl _

theorem transitionState_active_evicted :
    transitionState MemoryBlockState.active MemoryBlockState.evicted = some MemoryBlockState.evicted :=
  Eq.refl _

theorem transitionState_active_entangled :
    transitionState MemoryBlockState.active MemoryBlockState.entangled = some MemoryBlockState.entangled :=
  Eq.refl _

theorem transitionState_active_archived :
    transitionState MemoryBlockState.active MemoryBlockState.archived = some MemoryBlockState.archived :=
  Eq.refl _

theorem transitionState_entangled_active :
    transitionState MemoryBlockState.entangled MemoryBlockState.active = some MemoryBlockState.active :=
  Eq.refl _

theorem transitionState_archived_active :
    transitionState MemoryBlockState.archived MemoryBlockState.active = some MemoryBlockState.active :=
  Eq.refl _

theorem transitionState_self_none :
    ∀ (s : MemoryBlockState), transitionState s s = none :=
  fun s =>
    match s with
    | MemoryBlockState.active => Eq.refl _
    | MemoryBlockState.evicted => Eq.refl _
    | MemoryBlockState.entangled => Eq.refl _
    | MemoryBlockState.archived => Eq.refl _

theorem transitionState_result_matches_target :
    ∀ (s t : MemoryBlockState),
      match transitionState s t with
      | some r => r = t
      | none => True :=
  fun s t =>
    match s, t with
    | MemoryBlockState.active, MemoryBlockState.active => True.intro
    | MemoryBlockState.active, MemoryBlockState.evicted => Eq.refl _
    | MemoryBlockState.active, MemoryBlockState.entangled => Eq.refl _
    | MemoryBlockState.active, MemoryBlockState.archived => Eq.refl _
    | MemoryBlockState.evicted, MemoryBlockState.active => True.intro
    | MemoryBlockState.evicted, MemoryBlockState.evicted => True.intro
    | MemoryBlockState.evicted, MemoryBlockState.entangled => True.intro
    | MemoryBlockState.evicted, MemoryBlockState.archived => True.intro
    | MemoryBlockState.entangled, MemoryBlockState.active => Eq.refl _
    | MemoryBlockState.entangled, MemoryBlockState.evicted => True.intro
    | MemoryBlockState.entangled, MemoryBlockState.entangled => True.intro
    | MemoryBlockState.entangled, MemoryBlockState.archived => True.intro
    | MemoryBlockState.archived, MemoryBlockState.active => Eq.refl _
    | MemoryBlockState.archived, MemoryBlockState.evicted => True.intro
    | MemoryBlockState.archived, MemoryBlockState.entangled => True.intro
    | MemoryBlockState.archived, MemoryBlockState.archived => True.intro

inductive SurpriseLevel where
  | high
  | low
  | medium
deriving Repr, BEq

def classifySurprise (score threshold : Rational) : SurpriseLevel :=
  if decide (Rational.gt score threshold) then SurpriseLevel.high
  else if decide (Rational.lt score (Rational.mul threshold (Rational.mk' 1 2 (Nat.zero_lt_succ 1)))) then SurpriseLevel.low
  else SurpriseLevel.medium

theorem classifySurprise_high (score threshold : Rational)
    (h : decide (Rational.gt score threshold) = true) :
    classifySurprise score threshold = SurpriseLevel.high :=
  show (if decide (Rational.gt score threshold) then SurpriseLevel.high else _) = SurpriseLevel.high from
    Eq.mpr (congrArg (fun b => (if b then SurpriseLevel.high else _) = SurpriseLevel.high) h ▸ Eq.refl _)
      (Eq.refl _)

theorem classifySurprise_exhaustive (score threshold : Rational) :
    classifySurprise score threshold = SurpriseLevel.high ∨
    classifySurprise score threshold = SurpriseLevel.low ∨
    classifySurprise score threshold = SurpriseLevel.medium :=
  match h1 : decide (Rational.gt score threshold),
        h2 : decide (Rational.lt score (Rational.mul threshold (Rational.mk' 1 2 (Nat.zero_lt_succ 1)))) with
  | true, _ => Or.inl (Eq.refl _)
  | false, true => Or.inr (Or.inl (Eq.refl _))
  | false, false => Or.inr (Or.inr (Eq.refl _))

structure EntanglementPair where
  block_a : BlockId
  block_b : BlockId
  surprise_a : Rational
  surprise_b : Rational
deriving Repr

def findEntanglementCandidates (records : List (BlockId × SurpriseRecord)) (threshold : Rational) (maxPairs : Nat) : List EntanglementPair :=
  let high_surprise := records.filter (fun p => decide (Rational.gt p.2.surprise_score threshold))
  let limited := high_surprise.take MAX_ENTANGLEMENT_PAIRS
  let rec outer (remaining : List (BlockId × SurpriseRecord)) (acc : List EntanglementPair) : List EntanglementPair :=
    match remaining with
    | [] => acc.reverse
    | a :: tl =>
        let newAcc := tl.foldl (fun pairs b =>
          if pairs.length < maxPairs then
            { block_a := a.1, block_b := b.1, surprise_a := a.2.surprise_score, surprise_b := b.2.surprise_score } :: pairs
          else pairs) acc
        outer tl newAcc
  outer limited []

theorem findEntanglementCandidates_empty :
    ∀ (threshold : Rational) (maxPairs : Nat),
      findEntanglementCandidates [] threshold maxPairs = [] :=
  fun _ _ => Eq.refl _

structure RetentionInvariant (r : SurpriseRecord) : Prop where
  access_positive : r.access_frequency ≥ 1
  creation_before_access : r.creation_time ≤ r.last_access_time

theorem retention_invariant_init (bid : BlockId) (score : Rational) (now : Int) :
    RetentionInvariant (SurpriseRecord.init bid score now) :=
  { access_positive := Nat.le_refl 1
  , creation_before_access := Int.le_refl _ }

theorem retention_invariant_recordAccess (r : SurpriseRecord) (now : Int)
    (inv : RetentionInvariant r) :
    RetentionInvariant (r.recordAccess now) :=
  { access_positive :=
      show r.access_frequency + 1 ≥ 1 from
        Nat.succ_le_succ (Nat.zero_le _)
  , creation_before_access :=
      let updated := { r with access_frequency := r.access_frequency + 1, last_access_time := now }
      let age_component := Rational.mul retentionAgeWeight (recencyFactorFromAgeMs 0)
      let freq_component := Rational.mul retentionFrequencyWeight (frequencyFactorFromAccesses updated.access_frequency)
      let weight_sum := Rational.add (Rational.add retentionBase age_component) freq_component
      let new_priority := Rational.clamp Rational.zero Rational.one (Rational.mul updated.surprise_score weight_sum)
      let final := { updated with retention_priority := new_priority }
      show final.creation_time ≤ final.last_access_time from
        Eq.subst (motive := fun x => r.creation_time ≤ x) (Eq.refl _) inv.creation_before_access }

structure ManagerInvariant (m : ManagerState) : Prop where
  stats_inv : StatisticsInvariant m.statistics
  threshold_valid : Rational.le Rational.zero m.surprise_threshold ∧ Rational.le m.surprise_threshold Rational.one

theorem manager_invariant_init (cap : Nat) :
    ManagerInvariant (ManagerState.init cap) :=
  { stats_inv := statistics_invariant_init defaultThreshold
  , threshold_valid := And.intro (Rational.le_refl _) (Rational.le_refl _) }

inductive SurpriseOp where
  | store (data : List UInt8)
  | evict (target : Nat)
  | setThreshold (threshold : Rational)
  | access (blockId : BlockId)
deriving Repr

structure SurpriseSystemState where
  manager : ManagerState
  current_time : Int
  operation_count : Nat
deriving Repr

def SurpriseSystemState.init (capacity : Nat) (time : Int) : SurpriseSystemState :=
  { manager := ManagerState.init capacity
  , current_time := time
  , operation_count := 0 }

def SurpriseSystemState.storeData (sys : SurpriseSystemState) (data : List UInt8) : Except StoreError (SurpriseSystemState × BlockId) :=
  match storeWithSurprise sys.manager data sys.current_time with
  | Except.ok (mgr, bid) =>
    Except.ok ({ sys with manager := mgr, operation_count := sys.operation_count + 1 }, bid)
  | Except.error e => Except.error e

def SurpriseSystemState.evict (sys : SurpriseSystemState) (target : Nat) : SurpriseSystemState × Nat :=
  let (mgr, count) := evictLowSurpriseBlocks sys.manager target
  ({ sys with manager := mgr, operation_count := sys.operation_count + 1 }, count)

theorem system_init_op_count (cap : Nat) (time : Int) :
    (SurpriseSystemState.init cap time).operation_count = 0 :=
  Eq.refl _

theorem system_init_manager_empty (cap : Nat) (time : Int) :
    (SurpriseSystemState.init cap time).manager.storage.count = 0 :=
  Eq.refl _

theorem system_evict_increments_ops (sys : SurpriseSystemState) (target : Nat) :
    (sys.evict target).1.operation_count = sys.operation_count + 1 :=
  Eq.refl _

def countHighSurprise (records : List (BlockId × SurpriseRecord)) (threshold : Rational) : Nat :=
  (records.filter (fun p => decide (Rational.gt p.2.surprise_score threshold))).length

def countLowSurprise (records : List (BlockId × SurpriseRecord)) (threshold : Rational) : Nat :=
  (records.filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold)))).length

theorem countHighSurprise_empty (threshold : Rational) :
    countHighSurprise [] threshold = 0 :=
  Eq.refl _

theorem countLowSurprise_empty (threshold : Rational) :
    countLowSurprise [] threshold = 0 :=
  Eq.refl _

theorem countHighSurprise_le (records : List (BlockId × SurpriseRecord)) (threshold : Rational) :
    countHighSurprise records threshold ≤ records.length :=
  List.length_filter_le _ _

theorem countLowSurprise_le (records : List (BlockId × SurpriseRecord)) (threshold : Rational) :
    countLowSurprise records threshold ≤ records.length :=
  List.length_filter_le _ _

theorem high_plus_low_eq_total (records : List (BlockId × SurpriseRecord)) (threshold : Rational) :
    countHighSurprise records threshold + countLowSurprise records threshold = records.length :=
  List.recOn records
    (show 0 + 0 = 0 from Eq.refl _)
    (fun hd tl ih =>
      match h : decide (Rational.gt hd.2.surprise_score threshold) with
      | true =>
        show ((hd :: tl).filter (fun p => decide (Rational.gt p.2.surprise_score threshold))).length +
             ((hd :: tl).filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold)))).length =
             (hd :: tl).length from
          have h_filt_true : (hd :: tl).filter (fun p => decide (Rational.gt p.2.surprise_score threshold)) =
                             hd :: tl.filter (fun p => decide (Rational.gt p.2.surprise_score threshold)) :=
            Eq.refl _
          have h_filt_false : (hd :: tl).filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold))) =
                              tl.filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold))) :=
            Eq.refl _
          Eq.trans (congrArg (· + _) (congrArg List.length h_filt_true))
            (Eq.trans (congrArg (_ + ·) (congrArg List.length h_filt_false))
              (show (tl.filter _).length + 1 + (tl.filter _).length = tl.length + 1 from
                Eq.trans (Nat.add_right_comm _ 1 _) (congrArg (· + 1) ih)))
      | false =>
        show ((hd :: tl).filter (fun p => decide (Rational.gt p.2.surprise_score threshold))).length +
             ((hd :: tl).filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold)))).length =
             (hd :: tl).length from
          have h_filt_true : (hd :: tl).filter (fun p => decide (Rational.gt p.2.surprise_score threshold)) =
                             tl.filter (fun p => decide (Rational.gt p.2.surprise_score threshold)) :=
            Eq.refl _
          have h_filt_false : (hd :: tl).filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold))) =
                              hd :: tl.filter (fun p => !(decide (Rational.gt p.2.surprise_score threshold))) :=
            Eq.refl _
          Eq.trans (congrArg (· + _) (congrArg List.length h_filt_true))
            (Eq.trans (congrArg (_ + ·) (congrArg List.length h_filt_false))
              (show (tl.filter _).length + ((tl.filter _).length + 1) = tl.length + 1 from
                Eq.trans (Eq.symm (Nat.add_assoc _ _ 1)) (congrArg (· + 1) ih)))))

def updateAllRetentions (records : List (BlockId × SurpriseRecord)) (now : Int) : List (BlockId × SurpriseRecord) :=
  records.map (fun (bid, rec) => (bid, rec.updateRetention now))

theorem updateAllRetentions_preserves_length (records : List (BlockId × SurpriseRecord)) (now : Int) :
    (updateAllRetentions records now).length = records.length :=
  List.length_map _ records

theorem updateAllRetentions_empty (now : Int) :
    updateAllRetentions [] now = [] :=
  Eq.refl _

theorem updateAllRetentions_preserves_ids (records : List (BlockId × SurpriseRecord)) (now : Int) :
    (updateAllRetentions records now).map Prod.fst = records.map Prod.fst :=
  List.recOn records
    (Eq.refl _)
    (fun (bid, rec) rest ih =>
      show bid :: (updateAllRetentions rest now).map Prod.fst = bid :: rest.map Prod.fst from
        congrArg (List.cons bid) ih)

theorem updateAllRetentions_preserves_scores (records : List (BlockId × SurpriseRecord)) (now : Int) :
    (updateAllRetentions records now).map (fun p => p.2.surprise_score) =
    records.map (fun p => p.2.surprise_score) :=
  List.recOn records
    (Eq.refl _)
    (fun (bid, rec) rest ih =>
      show rec.surprise_score :: (updateAllRetentions rest now).map (fun p => p.2.surprise_score) =
           rec.surprise_score :: rest.map (fun p => p.2.surprise_score) from
        congrArg (List.cons rec.surprise_score) ih)

theorem updateAllRetentions_preserves_frequencies (records : List (BlockId × SurpriseRecord)) (now : Int) :
    (updateAllRetentions records now).map (fun p => p.2.access_frequency) =
    records.map (fun p => p.2.access_frequency) :=
  List.recOn records
    (Eq.refl _)
    (fun (bid, rec) rest ih =>
      show rec.access_frequency :: (updateAllRetentions rest now).map (fun p => p.2.access_frequency) =
           rec.access_frequency :: rest.map (fun p => p.2.access_frequency) from
        congrArg (List.cons rec.access_frequency) ih)

def totalSurpriseSum (records : List (BlockId × SurpriseRecord)) : Rational :=
  records.foldl (fun acc p => Rational.add acc p.2.surprise_score) Rational.zero

theorem totalSurpriseSum_empty :
    totalSurpriseSum [] = Rational.zero :=
  Eq.refl _

structure CapacityConfig where
  max_blocks : Nat
  high_watermark : Nat
  low_watermark : Nat
  high_le_max : high_watermark ≤ max_blocks
  low_le_high : low_watermark ≤ high_watermark

theorem capacityConfig_low_le_max (c : CapacityConfig) :
    c.low_watermark ≤ c.max_blocks :=
  Nat.le_trans c.low_le_high c.high_le_max

def shouldEvict (st : ManagerState) (config : CapacityConfig) : Bool :=
  st.storage.count > config.high_watermark

def evictionTarget (config : CapacityConfig) : Nat :=
  config.low_watermark

theorem evictionTarget_le_high (config : CapacityConfig) :
    evictionTarget config ≤ config.high_watermark :=
  config.low_le_high

theorem evictionTarget_le_max (config : CapacityConfig) :
    evictionTarget config ≤ config.max_blocks :=
  capacityConfig_low_le_max config

inductive DataFlowNode where
  | source (id : Nat)
  | transform (id : Nat) (input : DataFlowNode)
  | sink (id : Nat) (input : DataFlowNode)
  | merge (id : Nat) (left right : DataFlowNode)
deriving Repr

def DataFlowNode.getId : DataFlowNode → Nat
  | source id => id
  | transform id _ => id
  | sink id _ => id
  | merge id _ _ => id

def DataFlowNode.depth : DataFlowNode → Nat
  | source _ => 0
  | transform _ input => input.depth + 1
  | sink _ input => input.depth + 1
  | merge _ left right => max left.depth right.depth + 1

def DataFlowNode.nodeCount : DataFlowNode → Nat
  | source _ => 1
  | transform _ input => input.nodeCount + 1
  | sink _ input => input.nodeCount + 1
  | merge _ left right => left.nodeCount + right.nodeCount + 1

theorem dataFlowNode_source_depth (id : Nat) :
    (DataFlowNode.source id).depth = 0 := Eq.refl _

theorem dataFlowNode_transform_depth (id : Nat) (input : DataFlowNode) :
    (DataFlowNode.transform id input).depth = input.depth + 1 := Eq.refl _

theorem dataFlowNode_sink_depth (id : Nat) (input : DataFlowNode) :
    (DataFlowNode.sink id input).depth = input.depth + 1 := Eq.refl _

theorem dataFlowNode_merge_depth (id : Nat) (l r : DataFlowNode) :
    (DataFlowNode.merge id l r).depth = max l.depth r.depth + 1 := Eq.refl _

theorem dataFlowNode_nodeCount_positive : ∀ (n : DataFlowNode), n.nodeCount ≥ 1 :=
  fun n =>
    DataFlowNode.recOn n
      (fun _ => Nat.le_refl 1)
      (fun _ _ ih => Nat.succ_le_succ (Nat.zero_le _))
      (fun _ _ ih => Nat.succ_le_succ (Nat.zero_le _))
      (fun _ _ _ _ _ => Nat.succ_le_succ (Nat.zero_le _))

private theorem Nat.max_le_of_le_of_le {a b c : Nat} (h1 : a ≤ c) (h2 : b ≤ c) : max a b ≤ c :=
  match Nat.decLe a b with
  | Decidable.isTrue h =>
    have : max a b = b := Nat.max_eq_right h
    Eq.subst (Eq.symm this) h2
  | Decidable.isFalse h =>
    have h' : b ≤ a := Nat.le_of_not_le h
    have : max a b = a := Nat.max_eq_left h'
    Eq.subst (Eq.symm this) h1

theorem dataFlowNode_depth_le_nodeCount : ∀ (n : DataFlowNode), n.depth ≤ n.nodeCount :=
  fun n =>
    DataFlowNode.recOn n
      (fun _ => Nat.zero_le 1)
      (fun _ _ ih => Nat.succ_le_succ ih)
      (fun _ _ ih => Nat.succ_le_succ ih)
      (fun _ _ _ ih_l ih_r =>
        Nat.succ_le_succ (Nat.max_le_of_le_of_le
          (Nat.le_trans ih_l (Nat.le_add_right _ _))
          (Nat.le_trans ih_r (Nat.le_add_left _ _))))

def DataFlowNode.collectIds : DataFlowNode → List Nat
  | source id => [id]
  | transform id input => id :: input.collectIds
  | sink id input => id :: input.collectIds
  | merge id left right => id :: (left.collectIds ++ right.collectIds)

theorem dataFlowNode_collectIds_length : ∀ (n : DataFlowNode), n.collectIds.length = n.nodeCount :=
  fun n =>
    DataFlowNode.recOn n
      (fun _ => Eq.refl _)
      (fun _ _ ih => congrArg Nat.succ ih)
      (fun _ _ ih => congrArg Nat.succ ih)
      (fun _ _ _ ih_l ih_r =>
        show 1 + (DataFlowNode.collectIds _ ++ DataFlowNode.collectIds _).length =
             DataFlowNode.nodeCount _ + DataFlowNode.nodeCount _ + 1 from
          have h_app : (DataFlowNode.collectIds _ ++ DataFlowNode.collectIds _).length =
                       (DataFlowNode.collectIds _).length + (DataFlowNode.collectIds _).length :=
            List.length_append _ _
          Eq.trans
            (congrArg (1 + ·) h_app)
            (Eq.trans
              (congrArg (1 + ·) (Eq.trans (congrArg (· + _) ih_l) (congrArg (_ + ·) ih_r)))
              (Nat.add_comm 1 _)))

def HASH_SIZE_val : HASH_SIZE = 16 := Eq.refl _
def HASH_BITS_val : HASH_BITS = 128 := Eq.refl _
def MAX_INPUT_SIZE_val : MAX_INPUT_SIZE = 104857600 := Eq.refl _

theorem hash_bits_eq : HASH_BITS = HASH_SIZE * 8 := Eq.refl _

theorem weight_sum_is_ten :
    RETENTION_AGE_WEIGHT_NUM + RETENTION_FREQUENCY_WEIGHT_NUM + RETENTION_BASE_WEIGHT_NUM =
    RETENTION_AGE_WEIGHT_DEN :=
  Eq.refl _

def SurpriseRecord.isStale (r : SurpriseRecord) (now : Int) (staleThresholdNs : Int) : Bool :=
  decide (now - r.last_access_time > staleThresholdNs)

def findStaleRecords (records : List (BlockId × SurpriseRecord)) (now : Int) (staleThresholdNs : Int) : List BlockId :=
  (records.filter (fun p => p.2.isStale now staleThresholdNs)).map Prod.fst

theorem findStaleRecords_empty (now : Int) (threshold : Int) :
    findStaleRecords [] now threshold = [] :=
  Eq.refl _

theorem findStaleRecords_length_le (records : List (BlockId × SurpriseRecord)) (now : Int) (threshold : Int) :
    (findStaleRecords records now threshold).length ≤ records.length :=
  Nat.le_trans
    (show ((records.filter _).map Prod.fst).length ≤ (records.filter _).length from
      Nat.le_of_eq (List.length_map _ _))
    (List.length_filter_le _ _)

structure SurpriseAnalysis where
  metrics : SurpriseMetrics
  classification : SurpriseLevel
  recommended_action : Nat
deriving Repr

def analyzeSurprise (metrics : SurpriseMetrics) (threshold : Rational) : SurpriseAnalysis :=
  let classification := classifySurprise metrics.combined_surprise threshold
  let action := match classification with
    | SurpriseLevel.high => 2
    | SurpriseLevel.low => 0
    | SurpriseLevel.medium => 1
  { metrics := metrics, classification := classification, recommended_action := action }

theorem analyzeSurprise_preserves_metrics (m : SurpriseMetrics) (t : Rational) :
    (analyzeSurprise m t).metrics = m :=
  Eq.refl _

theorem analyzeSurprise_high_action (m : SurpriseMetrics) (t : Rational)
    (h : decide (Rational.gt m.combined_surprise t) = true) :
    (analyzeSurprise m t).classification = SurpriseLevel.high :=
  show classifySurprise m.combined_surprise t = SurpriseLevel.high from
    classifySurprise_high m.combined_surprise t h

theorem analyzeSurprise_high_recommended (m : SurpriseMetrics) (t : Rational)
    (h : decide (Rational.gt m.combined_surprise t) = true) :
    (analyzeSurprise m t).recommended_action = 2 :=
  show (match classifySurprise m.combined_surprise t with
        | SurpriseLevel.high => 2 | SurpriseLevel.low => 0 | SurpriseLevel.medium => 1) = 2 from
    have hc : classifySurprise m.combined_surprise t = SurpriseLevel.high :=
      classifySurprise_high m.combined_surprise t h
    Eq.subst (motive := fun c => (match c with | SurpriseLevel.high => 2 | SurpriseLevel.low => 0 | SurpriseLevel.medium => 1) = 2)
      (Eq.symm hc) (Eq.refl _)

def selectEvictionCandidates (records : List (BlockId × SurpriseRecord)) (k : Nat) : List BlockId :=
  let sorted := records.mergeSort (fun a b => decide (Rational.lt a.2.retention_priority b.2.retention_priority))
  (sorted.take k).map Prod.fst

theorem selectEvictionCandidates_length_le (records : List (BlockId × SurpriseRecord)) (k : Nat) :
    (selectEvictionCandidates records k).length ≤ k :=
  show ((records.mergeSort _).take k).map Prod.fst |>.length ≤ k from
    Nat.le_trans
      (Nat.le_of_eq (List.length_map _ _))
      (List.length_take_le _ _)

theorem selectEvictionCandidates_empty (k : Nat) :
    selectEvictionCandidates [] k = [] :=
  Eq.refl _

theorem selectEvictionCandidates_zero (records : List (BlockId × SurpriseRecord)) :
    selectEvictionCandidates records 0 = [] :=
  show ((records.mergeSort _).take 0).map Prod.fst = [] from
    Eq.refl _

section MetricSpaceAxioms

def hammingDistNat (a b : BlockId) : Nat :=
  let rec go (i : Nat) (dist : Nat) : Nat :=
    if h : i < HASH_SIZE then
      go (i + 1) (dist + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩))
    else dist
  go 0 0

theorem hammingDistNat_eq_hashDist_num (a b : BlockId) :
    (computeHashDistance a b).num = Int.ofNat (hammingDistNat a b) :=
  Eq.refl _

theorem xor_comm_uint8 (x y : UInt8) : x ^^^ y = y ^^^ x :=
  UInt8.ext (show (x ^^^ y).val = (y ^^^ x).val from
    have : x.val ^^^ y.val = y.val ^^^ x.val :=
      Fin.ext (show (x.val ^^^ y.val).val = (y.val ^^^ x.val).val from
        have : x.val.val ^^^ y.val.val = y.val.val ^^^ x.val.val := Nat.xor_comm x.val.val y.val.val
        this)
    this)

theorem popCount8_xor_comm (x y : UInt8) :
    popCount8 (x ^^^ y) = popCount8 (y ^^^ x) :=
  congrArg popCount8 (xor_comm_uint8 x y)

theorem hammingDistNat_symmetric_go (a b : BlockId) (i : Nat) (d : Nat) :
    hammingDistNat.go a b i d = hammingDistNat.go b a i d :=
  if h : i < HASH_SIZE then
    have h_xor : popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩) = popCount8 (b ⟨i, h⟩ ^^^ a ⟨i, h⟩) :=
      popCount8_xor_comm (a ⟨i, h⟩) (b ⟨i, h⟩)
    show (if h' : i < HASH_SIZE then hammingDistNat.go a b (i + 1) (d + popCount8 (a ⟨i, h'⟩ ^^^ b ⟨i, h'⟩)) else d) =
         (if h' : i < HASH_SIZE then hammingDistNat.go b a (i + 1) (d + popCount8 (b ⟨i, h'⟩ ^^^ a ⟨i, h'⟩)) else d) from
      have : hammingDistNat.go a b (i + 1) (d + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩)) =
             hammingDistNat.go b a (i + 1) (d + popCount8 (b ⟨i, h⟩ ^^^ a ⟨i, h⟩)) :=
        Eq.trans
          (congrArg (hammingDistNat.go a b (i + 1)) (congrArg (d + ·) h_xor))
          (hammingDistNat_symmetric_go a b (i + 1) (d + popCount8 (b ⟨i, h⟩ ^^^ a ⟨i, h⟩)))
      Eq.refl _ ▸ this ▸ Eq.refl _
  else
    show (if h' : i < HASH_SIZE then _ else d) = (if h' : i < HASH_SIZE then _ else d) from
      Eq.refl _
termination_by HASH_SIZE - i

theorem hammingDistNat_symmetric (a b : BlockId) :
    hammingDistNat a b = hammingDistNat b a :=
  hammingDistNat_symmetric_go a b 0 0

theorem computeHashDistance_symmetric (a b : BlockId) :
    computeHashDistance a b = computeHashDistance b a :=
  show { num := Int.ofNat (hammingDistNat a b), den := HASH_BITS, den_pos := _ } =
       { num := Int.ofNat (hammingDistNat b a), den := HASH_BITS, den_pos := _ } from
    have h_eq : hammingDistNat a b = hammingDistNat b a := hammingDistNat_symmetric a b
    congrArg (fun n => { num := Int.ofNat n, den := HASH_BITS, den_pos := Nat.zero_lt_succ 127 }) h_eq

theorem computeHashDistance_self_zero (h : BlockId) :
    computeHashDistance h h = Rational.mk' 0 HASH_BITS (Nat.zero_lt_succ 127) :=
  show { num := Int.ofNat (hammingDistNat h h), den := HASH_BITS, den_pos := _ } =
       { num := 0, den := HASH_BITS, den_pos := _ } from
    have h_self : hammingDistNat h h = 0 :=
      show hammingDistNat.go h h 0 0 = 0 from Eq.refl _
    congrArg (fun n => { num := Int.ofNat n, den := HASH_BITS, den_pos := Nat.zero_lt_succ 127 }) h_self

theorem xor_self_zero (x : UInt8) : x ^^^ x = 0 :=
  UInt8.ext (show (x ^^^ x).val = (0 : UInt8).val from
    Fin.ext (show (x.val ^^^ x.val).val = 0 from Nat.xor_self x.val.val))

theorem popCount8_zero_is_zero : popCount8 0 = 0 := Eq.refl _

theorem popCount8_xor_self (x : UInt8) : popCount8 (x ^^^ x) = 0 :=
  show popCount8 (x ^^^ x) = 0 from
    have : x ^^^ x = 0 := xor_self_zero x
    Eq.subst (motive := fun v => popCount8 v = 0) (Eq.symm this) popCount8_zero_is_zero

theorem hammingDistNat_bounded_go (a b : BlockId) (i : Nat) (d : Nat) (h_i_le : i ≤ HASH_SIZE) :
    hammingDistNat.go a b i d ≤ d + (HASH_SIZE - i) * 8 :=
  if h : i < HASH_SIZE then
    have h_pop_le : popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩) ≤ 8 :=
      show popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩) ≤ 8 from
        let v := a ⟨i, h⟩ ^^^ b ⟨i, h⟩
        let rec check_bound (byte : UInt8) : popCount8 byte ≤ 8 :=
          Nat.le_of_lt_succ (Nat.lt_succ_of_le (Nat.le_of_lt_succ (show popCount8 byte < 9 from
            Nat.lt_of_le_of_lt (show popCount8 byte ≤ 8 from
              match byte with | _ => Nat.le_of_ble_eq_true (show Nat.ble (popCount8 byte) 8 = true from Eq.refl _)
            ) (Nat.lt_succ_of_le (Nat.le_refl 8)))))
        check_bound v
    have h_rec : hammingDistNat.go a b (i + 1) (d + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩)) ≤
                 (d + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩)) + (HASH_SIZE - (i + 1)) * 8 :=
      hammingDistNat_bounded_go a b (i + 1) (d + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩))
        (Nat.succ_le_of_lt h)
    show hammingDistNat.go a b i d ≤ d + (HASH_SIZE - i) * 8 from
      have expand : (HASH_SIZE - i) * 8 = (HASH_SIZE - (i + 1)) * 8 + 8 :=
        have : HASH_SIZE - i = (HASH_SIZE - (i + 1)) + 1 := Nat.sub_succ' HASH_SIZE i ▸ Eq.refl _
        show (HASH_SIZE - i) * 8 = (HASH_SIZE - (i + 1)) * 8 + 8 from
          Eq.subst (motive := fun n => n * 8 = (HASH_SIZE - (i + 1)) * 8 + 8) (Eq.symm this)
            (Nat.succ_mul (HASH_SIZE - (i + 1)) 8 ▸ Nat.add_comm ((HASH_SIZE - (i + 1)) * 8) 8 ▸ Eq.refl _)
      Nat.le_trans h_rec (
        Nat.le_trans
          (Nat.add_le_add_right (Nat.add_le_add_left h_pop_le d) _)
          (show d + 8 + (HASH_SIZE - (i + 1)) * 8 ≤ d + (HASH_SIZE - i) * 8 from
            Eq.subst (motive := fun n => d + 8 + (HASH_SIZE - (i + 1)) * 8 ≤ d + n) (Eq.symm expand)
              (Nat.le_of_eq (Nat.add_assoc d 8 ((HASH_SIZE - (i + 1)) * 8) ▸
                congrArg (d + ·) (Nat.add_comm 8 ((HASH_SIZE - (i + 1)) * 8))))))
  else
    show (if h' : i < HASH_SIZE then _ else d) ≤ d + (HASH_SIZE - i) * 8 from
      have : ¬(i < HASH_SIZE) := h
      have h_ge : i ≥ HASH_SIZE := Nat.le_of_not_lt this
      have h_sub_zero : HASH_SIZE - i = 0 := Nat.sub_eq_zero_of_le h_ge
      show d ≤ d + (HASH_SIZE - i) * 8 from
        Eq.subst (motive := fun n => d ≤ d + n * 8) (Eq.symm h_sub_zero) (Nat.le_refl d)
termination_by HASH_SIZE - i

theorem hammingDistNat_bounded (a b : BlockId) :
    hammingDistNat a b ≤ HASH_BITS :=
  show hammingDistNat.go a b 0 0 ≤ HASH_BITS from
    have h := hammingDistNat_bounded_go a b 0 0 (Nat.zero_le _)
    show hammingDistNat.go a b 0 0 ≤ 128 from
      Nat.le_trans h (show 0 + (HASH_SIZE - 0) * 8 ≤ 128 from Nat.le_refl _)

theorem computeHashDistance_nonneg (a b : BlockId) :
    (computeHashDistance a b).num ≥ 0 :=
  show Int.ofNat (hammingDistNat a b) ≥ 0 from
    Int.ofNat_le.mpr (Nat.zero_le _)

theorem computeHashDistance_bounded_num (a b : BlockId) :
    (computeHashDistance a b).num ≤ Int.ofNat HASH_BITS :=
  show Int.ofNat (hammingDistNat a b) ≤ Int.ofNat HASH_BITS from
    Int.ofNat_le.mpr (hammingDistNat_bounded a b)

private theorem Nat.sub_succ' (n m : Nat) : n - (m + 1) = n - m - 1 :=
  Eq.symm (Nat.sub_add_eq n m 1)

theorem xor_assoc_nat (a b c : Nat) : Nat.xor (Nat.xor a b) c = Nat.xor a (Nat.xor b c) :=
  Nat.xor_assoc a b c

theorem popCount8_triangle_single (x y z : UInt8) :
    popCount8 (x ^^^ z) ≤ popCount8 (x ^^^ y) + popCount8 (y ^^^ z) :=
  Nat.le_of_ble_eq_true (show Nat.ble (popCount8 (x ^^^ z)) (popCount8 (x ^^^ y) + popCount8 (y ^^^ z)) = true from Eq.refl _)

theorem hammingDistNat_triangle_go (a b c : BlockId) (i : Nat) (d_ac d_ab d_bc : Nat)
    (h_base : d_ac ≤ d_ab + d_bc) :
    hammingDistNat.go a c i d_ac ≤ hammingDistNat.go a b i d_ab + hammingDistNat.go b c i d_bc :=
  if h : i < HASH_SIZE then
    have h_pop_tri : popCount8 (a ⟨i, h⟩ ^^^ c ⟨i, h⟩) ≤
                     popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩) + popCount8 (b ⟨i, h⟩ ^^^ c ⟨i, h⟩) :=
      popCount8_triangle_single (a ⟨i, h⟩) (b ⟨i, h⟩) (c ⟨i, h⟩)
    have h_new_base : d_ac + popCount8 (a ⟨i, h⟩ ^^^ c ⟨i, h⟩) ≤
                      (d_ab + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩)) + (d_bc + popCount8 (b ⟨i, h⟩ ^^^ c ⟨i, h⟩)) :=
      Nat.le_trans
        (Nat.add_le_add h_base h_pop_tri)
        (show (d_ab + d_bc) + (popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩) + popCount8 (b ⟨i, h⟩ ^^^ c ⟨i, h⟩)) ≤
              (d_ab + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩)) + (d_bc + popCount8 (b ⟨i, h⟩ ^^^ c ⟨i, h⟩)) from
          Nat.le_of_eq (
            Eq.trans
              (Nat.add_assoc d_ab d_bc _ ▸
               Nat.add_assoc d_bc (popCount8 _) (popCount8 _) ▸
               Nat.add_comm d_bc (popCount8 _) ▸
               Nat.add_assoc (popCount8 _) d_bc (popCount8 _) ▸
               Nat.add_assoc d_ab (popCount8 _ + d_bc) _ ▸
               Nat.add_assoc d_ab (popCount8 _) (d_bc + popCount8 _) ▸
               Eq.refl _)
              (Eq.refl _)))
    hammingDistNat_triangle_go a b c (i + 1)
      (d_ac + popCount8 (a ⟨i, h⟩ ^^^ c ⟨i, h⟩))
      (d_ab + popCount8 (a ⟨i, h⟩ ^^^ b ⟨i, h⟩))
      (d_bc + popCount8 (b ⟨i, h⟩ ^^^ c ⟨i, h⟩))
      h_new_base
  else
    h_base
termination_by HASH_SIZE - i

theorem hammingDistNat_triangle (a b c : BlockId) :
    hammingDistNat a c ≤ hammingDistNat a b + hammingDistNat b c :=
  hammingDistNat_triangle_go a b c 0 0 0 0 (Nat.zero_le 0)

theorem computeHashDistance_triangle_num (a b c : BlockId) :
    (computeHashDistance a c).num ≤ (computeHashDistance a b).num + (computeHashDistance b c).num :=
  show Int.ofNat (hammingDistNat a c) ≤ Int.ofNat (hammingDistNat a b) + Int.ofNat (hammingDistNat b c) from
    have h := hammingDistNat_triangle a b c
    show Int.ofNat (hammingDistNat a c) ≤ Int.ofNat (hammingDistNat a b + hammingDistNat b c) from
      Int.ofNat_le.mpr h

theorem computeHashDistance_triangle_rational (a b c : BlockId) :
    Rational.le (computeHashDistance a c) (Rational.add (computeHashDistance a b) (computeHashDistance b c)) :=
  show (computeHashDistance a c).num * Int.ofNat (Rational.add (computeHashDistance a b) (computeHashDistance b c)).den ≤
       (Rational.add (computeHashDistance a b) (computeHashDistance b c)).num * Int.ofNat (computeHashDistance a c).den from
    have h_den_ac : (computeHashDistance a c).den = HASH_BITS := Eq.refl _
    have h_den_ab : (computeHashDistance a b).den = HASH_BITS := Eq.refl _
    have h_den_bc : (computeHashDistance b c).den = HASH_BITS := Eq.refl _
    have h_add_den : (Rational.add (computeHashDistance a b) (computeHashDistance b c)).den = HASH_BITS * HASH_BITS := Eq.refl _
    have h_add_num : (Rational.add (computeHashDistance a b) (computeHashDistance b c)).num =
                     Int.ofNat (hammingDistNat a b) * Int.ofNat HASH_BITS + Int.ofNat (hammingDistNat b c) * Int.ofNat HASH_BITS :=
      Eq.refl _
    show Int.ofNat (hammingDistNat a c) * Int.ofNat (HASH_BITS * HASH_BITS) ≤
         (Int.ofNat (hammingDistNat a b) * Int.ofNat HASH_BITS + Int.ofNat (hammingDistNat b c) * Int.ofNat HASH_BITS) * Int.ofNat HASH_BITS from
      have factor_rhs : (Int.ofNat (hammingDistNat a b) * Int.ofNat HASH_BITS + Int.ofNat (hammingDistNat b c) * Int.ofNat HASH_BITS) * Int.ofNat HASH_BITS =
                         (Int.ofNat (hammingDistNat a b) + Int.ofNat (hammingDistNat b c)) * (Int.ofNat HASH_BITS * Int.ofNat HASH_BITS) :=
        have h1 : (Int.ofNat (hammingDistNat a b) * Int.ofNat HASH_BITS + Int.ofNat (hammingDistNat b c) * Int.ofNat HASH_BITS) =
                   (Int.ofNat (hammingDistNat a b) + Int.ofNat (hammingDistNat b c)) * Int.ofNat HASH_BITS :=
          Eq.symm (Int.add_mul _ _ _)
        Eq.trans (congrArg (· * Int.ofNat HASH_BITS) h1) (Int.mul_assoc _ _ _)
      have factor_lhs : Int.ofNat (hammingDistNat a c) * Int.ofNat (HASH_BITS * HASH_BITS) =
                         Int.ofNat (hammingDistNat a c) * (Int.ofNat HASH_BITS * Int.ofNat HASH_BITS) :=
        congrArg (Int.ofNat (hammingDistNat a c) * ·) (Int.ofNat_mul HASH_BITS HASH_BITS)
      Eq.subst (Eq.symm factor_rhs) (Eq.subst (Eq.symm factor_lhs)
        (Int.mul_le_mul_of_nonneg_right
          (show Int.ofNat (hammingDistNat a c) ≤ Int.ofNat (hammingDistNat a b) + Int.ofNat (hammingDistNat b c) from
            Eq.subst (Eq.symm (Int.ofNat_add (hammingDistNat a b) (hammingDistNat b c)))
              (Int.ofNat_le.mpr (hammingDistNat_triangle a b c)))
          (show (0 : Int) ≤ Int.ofNat HASH_BITS * Int.ofNat HASH_BITS from
            Int.mul_nonneg (Int.ofNat_nonneg HASH_BITS) (Int.ofNat_nonneg HASH_BITS))))

end MetricSpaceAxioms

section EvictionSafety

def isMaxPriority (records : List (BlockId × SurpriseRecord)) (bid : BlockId) : Prop :=
  ∃ rec : SurpriseRecord, (bid, rec) ∈ records ∧
    ∀ (bid' : BlockId) (rec' : SurpriseRecord), (bid', rec') ∈ records → bid' ≠ bid →
      Rational.lt rec'.retention_priority rec

def allPrioritiesLt (candidates : List CandidateItem) (bound : Rational) : Prop :=
  ∀ c, c ∈ candidates → Rational.lt c.priority bound

theorem mergeSort_permutation_mem {α : Type} (cmp : α → α → Bool) (l : List α) (x : α) :
    x ∈ l ↔ x ∈ l.mergeSort cmp :=
  ⟨fun h => List.mem_of_perm_of_mem (List.perm_mergeSort cmp l) h,
   fun h => List.mem_of_perm_of_mem (List.Perm.symm (List.perm_mergeSort cmp l)) h⟩

theorem take_subset_mem {α : Type} (l : List α) (k : Nat) (x : α) :
    x ∈ l.take k → x ∈ l :=
  fun h => List.mem_of_mem_take h

def candidateFromRecord (p : BlockId × SurpriseRecord) : CandidateItem :=
  { block_id := p.1, priority := p.2.retention_priority }

theorem eviction_never_removes_max_priority
    (st : ManagerState) (target_capacity : Nat)
    (b_max : BlockId) (rec_max : SurpriseRecord)
    (h_in_records : (b_max, rec_max) ∈ st.surprise_records)
    (h_in_storage : st.storage.containsBlock b_max = true)
    (h_max : ∀ (bid' : BlockId) (rec' : SurpriseRecord),
      (bid', rec') ∈ st.surprise_records → bid' ≠ b_max →
      Rational.lt rec'.retention_priority rec_max)
    (h_target_pos : target_capacity > 0)
    (h_records_gt_one : st.surprise_records.length > 1)
    (h_capacity_lt : st.storage.count > target_capacity) :
    ∀ (cand : CandidateItem),
      cand ∈ (partialSortCandidates
        (st.surprise_records.map candidateFromRecord)
        (st.storage.count - target_capacity)).take (min (st.storage.count - target_capacity) (st.surprise_records.length)) →
      cand.block_id ≠ b_max :=
  fun cand h_mem =>
    let candidates := st.surprise_records.map candidateFromRecord
    let sorted := candidates.mergeSort (fun a b => decide (Rational.lt a.priority b.priority))
    let to_evict := st.storage.count - target_capacity
    let k := min to_evict sorted.length
    let taken := sorted.take k
    show cand.block_id ≠ b_max from
      have h_sorted_perm : sorted ~ candidates := List.perm_mergeSort _ candidates
      have h_taken_subset : ∀ x, x ∈ taken → x ∈ sorted := List.mem_take_of_mem
      have h_cand_in_taken : cand ∈ taken := h_mem
      have h_cand_in_sorted : cand ∈ sorted := h_taken_subset cand h_cand_in_taken
      have h_cand_in_candidates : cand ∈ candidates := (mergeSort_permutation_mem _ candidates cand).mpr h_cand_in_sorted
      have h_exists_p : ∃ p, p ∈ st.surprise_records ∧ candidateFromRecord p = cand :=
        List.mem_map.mp h_cand_in_candidates
      match h_exists_p with
      | ⟨p, h_p_in, h_cand_eq⟩ =>
        let (bid_p, rec_p) := p
        have h_bid_p_eq : bid_p = cand.block_id := congrArg CandidateItem.block_id (Eq.symm h_cand_eq)
        have h_pri_eq : rec_p.retention_priority = cand.priority := congrArg CandidateItem.priority (Eq.symm h_cand_eq)
        have h_pri_lt_max : Rational.lt rec_p.retention_priority rec_max :=
          h_max bid_p rec_p h_p_in (fun h_eq => absurd (Eq.trans (Eq.symm h_bid_p_eq) h_eq) (show cand.block_id ≠ b_max from h_mem ▸ h_cand_eq ▸ Eq.refl _))
        have h_cand_lt_max : Rational.lt cand.priority rec_max.retention_priority :=
          Eq.subst (motive := fun r => Rational.lt r rec_max.retention_priority) (Eq.symm h_pri_eq) h_pri_lt_max
        absurd h_cand_lt_max (show ¬Rational.lt cand.priority rec_max.retention_priority from
          have h_max_in_candidates : candidateFromRecord (b_max, rec_max) ∈ candidates := List.mem_map_of_mem _ h_in_records
          have h_max_in_sorted : candidateFromRecord (b_max, rec_max) ∈ sorted := (mergeSort_permutation_mem _ candidates _).mpr h_max_in_candidates
          have h_max_not_in_taken : candidateFromRecord (b_max, rec_max) ∉ taken :=
            fun h_max_taken =>
              have h_max_pri : (candidateFromRecord (b_max, rec_max)).priority = rec_max.retention_priority := Eq.refl _
              have h_all_lt : ∀ c, c ∈ taken → Rational.lt c.priority rec_max.retention_priority :=
                fun c h_c_taken =>
                  have h_c_sorted : c ∈ sorted := List.mem_take_of_mem h_c_taken
                  have h_c_candidates : c ∈ candidates := (mergeSort_permutation_mem _ candidates c).mpr h_c_sorted
                  have h_c_p : ∃ p', p' ∈ st.surprise_records ∧ candidateFromRecord p' = c := List.mem_map.mp h_c_candidates
                  match h_c_p with
                  | ⟨p', h_p'_in, h_c_eq⟩ =>
                    let (bid', rec') := p'
                    have h_bid' : bid' = c.block_id := congrArg CandidateItem.block_id (Eq.symm h_c_eq)
                    have h_pri' : rec'.retention_priority = c.priority := congrArg CandidateItem.priority (Eq.symm h_c_eq)
                    have h_lt : Rational.lt rec'.retention_priority rec_max :=
                      h_max bid' rec' h_p'_in (fun h_eq => absurd (Eq.trans (Eq.symm h_bid') h_eq) (show c.block_id ≠ b_max from h_c_taken ▸ h_c_eq ▸ Eq.refl _))
                    Eq.subst (motive := fun r => Rational.lt r rec_max.retention_priority) (Eq.symm h_pri') h_lt
              h_all_lt (candidateFromRecord (b_max, rec_max)) h_max_taken
          h_max_not_in_taken ▸ (fun _ => h_cand_lt_max))

theorem eviction_max_survives_structural
    (st : ManagerState) (target_capacity : Nat)
    (b_max : BlockId) (rec_max : SurpriseRecord)
    (h_in_records : (b_max, rec_max) ∈ st.surprise_records)
    (h_in_storage : st.storage.containsBlock b_max = true)
    (h_max : ∀ (bid' : BlockId) (rec' : SurpriseRecord),
      (bid', rec') ∈ st.surprise_records → bid' ≠ b_max →
      Rational.lt rec'.retention_priority rec_max)
    (h_target_pos : target_capacity > 0)
    (h_enough_others : st.surprise_records.length > st.storage.count - target_capacity) :
    let candidates := st.surprise_records.map candidateFromRecord
    let sorted := candidates.mergeSort (fun a b => decide (Rational.lt a.priority b.priority))
    let to_evict := st.storage.count - target_capacity
    let k := min to_evict sorted.length
    sorted.length > k →
    (∀ (c : CandidateItem), c ∈ sorted.take k → Rational.lt c.priority rec_max.retention_priority) →
    ¬(b_max ∈ (sorted.take k).map (fun c => c.block_id)) :=
  fun h_sorted_len_gt h_all_lt h_mem =>
    have h_exists : ∃ c, c ∈ sorted.take k ∧ c.block_id = b_max :=
      List.exists_mem_of_mem_map (show b_max ∈ (sorted.take k).map (fun c => c.block_id) from h_mem)
    match h_exists with
    | ⟨c, h_c_in, h_c_bid⟩ =>
      have h_lt_self : Rational.lt c.priority rec_max.retention_priority := h_all_lt c h_c_in
      have h_c_is_max : c.priority = rec_max.retention_priority :=
        have h_c_in_sorted : c ∈ sorted := take_subset_mem sorted _ c h_c_in
        have h_c_in_orig : c ∈ candidates :=
          (mergeSort_permutation_mem _ candidates c).mpr h_c_in_sorted
        have : ∃ p, p ∈ st.surprise_records ∧ candidateFromRecord p = c :=
          List.mem_map.mp h_c_in_orig
        match this with
        | ⟨(bid_p, rec_p), h_p_in, h_p_eq⟩ =>
          have h_bid_eq : bid_p = b_max :=
            show bid_p = b_max from
              have : c.block_id = bid_p :=
                show c.block_id = (candidateFromRecord (bid_p, rec_p)).block_id from
                  congrArg CandidateItem.block_id (Eq.symm h_p_eq)
              Eq.trans (Eq.symm this) h_c_bid
          have h_pri_eq : c.priority = rec_p.retention_priority :=
            show c.priority = (candidateFromRecord (bid_p, rec_p)).priority from
              congrArg CandidateItem.priority (Eq.symm h_p_eq)
          have h_rec_eq : rec_p = rec_max :=
            show rec_p = rec_max from
              have : bid_p = b_max := h_bid_eq
              have h_both_in : (b_max, rec_p) ∈ st.surprise_records :=
                Eq.subst (motive := fun b => (b, rec_p) ∈ st.surprise_records) this h_p_in
              have : (b_max, rec_max) ∈ st.surprise_records := h_in_records
              Eq.refl _ ▸ Eq.refl _ ▸ Eq.refl rec_p
          Eq.trans h_pri_eq (congrArg SurpriseRecord.retention_priority h_rec_eq)
      absurd h_lt_self (
        show ¬Rational.lt c.priority rec_max.retention_priority from
          Eq.subst (motive := fun p => ¬Rational.lt p rec_max.retention_priority) (Eq.symm h_c_is_max)
            (Int.lt_irrefl _))

end EvictionSafety

section StateTraceInvariants

inductive SystemTrace where
  | empty
  | storeOp (prev : SystemTrace) (data : List UInt8)
  | evictOp (prev : SystemTrace) (target : Nat)
  | setThresholdOp (prev : SystemTrace) (threshold : Rational)
  | accessOp (prev : SystemTrace) (blockId : BlockId)
deriving Repr

def applyTrace (trace : SystemTrace) (sys : SurpriseSystemState) : Except StoreError SurpriseSystemState :=
  match trace with
  | SystemTrace.empty => Except.ok sys
  | SystemTrace.storeOp prev data =>
    match applyTrace prev sys with
    | Except.error e => Except.error e
    | Except.ok prev_state =>
      match prev_state.storeData data with
      | Except.ok (new_state, _) => Except.ok new_state
      | Except.error e => Except.error e
  | SystemTrace.evictOp prev target =>
    match applyTrace prev sys with
    | Except.error e => Except.error e
    | Except.ok prev_state => Except.ok (prev_state.evict target).1
  | SystemTrace.setThresholdOp prev threshold =>
    match applyTrace prev sys with
    | Except.error e => Except.error e
    | Except.ok prev_state =>
      Except.ok { prev_state with
        manager := prev_state.manager.setSurpriseThreshold threshold
      , operation_count := prev_state.operation_count + 1 }
  | SystemTrace.accessOp prev blockId =>
    match applyTrace prev sys with
    | Except.error e => Except.error e
    | Except.ok prev_state =>
      match prev_state.manager.findRecord blockId with
      | some rec =>
        let rec' := rec.recordAccess prev_state.current_time
        Except.ok { prev_state with
          manager := prev_state.manager.putRecord blockId rec'
        , operation_count := prev_state.operation_count + 1 }
      | none => Except.error StoreError.notFound

theorem applyTrace_empty (sys : SurpriseSystemState) :
    applyTrace SystemTrace.empty sys = Except.ok sys :=
  Eq.refl _

structure TraceInvariant (sys : SurpriseSystemState) : Prop where
  manager_inv : ManagerInvariant sys.manager
  ops_nonneg : True

theorem trace_invariant_init (cap : Nat) (time : Int) :
    TraceInvariant (SurpriseSystemState.init cap time) :=
  { manager_inv := manager_invariant_init cap
  , ops_nonneg := True.intro }

theorem applyTrace_empty_preserves_invariant (sys : SurpriseSystemState)
    (inv : TraceInvariant sys) :
    match applyTrace SystemTrace.empty sys with
    | Except.ok final => TraceInvariant final
    | Except.error _ => True :=
  show TraceInvariant sys from inv

theorem setThreshold_preserves_statistics_invariant (s : SurpriseMemoryStatistics) (t : Rational)
    (inv : StatisticsInvariant s) :
    StatisticsInvariant { s with surprise_threshold := Rational.clamp Rational.zero Rational.one t } :=
  { partition_le := inv.partition_le
  , blocks_consistent := inv.blocks_consistent }

theorem putRecord_preserves_statistics (m : ManagerState) (bid : BlockId) (rec : SurpriseRecord)
    (inv : StatisticsInvariant m.statistics) :
    StatisticsInvariant (m.putRecord bid rec).statistics :=
  show StatisticsInvariant m.statistics from inv

theorem trace_setThreshold_preserves_invariant (sys : SurpriseSystemState) (t : Rational)
    (inv : TraceInvariant sys) :
    TraceInvariant { sys with
      manager := sys.manager.setSurpriseThreshold t
    , operation_count := sys.operation_count + 1 } :=
  { manager_inv :=
      { stats_inv := setThreshold_preserves_statistics_invariant sys.manager.statistics t inv.manager_inv.stats_inv
      , threshold_valid := And.intro (Rational.le_refl _) (Rational.le_refl _) }
  , ops_nonneg := True.intro }

theorem trace_access_preserves_invariant (sys : SurpriseSystemState) (bid : BlockId) (rec : SurpriseRecord)
    (inv : TraceInvariant sys)
    (h_found : sys.manager.findRecord bid = some rec) :
    TraceInvariant { sys with
      manager := sys.manager.putRecord bid (rec.recordAccess sys.current_time)
    , operation_count := sys.operation_count + 1 } :=
  { manager_inv :=
      { stats_inv := putRecord_preserves_statistics sys.manager bid _ inv.manager_inv.stats_inv
      , threshold_valid := And.intro (Rational.le_refl _) (Rational.le_refl _) }
  , ops_nonneg := True.intro }

theorem trace_evict_preserves_statistics_partition (m : ManagerState) (target : Nat) :
    let result := evictLowSurpriseBlocks m target
    let s := result.1.statistics
    s.high_surprise_blocks + s.low_surprise_blocks = s.total_blocks :=
  let refreshed := m.refreshRetention 0
  let current_size := refreshed.storage.count
  if h : current_size ≤ target then
    have : evictLowSurpriseBlocks m target = (refreshed, 0) := evictLowSurprise_noop_under_capacity m target h
    have : result.1.statistics = m.statistics := Eq.refl _
    have : s.high_surprise_blocks + s.low_surprise_blocks = m.statistics.high_surprise_blocks + m.statistics.low_surprise_blocks :=
      Eq.refl _
    have : m.statistics.high_surprise_blocks + m.statistics.low_surprise_blocks = m.statistics.total_blocks :=
      m.statistics.partition_le ▸ Eq.refl _
    Eq.trans this (Eq.trans (Eq.symm this) (Eq.refl _))
  else
    let to_evict := current_size - target
    let candidates : List CandidateItem := refreshed.storage.blocks.map (fun blk =>
      let priority := match refreshed.findRecord blk.block_id with
        | some rec => rec.retention_priority
        | none => Rational.zero
      { block_id := blk.block_id, priority := priority })
    if h_cand : candidates.length == 0 then
      have : evictLowSurpriseBlocks m target = (refreshed, 0) := Eq.refl _
      have : result.1.statistics = m.statistics := Eq.refl _
      have : s.high_surprise_blocks + s.low_surprise_blocks = m.statistics.high_surprise_blocks + m.statistics.low_surprise_blocks :=
        Eq.refl _
      have : m.statistics.high_surprise_blocks + m.statistics.low_surprise_blocks = m.statistics.total_blocks :=
        m.statistics.partition_le ▸ Eq.refl _
      Eq.trans this (Eq.trans (Eq.symm this) (Eq.refl _))
    else
      let sorted := partialSortCandidates candidates to_evict
      let k := min to_evict sorted.length
      let to_remove := sorted.take k
      let result_fold := to_remove.foldl (fun (acc : ManagerState × Nat) cand =>
        let (s, count) := acc
        if s.storage.containsBlock cand.block_id then
          let score := match s.findRecord cand.block_id with
            | some rec => rec.surprise_score
            | none => Rational.zero
          let s1 := { s with
            storage := s.storage.removeBlock cand.block_id
          , statistics := s.statistics.removeBlock score s.surprise_threshold }
          let s2 := s1.removeRecord cand.block_id
          let s3 := { s2 with statistics := { s2.statistics with evictions_due_to_low_surprise := s2.statistics.evictions_due_to_low_surprise + 1 } }
          (s3, count + 1)
        else (s, count)
      ) (refreshed, 0)
      let final := (result_fold.1).recomputeStatistics
      have : result.1 = final := Eq.refl _
      have : s = final.statistics := Eq.refl _
      have : final.statistics.high_surprise_blocks + final.statistics.low_surprise_blocks = final.statistics.total_blocks :=
        List.recOn to_remove
          (show refreshed.statistics.high_surprise_blocks + refreshed.statistics.low_surprise_blocks = refreshed.statistics.total_blocks from
            refreshed.statistics.partition_le ▸ Eq.refl _)
          (fun hd tl ih =>
            let (s_acc, _) := List.foldl (fun (acc : ManagerState × Nat) cand =>
              let (s, count) := acc
              if s.storage.containsBlock cand.block_id then
                let score := match s.findRecord cand.block_id with
                  | some rec => rec.surprise_score
                  | none => Rational.zero
                let s1 := { s with
                  storage := s.storage.removeBlock cand.block_id
                , statistics := s.statistics.removeBlock score s.surprise_threshold }
                let s2 := s1.removeRecord cand.block_id
                let s3 := { s2 with statistics := { s2.statistics with evictions_due_to_low_surprise := s2.statistics.evictions_due_to_low_surprise + 1 } }
                (s3, count + 1)
              else (s, count)
            ) (refreshed, 0) tl
            let score_hd := match s_acc.findRecord hd.block_id with
              | some rec => rec.surprise_score
              | none => Rational.zero
            let s1_hd := { s_acc with
              storage := s_acc.storage.removeBlock hd.block_id
            , statistics := s_acc.statistics.removeBlock score_hd s_acc.surprise_threshold }
            let s2_hd := s1_hd.removeRecord hd.block_id
            let s3_hd := { s2_hd with statistics := { s2_hd.statistics with evictions_due_to_low_surprise := s2_hd.statistics.evictions_due_to_low_surprise + 1 } }
            let final_hd := s3_hd.recomputeStatistics
            have h_remove_total : final_hd.statistics.total_blocks = s_acc.statistics.total_blocks - 1 :=
              statistics_removeBlock_total s_acc.statistics score_hd s_acc.surprise_threshold (Nat.zero_lt_succ _)
            have h_remove_high_or_low : final_hd.statistics.high_surprise_blocks + final_hd.statistics.low_surprise_blocks = s_acc.statistics.high_surprise_blocks + s_acc.statistics.low_surprise_blocks :=
              match decide (Rational.gt score_hd s_acc.surprise_threshold) with
              | true =>
                have : final_hd.statistics.high_surprise_blocks = s_acc.statistics.high_surprise_blocks - 1 := Eq.refl _
                have : final_hd.statistics.low_surprise_blocks = s_acc.statistics.low_surprise_blocks := Eq.refl _
                Nat.sub_add_cancel (Nat.zero_lt_succ _) ▸ Eq.refl _
              | false =>
                have : final_hd.statistics.low_surprise_blocks = s_acc.statistics.low_surprise_blocks - 1 := Eq.refl _
                have : final_hd.statistics.high_surprise_blocks = s_acc.statistics.high_surprise_blocks := Eq.refl _
                Nat.sub_add_cancel (Nat.zero_lt_succ _) ▸ Eq.refl _
            Eq.trans (congrArg (· + final_hd.statistics.low_surprise_blocks) h_remove_high_or_low)
              (Eq.trans h_remove_total ih))
        have h_final_total : final.statistics.total_blocks = result_fold.1.statistics.total_blocks := Eq.refl _
        Eq.trans (Eq.symm h_final_total) (Eq.trans this (Eq.refl _))

theorem applyTrace_preserves_invariant_structural (trace : SystemTrace) (sys : SurpriseSystemState)
    (inv : TraceInvariant sys) :
    match applyTrace trace sys with
    | Except.ok final => TraceInvariant final
    | Except.error _ => True :=
  SystemTrace.recOn trace
    (show TraceInvariant sys from inv)
    (fun prev data ih =>
      match h_prev : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        match prev_state.storeData data with
        | Except.ok (new_state, _) =>
          have h_inv_prev : TraceInvariant prev_state := ih
          have h_inv_new : TraceInvariant new_state :=
            { manager_inv := h_inv_prev.manager_inv
            , ops_nonneg := True.intro }
          h_inv_new
        | Except.error _ => True.intro)
    (fun prev target ih =>
      match h_prev : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        have h_inv_prev : TraceInvariant prev_state := ih
        have h_inv_new : TraceInvariant (prev_state.evict target).1 :=
          { manager_inv := h_inv_prev.manager_inv
          , ops_nonneg := True.intro }
        h_inv_new)
    (fun prev threshold ih =>
      match h_prev : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        have h_inv_prev : TraceInvariant prev_state := ih
        trace_setThreshold_preserves_invariant prev_state threshold h_inv_prev)
    (fun prev blockId ih =>
      match h_prev : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        match prev_state.manager.findRecord blockId with
        | some rec => trace_access_preserves_invariant prev_state blockId rec ih (Eq.refl _)
        | none => True.intro)

def traceLength : SystemTrace → Nat
  | SystemTrace.empty => 0
  | SystemTrace.storeOp prev _ => traceLength prev + 1
  | SystemTrace.evictOp prev _ => traceLength prev + 1
  | SystemTrace.setThresholdOp prev _ => traceLength prev + 1
  | SystemTrace.accessOp prev _ => traceLength prev + 1

theorem traceLength_empty : traceLength SystemTrace.empty = 0 := Eq.refl _

theorem traceLength_storeOp (prev : SystemTrace) (data : List UInt8) :
    traceLength (SystemTrace.storeOp prev data) = traceLength prev + 1 := Eq.refl _

theorem traceLength_evictOp (prev : SystemTrace) (target : Nat) :
    traceLength (SystemTrace.evictOp prev target) = traceLength prev + 1 := Eq.refl _

theorem applyTrace_ok_implies_op_count_ge (trace : SystemTrace) (sys : SurpriseSystemState) :
    match applyTrace trace sys with
    | Except.ok final => final.operation_count ≥ sys.operation_count
    | Except.error _ => True :=
  SystemTrace.recOn trace
    (show sys.operation_count ≥ sys.operation_count from Nat.le_refl _)
    (fun prev data ih =>
      match h : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        match prev_state.storeData data with
        | Except.ok (new_state, _) =>
          show new_state.operation_count ≥ sys.operation_count from
            have h1 : new_state.operation_count = prev_state.operation_count + 1 := Eq.refl _
            have h2 : prev_state.operation_count ≥ sys.operation_count :=
              ih ▸ Nat.le_refl _ ▸ Nat.le_refl _
            Nat.le_trans h2 (Nat.le_succ _)
        | Except.error _ => True.intro)
    (fun prev target ih =>
      match h : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        show (prev_state.evict target).1.operation_count ≥ sys.operation_count from
          have h1 : (prev_state.evict target).1.operation_count = prev_state.operation_count + 1 :=
            system_evict_increments_ops prev_state target
          Nat.le_trans (ih ▸ Nat.le_refl _) (Nat.le_succ _))
    (fun prev threshold ih =>
      match h : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        show prev_state.operation_count + 1 ≥ sys.operation_count from
          Nat.le_trans (ih ▸ Nat.le_refl _) (Nat.le_succ _))
    (fun prev blockId ih =>
      match h : applyTrace prev sys with
      | Except.error _ => True.intro
      | Except.ok prev_state =>
        match prev_state.manager.findRecord blockId with
        | some rec =>
          show prev_state.operation_count + 1 ≥ sys.operation_count from
            Nat.le_trans (ih ▸ Nat.le_refl _) (Nat.le_succ _)
        | none => True.intro)

end StateTraceInvariants

section TemporalMonotonicity

theorem strict_temporal_double_access (r : SurpriseRecord) (t1 t2 : Int) :
    let r1 := r.recordAccess t1
    let r2 := r1.recordAccess t2
    r2.access_frequency = r.access_frequency + 2 :=
  show (r.recordAccess t1).recordAccess t2 |>.access_frequency = r.access_frequency + 2 from
    have h1 : (r.recordAccess t1).access_frequency = r.access_frequency + 1 :=
      surpriseRecord_recordAccess_frequency r t1
    have h2 : ((r.recordAccess t1).recordAccess t2).access_frequency = (r.recordAccess t1).access_frequency + 1 :=
      surpriseRecord_recordAccess_frequency (r.recordAccess t1) t2
    Eq.trans h2 (Eq.trans (congrArg (· + 1) h1) (Nat.add_assoc r.access_frequency 1 1))

theorem strict_temporal_double_access_time (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).last_access_time = t2 :=
  surpriseRecord_recordAccess_time (r.recordAccess t1) t2

theorem strict_temporal_double_access_creation (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).creation_time = r.creation_time :=
  Eq.trans
    (surpriseRecord_recordAccess_preserves_creation (r.recordAccess t1) t2)
    (surpriseRecord_recordAccess_preserves_creation r t1)

theorem strict_temporal_double_access_score (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).surprise_score = r.surprise_score :=
  Eq.trans
    (surpriseRecord_recordAccess_preserves_score (r.recordAccess t1) t2)
    (surpriseRecord_recordAccess_preserves_score r t1)

theorem strict_temporal_double_access_block_id (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).block_id = r.block_id :=
  Eq.trans
    (surpriseRecord_recordAccess_preserves_block_id (r.recordAccess t1) t2)
    (surpriseRecord_recordAccess_preserves_block_id r t1)

theorem temporal_frequency_strictly_increasing (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).access_frequency > r.access_frequency :=
  have h : ((r.recordAccess t1).recordAccess t2).access_frequency = r.access_frequency + 2 :=
    strict_temporal_double_access r t1 t2
  Eq.subst (motive := fun n => n > r.access_frequency) (Eq.symm h)
    (show r.access_frequency + 2 > r.access_frequency from
      Nat.lt_of_lt_of_le (Nat.lt_succ_of_le (Nat.le_refl _)) (Nat.succ_le_succ (Nat.le_succ _)))

theorem temporal_frequency_plus_2 (bid : BlockId) (score : Rational) (t0 t1 t2 : Int) :
    let r := SurpriseRecord.init bid score t0
    let r_final := (r.recordAccess t1).recordAccess t2
    r_final.access_frequency = 3 ∧
    r_final.last_access_time = t2 ∧
    r_final.creation_time = t0 ∧
    r_final.surprise_score = score ∧
    r_final.block_id = bid :=
  And.intro (Eq.refl _)
    (And.intro (Eq.refl _)
      (And.intro (Eq.refl _)
        (And.intro (Eq.refl _)
          (Eq.refl _))))

theorem temporal_n_accesses_frequency (r : SurpriseRecord) (times : List Int) :
    (times.foldl (fun acc t => acc.recordAccess t) r).access_frequency =
    r.access_frequency + times.length :=
  List.recOn times
    (show r.access_frequency = r.access_frequency + 0 from Eq.symm (Nat.add_zero _))
    (fun t rest ih =>
      show (rest.foldl (fun acc ti => acc.recordAccess ti) (r.recordAccess t)).access_frequency =
           r.access_frequency + (rest.length + 1) from
        have h_base : (r.recordAccess t).access_frequency = r.access_frequency + 1 :=
          surpriseRecord_recordAccess_frequency r t
        have h_rest : (rest.foldl (fun acc ti => acc.recordAccess ti) (r.recordAccess t)).access_frequency =
                      (r.recordAccess t).access_frequency + rest.length :=
          temporal_n_accesses_frequency (r.recordAccess t) rest
        Eq.trans h_rest (Eq.trans (congrArg (· + rest.length) h_base)
          (Eq.trans (Nat.add_assoc r.access_frequency 1 rest.length)
            (congrArg (r.access_frequency + ·) (Nat.add_comm 1 rest.length)))))

theorem temporal_n_accesses_last_time (r : SurpriseRecord) (times : List Int) (h_ne : times ≠ []) :
    (times.foldl (fun acc t => acc.recordAccess t) r).last_access_time =
    match times.getLast h_ne with
    | t_last => t_last :=
  List.recOn times
    (absurd (Eq.refl ([] : List Int)) h_ne)
    (fun t rest ih =>
      match rest, ih with
      | [], _ =>
        show (r.recordAccess t).last_access_time = t from
          surpriseRecord_recordAccess_time r t
      | t2 :: rest2, _ =>
        show ((t2 :: rest2).foldl (fun acc ti => acc.recordAccess ti) (r.recordAccess t)).last_access_time =
             (t :: t2 :: rest2).getLast h_ne from
          have : (t :: t2 :: rest2).getLast h_ne = (t2 :: rest2).getLast (List.cons_ne_nil t2 rest2) :=
            List.getLast_cons h_ne
          Eq.refl _ ▸ Eq.refl _)

theorem temporal_monotone_frequency_chain (bid : BlockId) (score : Rational) (t0 : Int) (times : List Int) :
    let r := SurpriseRecord.init bid score t0
    (times.foldl (fun acc t => acc.recordAccess t) r).access_frequency = 1 + times.length :=
  show (times.foldl (fun acc t => acc.recordAccess t) (SurpriseRecord.init bid score t0)).access_frequency =
       1 + times.length from
    have h := temporal_n_accesses_frequency (SurpriseRecord.init bid score t0) times
    Eq.trans h (congrArg (· + times.length) (surpriseRecord_init_frequency bid score t0))

theorem temporal_frequency_strict_gt_initial (r : SurpriseRecord) (t : Int) :
    (r.recordAccess t).access_frequency > r.access_frequency :=
  surpriseRecord_access_monotone r t

theorem temporal_double_frequency_strict_gt (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).access_frequency > (r.recordAccess t1).access_frequency :=
  surpriseRecord_access_monotone (r.recordAccess t1) t2

theorem temporal_double_access_gt_initial_by_2 (r : SurpriseRecord) (t1 t2 : Int) :
    ((r.recordAccess t1).recordAccess t2).access_frequency = r.access_frequency + 2 :=
  strict_temporal_double_access r t1 t2

theorem temporal_creation_time_never_changes (r : SurpriseRecord) (times : List Int) :
    (times.foldl (fun acc t => acc.recordAccess t) r).creation_time = r.creation_time :=
  List.recOn times
    (Eq.refl _)
    (fun t rest ih =>
      show (rest.foldl (fun acc ti => acc.recordAccess ti) (r.recordAccess t)).creation_time = r.creation_time from
        Eq.trans
          (temporal_creation_time_never_changes (r.recordAccess t) rest)
          (surpriseRecord_recordAccess_preserves_creation r t))

theorem temporal_surprise_score_never_changes (r : SurpriseRecord) (times : List Int) :
    (times.foldl (fun acc t => acc.recordAccess t) r).surprise_score = r.surprise_score :=
  List.recOn times
    (Eq.refl _)
    (fun t rest ih =>
      show (rest.foldl (fun acc ti => acc.recordAccess ti) (r.recordAccess t)).surprise_score = r.surprise_score from
        Eq.trans
          (temporal_surprise_score_never_changes (r.recordAccess t) rest)
          (surpriseRecord_recordAccess_preserves_score r t))

theorem temporal_block_id_never_changes (r : SurpriseRecord) (times : List Int) :
    (times.foldl (fun acc t => acc.recordAccess t) r).block_id = r.block_id :=
  List.recOn times
    (Eq.refl _)
    (fun t rest ih =>
      show (rest.foldl (fun acc ti => acc.recordAccess ti) (r.recordAccess t)).block_id = r.block_id from
        Eq.trans
          (temporal_block_id_never_changes (r.recordAccess t) rest)
          (surpriseRecord_recordAccess_preserves_block_id r t))

end TemporalMonotonicity

section AdditionalDeepInvariants

theorem filter_complement_partition {α : Type} (l : List α) (p : α → Bool) :
    (l.filter p).length + (l.filter (fun x => !p x)).length = l.length :=
  List.recOn l
    (Eq.refl _)
    (fun hd tl ih =>
      match h : p hd with
      | true =>
        have h_neg : !p hd = false := show !true = false from Eq.refl _
        show ((hd :: tl).filter p).length + ((hd :: tl).filter (fun x => !p x)).length = (hd :: tl).length from
          have h1 : (hd :: tl).filter p = hd :: tl.filter p :=
            Eq.refl _
          have h2 : (hd :: tl).filter (fun x => !p x) = tl.filter (fun x => !p x) :=
            Eq.refl _
          Eq.trans (congrArg (· + _) (congrArg List.length h1))
            (Eq.trans (congrArg (_ + ·) (congrArg List.length h2))
              (show (tl.filter p).length + 1 + (tl.filter (fun x => !p x)).length = tl.length + 1 from
                Eq.trans (Nat.add_right_comm _ 1 _) (congrArg (· + 1) ih)))
      | false =>
        have h_neg : !p hd = true := show !false = true from Eq.refl _
        show ((hd :: tl).filter p).length + ((hd :: tl).filter (fun x => !p x)).length = (hd :: tl).length from
          have h1 : (hd :: tl).filter p = tl.filter p :=
            Eq.refl _
          have h2 : (hd :: tl).filter (fun x => !p x) = hd :: tl.filter (fun x => !p x) :=
            Eq.refl _
          Eq.trans (congrArg (· + _) (congrArg List.length h1))
            (Eq.trans (congrArg (_ + ·) (congrArg List.length h2))
              (show (tl.filter p).length + ((tl.filter (fun x => !p x)).length + 1) = tl.length + 1 from
                Eq.trans (Eq.symm (Nat.add_assoc _ _ 1)) (congrArg (· + 1) ih))))

theorem map_preserves_none_membership {α β : Type} [BEq α] [BEq β] (l : List α) (f : α → β) (x : α)
    (hf : ∀ y, f y = f x → y = x)
    (h : ¬(x ∈ l)) :
    ¬(f x ∈ l.map f) :=
  fun hmem =>
    have h_mem_map : x ∈ l := List.mem_of_mem_map hf hmem
    absurd h_mem_map h

theorem storage_remove_decreases_count (st : StorageState) (bid : BlockId) :
    (st.removeBlock bid).count ≤ st.count :=
  show (st.blocks.filter (fun b => !(b.block_id == bid))).length ≤ st.blocks.length from
    List.length_filter_le _ _

theorem storage_store_then_remove_count (st : StorageState) (data : List UInt8) :
    let (st', bid) := st.store data
    (st'.removeBlock bid).count ≤ st.count + 1 :=
  show ((st.store data).1.removeBlock (st.store data).2).count ≤ st.count + 1 from
    Nat.le_trans
      (storage_remove_decreases_count (st.store data).1 (st.store data).2)
      (Nat.le_of_eq (storageState_store_count st data))

theorem rational_sub_add_cancel_num (a b : Rational) :
    (Rational.add (Rational.sub a b) b).num =
    (a.num * Int.ofNat b.den - b.num * Int.ofNat a.den) * Int.ofNat b.den +
    b.num * Int.ofNat (a.den * b.den) :=
  Eq.refl _

theorem nat_add_sub_cancel (a b : Nat) (h : b ≤ a) :
    a - b + b = a :=
  Nat.sub_add_cancel h

theorem statistics_invariant_preserved_by_addBlock_sequence
    (threshold : Rational) (scores : List Rational) :
    StatisticsInvariant (scores.foldl (fun acc s => acc.addBlock s threshold) (SurpriseMemoryStatistics.init threshold)) :=
  List.recOn scores
    (statistics_invariant_init threshold)
    (fun score rest ih =>
      have base_inv : StatisticsInvariant ((SurpriseMemoryStatistics.init threshold).addBlock score threshold) :=
        statistics_invariant_addBlock _ _ _ (statistics_invariant_init threshold)
      List.recOn rest
        base_inv
        (fun s2 rest2 _ =>
          statistics_invariant_addBlock _ _ _ (
            statistics_invariant_addBlock _ _ _ base_inv)))

end AdditionalDeepInvariants

end SurpriseMemory
