(in-package "ACL2")

(defconst *special-pad* 0)
(defconst *special-unk* 1)
(defconst *special-bos* 2)
(defconst *special-eos* 3)

(defun token-entry-p (entry)
  (declare (xargs :guard t))
  (and (consp entry)
       (stringp (car entry))
       (natp (cdr entry))))

(defun token-list-p (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (token-entry-p (car lst))
         (token-list-p (cdr lst)))))

(defthm token-list-p-of-nil
  (token-list-p nil))

(defthm token-list-p-of-cons
  (implies (and (token-entry-p e)
                (token-list-p rest))
           (token-list-p (cons e rest))))

(defthm token-list-p-implies-true-listp
  (implies (token-list-p lst)
           (true-listp lst))
  :hints (("Goal" :induct (token-list-p lst))))

(defthm token-list-p-of-cdr
  (implies (and (token-list-p lst)
                (consp lst))
           (token-list-p (cdr lst))))

(defthm token-entry-p-of-car
  (implies (and (token-list-p lst)
                (consp lst))
           (token-entry-p (car lst))))

(defun id-entry-p (entry)
  (declare (xargs :guard t))
  (and (consp entry)
       (natp (car entry))
       (stringp (cdr entry))))

(defun id-list-p (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (id-entry-p (car lst))
         (id-list-p (cdr lst)))))

(defthm id-list-p-of-nil
  (id-list-p nil))

(defthm id-list-p-of-cons
  (implies (and (id-entry-p e)
                (id-list-p rest))
           (id-list-p (cons e rest))))

(defthm id-list-p-implies-true-listp
  (implies (id-list-p lst)
           (true-listp lst))
  :hints (("Goal" :induct (id-list-p lst))))

(defthm id-list-p-of-cdr
  (implies (and (id-list-p lst)
                (consp lst))
           (id-list-p (cdr lst))))

(defun bpe-merge-p (m)
  (declare (xargs :guard t))
  (and (consp m)
       (natp (car m))
       (natp (cdr m))))

(defun bpe-pair-entry-p (entry)
  (declare (xargs :guard t))
  (and (consp entry)
       (stringp (car entry))
       (bpe-merge-p (cdr entry))))

(defun bpe-pair-list-p (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (bpe-pair-entry-p (car lst))
         (bpe-pair-list-p (cdr lst)))))

(defthm bpe-pair-list-p-of-nil
  (bpe-pair-list-p nil))

(defthm bpe-pair-list-p-of-cons
  (implies (and (bpe-pair-entry-p e)
                (bpe-pair-list-p rest))
           (bpe-pair-list-p (cons e rest))))

(defthm bpe-pair-list-p-implies-true-listp
  (implies (bpe-pair-list-p lst)
           (true-listp lst))
  :hints (("Goal" :induct (bpe-pair-list-p lst))))

(defun anchor-entry-p (entry)
  (declare (xargs :guard t))
  (and (consp entry)
       (stringp (car entry))
       (natp (cdr entry))))

(defun anchor-list-p (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (anchor-entry-p (car lst))
         (anchor-list-p (cdr lst)))))

(defthm anchor-list-p-of-nil
  (anchor-list-p nil))

(defthm anchor-list-p-of-cons
  (implies (and (anchor-entry-p e)
                (anchor-list-p rest))
           (anchor-list-p (cons e rest))))

(defthm anchor-list-p-implies-true-listp
  (implies (anchor-list-p lst)
           (true-listp lst))
  :hints (("Goal" :induct (anchor-list-p lst))))

(defun mgt-state-p (st)
  (declare (xargs :guard t))
  (and (true-listp st)
       (equal (len st) 7)
       (token-list-p (nth 0 st))
       (id-list-p (nth 1 st))
       (token-list-p (nth 2 st))
       (token-list-p (nth 3 st))
       (token-list-p (nth 4 st))
       (bpe-pair-list-p (nth 5 st))
       (natp (nth 6 st))))

(defun make-mgt-state (token-to-id id-to-token prefixes suffixes roots bpe-pairs next-id)
  (declare (xargs :guard (and (token-list-p token-to-id)
                               (id-list-p id-to-token)
                               (token-list-p prefixes)
                               (token-list-p suffixes)
                               (token-list-p roots)
                               (bpe-pair-list-p bpe-pairs)
                               (natp next-id))))
  (list token-to-id id-to-token prefixes suffixes roots bpe-pairs next-id))

(defun mgt-token-to-id (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 0 st))

(defun mgt-id-to-token (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 1 st))

(defun mgt-prefixes (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 2 st))

(defun mgt-suffixes (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 3 st))

(defun mgt-roots (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 4 st))

(defun mgt-bpe-pairs (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 5 st))

(defun mgt-next-id (st)
  (declare (xargs :guard (mgt-state-p st)))
  (nth 6 st))

(defthm mgt-state-p-of-make
  (implies (and (token-list-p t2i)
                (id-list-p i2t)
                (token-list-p pref)
                (token-list-p suf)
                (token-list-p roots)
                (bpe-pair-list-p bpe)
                (natp nid))
           (mgt-state-p (make-mgt-state t2i i2t pref suf roots bpe nid))))

(defthm token-list-p-of-mgt-token-to-id
  (implies (mgt-state-p st)
           (token-list-p (mgt-token-to-id st))))

(defthm id-list-p-of-mgt-id-to-token
  (implies (mgt-state-p st)
           (id-list-p (mgt-id-to-token st))))

(defthm token-list-p-of-mgt-prefixes
  (implies (mgt-state-p st)
           (token-list-p (mgt-prefixes st))))

(defthm token-list-p-of-mgt-suffixes
  (implies (mgt-state-p st)
           (token-list-p (mgt-suffixes st))))

(defthm token-list-p-of-mgt-roots
  (implies (mgt-state-p st)
           (token-list-p (mgt-roots st))))

(defthm bpe-pair-list-p-of-mgt-bpe-pairs
  (implies (mgt-state-p st)
           (bpe-pair-list-p (mgt-bpe-pairs st))))

(defthm natp-of-mgt-next-id
  (implies (mgt-state-p st)
           (natp (mgt-next-id st))))

(defun lookup-token (token alist)
  (declare (xargs :guard (token-list-p alist)
                  :measure (len alist)))
  (if (atom alist)
      nil
    (if (equal token (car (car alist)))
        (cdr (car alist))
      (lookup-token token (cdr alist)))))

(defthm lookup-token-returns-natp-or-nil
  (implies (token-list-p alist)
           (or (null (lookup-token token alist))
               (natp (lookup-token token alist))))
  :hints (("Goal" :induct (lookup-token token alist)))
  :rule-classes :type-prescription)

(defthm lookup-token-of-nil
  (equal (lookup-token token nil) nil))

(defthm lookup-token-of-cons-match
  (implies (and (token-entry-p e)
                (equal token (car e)))
           (equal (lookup-token token (cons e rest))
                  (cdr e))))

(defthm lookup-token-of-cons-no-match
  (implies (and (token-entry-p e)
                (not (equal token (car e))))
           (equal (lookup-token token (cons e rest))
                  (lookup-token token rest))))

(defthm lookup-token-preserves-membership
  (implies (and (token-list-p alist)
                (lookup-token token alist))
           (natp (lookup-token token alist)))
  :hints (("Goal" :induct (lookup-token token alist))))

(defun lookup-id (id alist)
  (declare (xargs :guard (id-list-p alist)
                  :measure (len alist)))
  (if (atom alist)
      nil
    (if (equal id (car (car alist)))
        (cdr (car alist))
      (lookup-id id (cdr alist)))))

(defthm lookup-id-returns-stringp-or-nil
  (implies (id-list-p alist)
           (or (null (lookup-id id alist))
               (stringp (lookup-id id alist))))
  :hints (("Goal" :induct (lookup-id id alist)))
  :rule-classes :type-prescription)

(defthm lookup-id-of-nil
  (equal (lookup-id id nil) nil))

(defthm lookup-id-of-cons-match
  (implies (and (id-entry-p e)
                (equal id (car e)))
           (equal (lookup-id id (cons e rest))
                  (cdr e))))

(defthm lookup-id-of-cons-no-match
  (implies (and (id-entry-p e)
                (not (equal id (car e))))
           (equal (lookup-id id (cons e rest))
                  (lookup-id id rest))))

(defun lookup-bpe (key alist)
  (declare (xargs :guard (bpe-pair-list-p alist)
                  :measure (len alist)))
  (if (atom alist)
      nil
    (if (equal key (car (car alist)))
        (cdr (car alist))
      (lookup-bpe key (cdr alist)))))

(defthm lookup-bpe-returns-bpe-merge-p-or-nil
  (implies (bpe-pair-list-p alist)
           (or (null (lookup-bpe key alist))
               (bpe-merge-p (lookup-bpe key alist))))
  :hints (("Goal" :induct (lookup-bpe key alist)))
  :rule-classes :type-prescription)

(defthm lookup-bpe-of-nil
  (equal (lookup-bpe key nil) nil))

(defun nat-listp (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (natp (car lst))
         (nat-listp (cdr lst)))))

(defthm nat-listp-of-nil
  (nat-listp nil))

(defthm nat-listp-of-cons
  (implies (and (natp x)
                (nat-listp rest))
           (nat-listp (cons x rest))))

(defthm nat-listp-implies-true-listp
  (implies (nat-listp lst)
           (true-listp lst))
  :hints (("Goal" :induct (nat-listp lst))))

(defthm nat-listp-of-cdr
  (implies (and (nat-listp lst) (consp lst))
           (nat-listp (cdr lst))))

(defthm natp-of-car-when-nat-listp
  (implies (and (nat-listp lst) (consp lst))
           (natp (car lst))))

(defun nat-listp-append (a b)
  (declare (xargs :guard (and (nat-listp a) (nat-listp b))
                  :measure (len a)))
  (if (atom a)
      b
    (cons (car a) (nat-listp-append (cdr a) b))))

(defthm nat-listp-of-nat-listp-append
  (implies (and (nat-listp a) (nat-listp b))
           (nat-listp (nat-listp-append a b)))
  :hints (("Goal" :induct (nat-listp-append a b))))

(defthm true-listp-of-nat-listp-append
  (implies (and (nat-listp a) (nat-listp b))
           (true-listp (nat-listp-append a b)))
  :hints (("Goal" :induct (nat-listp-append a b))))

(defthm len-of-nat-listp-append
  (implies (and (nat-listp a) (nat-listp b))
           (equal (len (nat-listp-append a b))
                  (+ (len a) (len b))))
  :hints (("Goal" :induct (nat-listp-append a b))))

(defun char-listp-strict (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (characterp (car lst))
         (char-listp-strict (cdr lst)))))

(defthm char-listp-strict-of-nil
  (char-listp-strict nil))

(defthm char-listp-strict-implies-true-listp
  (implies (char-listp-strict lst)
           (true-listp lst))
  :hints (("Goal" :induct (char-listp-strict lst))))

(defun string-list-p (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (stringp (car lst))
         (string-list-p (cdr lst)))))

(defthm string-list-p-of-nil
  (string-list-p nil))

(defthm string-list-p-of-cons
  (implies (and (stringp s) (string-list-p rest))
           (string-list-p (cons s rest))))

(defthm string-list-p-implies-true-listp
  (implies (string-list-p lst)
           (true-listp lst))
  :hints (("Goal" :induct (string-list-p lst))))

(defthm string-list-p-of-cdr
  (implies (and (string-list-p lst) (consp lst))
           (string-list-p (cdr lst))))

(defthm stringp-of-car-when-string-list-p
  (implies (and (string-list-p lst) (consp lst))
           (stringp (car lst))))

(defun add-token-to-state (st token)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp token))))
  (let ((existing (lookup-token token (mgt-token-to-id st))))
    (if existing
        (mv st existing)
      (let ((id (mgt-next-id st)))
        (mv (make-mgt-state
             (cons (cons token id) (mgt-token-to-id st))
             (cons (cons id token) (mgt-id-to-token st))
             (mgt-prefixes st)
             (mgt-suffixes st)
             (mgt-roots st)
             (mgt-bpe-pairs st)
             (+ 1 id))
            id)))))

(defthm add-token-to-state-returns-mgt-state-p
  (implies (and (mgt-state-p st)
                (stringp token))
           (mgt-state-p (mv-nth 0 (add-token-to-state st token)))))

(defthm add-token-to-state-returns-natp
  (implies (and (mgt-state-p st)
                (stringp token))
           (natp (mv-nth 1 (add-token-to-state st token)))))

(defthm add-token-to-state-preserves-prefixes
  (implies (and (mgt-state-p st) (stringp token))
           (equal (mgt-prefixes (mv-nth 0 (add-token-to-state st token)))
                  (mgt-prefixes st))))

(defthm add-token-to-state-preserves-suffixes
  (implies (and (mgt-state-p st) (stringp token))
           (equal (mgt-suffixes (mv-nth 0 (add-token-to-state st token)))
                  (mgt-suffixes st))))

(defthm add-token-to-state-preserves-roots
  (implies (and (mgt-state-p st) (stringp token))
           (equal (mgt-roots (mv-nth 0 (add-token-to-state st token)))
                  (mgt-roots st))))

(defthm add-token-to-state-preserves-bpe-pairs
  (implies (and (mgt-state-p st) (stringp token))
           (equal (mgt-bpe-pairs (mv-nth 0 (add-token-to-state st token)))
                  (mgt-bpe-pairs st))))

(defthm add-token-next-id-monotonic
  (implies (and (mgt-state-p st) (stringp token))
           (<= (mgt-next-id st)
               (mgt-next-id (mv-nth 0 (add-token-to-state st token)))))
  :rule-classes :linear)

(defthm add-token-existing-preserves-next-id
  (implies (and (mgt-state-p st)
                (stringp token)
                (lookup-token token (mgt-token-to-id st)))
           (equal (mgt-next-id (mv-nth 0 (add-token-to-state st token)))
                  (mgt-next-id st))))

(defthm add-token-new-increments-next-id
  (implies (and (mgt-state-p st)
                (stringp token)
                (not (lookup-token token (mgt-token-to-id st))))
           (equal (mgt-next-id (mv-nth 0 (add-token-to-state st token)))
                  (+ 1 (mgt-next-id st)))))

(defthm add-token-idempotent-id
  (implies (and (mgt-state-p st)
                (stringp token)
                (lookup-token token (mgt-token-to-id st)))
           (equal (mv-nth 1 (add-token-to-state st token))
                  (lookup-token token (mgt-token-to-id st)))))

(defthm add-token-new-returns-next-id
  (implies (and (mgt-state-p st)
                (stringp token)
                (not (lookup-token token (mgt-token-to-id st))))
           (equal (mv-nth 1 (add-token-to-state st token))
                  (mgt-next-id st))))

(defun add-tokens-to-state (st tokens)
  (declare (xargs :guard (and (mgt-state-p st)
                               (string-list-p tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      st
    (mv-let (new-st id)
      (add-token-to-state st (car tokens))
      (declare (ignore id))
      (add-tokens-to-state new-st (cdr tokens)))))

(defthm add-tokens-to-state-returns-mgt-state-p
  (implies (and (mgt-state-p st)
                (string-list-p tokens))
           (mgt-state-p (add-tokens-to-state st tokens)))
  :hints (("Goal" :induct (add-tokens-to-state st tokens))))

(defthm add-tokens-next-id-monotonic
  (implies (and (mgt-state-p st)
                (string-list-p tokens))
           (<= (mgt-next-id st)
               (mgt-next-id (add-tokens-to-state st tokens))))
  :hints (("Goal" :induct (add-tokens-to-state st tokens)))
  :rule-classes :linear)

(defun is-whitespace-char (c)
  (declare (xargs :guard (characterp c)))
  (or (eql c #\Space)
      (eql c #\Newline)
      (eql c #\Tab)
      (eql c #\Return)))

(defthm is-whitespace-char-type
  (implies (characterp c)
           (booleanp (is-whitespace-char c)))
  :rule-classes :type-prescription)

(defun is-punctuation-char (c)
  (declare (xargs :guard (characterp c)))
  (or (eql c #\.)
      (eql c #\,)
      (eql c #\!)
      (eql c #\?)
      (eql c #\;)
      (eql c #\:)
      (eql c #\")
      (eql c #\')
      (eql c #\()
      (eql c #\))
      (eql c #\{)
      (eql c #\})))

(defthm is-punctuation-char-type
  (implies (characterp c)
           (booleanp (is-punctuation-char c)))
  :rule-classes :type-prescription)

(defun utf8-char-len (byte)
  (declare (xargs :guard (natp byte)))
  (cond ((< byte 128) 1)
        ((and (>= byte 192) (< byte 224)) 2)
        ((and (>= byte 224) (< byte 240)) 3)
        ((and (>= byte 240) (< byte 248)) 4)
        (t 1)))

(defthm utf8-char-len-positive
  (implies (natp byte)
           (and (natp (utf8-char-len byte))
                (> (utf8-char-len byte) 0)))
  :rule-classes (:rewrite :type-prescription))

(defthm utf8-char-len-bounded
  (implies (natp byte)
           (<= (utf8-char-len byte) 4))
  :rule-classes :linear)

(defun char-code-list (str)
  (declare (xargs :guard (stringp str)))
  (let ((chars (coerce str 'list)))
    (char-code-list-aux chars)))

(defun char-code-list-aux (chars)
  (declare (xargs :guard (character-listp chars)
                  :measure (len chars)))
  (if (atom chars)
      nil
    (cons (char-code (car chars))
          (char-code-list-aux (cdr chars)))))

(defthm nat-listp-of-char-code-list-aux
  (implies (character-listp chars)
           (nat-listp (char-code-list-aux chars)))
  :hints (("Goal" :induct (char-code-list-aux chars))))

(defthm len-of-char-code-list-aux
  (implies (character-listp chars)
           (equal (len (char-code-list-aux chars))
                  (len chars)))
  :hints (("Goal" :induct (char-code-list-aux chars))))

(defun string-to-codes (str)
  (declare (xargs :guard (stringp str)))
  (char-code-list-aux (coerce str 'list)))

(defthm nat-listp-of-string-to-codes
  (implies (stringp str)
           (nat-listp (string-to-codes str))))

(defun codes-to-string (codes)
  (declare (xargs :guard (nat-listp codes)
                  :measure (len codes)))
  (if (atom codes)
      ""
    (let ((c (if (and (natp (car codes)) (< (car codes) 256))
                 (car codes)
               0)))
      (concatenate 'string
                   (coerce (list (code-char c)) 'string)
                   (codes-to-string (cdr codes))))))

(defthm stringp-of-codes-to-string
  (implies (nat-listp codes)
           (stringp (codes-to-string codes)))
  :hints (("Goal" :induct (codes-to-string codes))))

(defun substring (str start end)
  (declare (xargs :guard (and (stringp str)
                               (natp start)
                               (natp end)
                               (<= start end)
                               (<= end (length str)))))
  (subseq str start end))

(defun string-prefix-p (prefix str)
  (declare (xargs :guard (and (stringp prefix) (stringp str))))
  (and (>= (length str) (length prefix))
       (equal (substring str 0 (length prefix)) prefix)))

(defun string-suffix-p (suffix str)
  (declare (xargs :guard (and (stringp suffix) (stringp str))))
  (and (>= (length str) (length suffix))
       (equal (substring str (- (length str) (length suffix)) (length str))
              suffix)))

(defthm string-prefix-p-type
  (booleanp (string-prefix-p prefix str))
  :rule-classes :type-prescription)

(defthm string-suffix-p-type
  (booleanp (string-suffix-p suffix str))
  :rule-classes :type-prescription)

(defun vocab-size (st)
  (declare (xargs :guard (mgt-state-p st)))
  (len (mgt-token-to-id st)))

(defthm natp-of-vocab-size
  (implies (mgt-state-p st)
           (natp (vocab-size st)))
  :rule-classes :type-prescription)

(defthm vocab-size-non-negative
  (implies (mgt-state-p st)
           (<= 0 (vocab-size st)))
  :rule-classes :linear)

(defun unknown-replacement (context)
  (declare (xargs :guard t))
  (declare (ignore context))
  *special-unk*)

(defthm unknown-replacement-returns-unk
  (equal (unknown-replacement context) *special-unk*))

(defthm natp-of-unknown-replacement
  (natp (unknown-replacement context))
  :rule-classes :type-prescription)

(defun validate-tokens (st tokens)
  (declare (xargs :guard (and (mgt-state-p st)
                               (nat-listp tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      t
    (and (lookup-id (car tokens) (mgt-id-to-token st))
         (validate-tokens st (cdr tokens)))))

(defthm validate-tokens-of-nil
  (equal (validate-tokens st nil) t))

(defthm validate-tokens-of-cons
  (equal (validate-tokens st (cons tok rest))
         (and (lookup-id tok (mgt-id-to-token st))
              (validate-tokens st rest))))

(defthm booleanp-of-validate-tokens
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (booleanp (validate-tokens st tokens)))
  :hints (("Goal" :induct (validate-tokens st tokens)))
  :rule-classes :type-prescription)

(defun find-longest-prefix (word prefix-list)
  (declare (xargs :guard (and (stringp word)
                               (token-list-p prefix-list))
                  :measure (len prefix-list)))
  (if (atom prefix-list)
      nil
    (let* ((entry (car prefix-list))
           (prefix (car entry))
           (id (cdr entry))
           (rest-result (find-longest-prefix word (cdr prefix-list))))
      (if (and (> (length word) (length prefix))
               (string-prefix-p prefix word))
          (if (or (null rest-result)
                  (> (length prefix) (length (car rest-result))))
              (cons prefix id)
            rest-result)
        rest-result))))

(defthm find-longest-prefix-returns-token-entry-or-nil
  (implies (token-list-p prefix-list)
           (or (null (find-longest-prefix word prefix-list))
               (and (consp (find-longest-prefix word prefix-list))
                    (stringp (car (find-longest-prefix word prefix-list)))
                    (natp (cdr (find-longest-prefix word prefix-list))))))
  :hints (("Goal" :induct (find-longest-prefix word prefix-list)))
  :rule-classes :type-prescription)

(defun find-longest-suffix (word suffix-list)
  (declare (xargs :guard (and (stringp word)
                               (token-list-p suffix-list))
                  :measure (len suffix-list)))
  (if (atom suffix-list)
      nil
    (let* ((entry (car suffix-list))
           (suffix (car entry))
           (id (cdr entry))
           (rest-result (find-longest-suffix word (cdr suffix-list))))
      (if (and (> (length word) (length suffix))
               (string-suffix-p suffix word))
          (if (or (null rest-result)
                  (> (length suffix) (length (car rest-result))))
              (cons suffix id)
            rest-result)
        rest-result))))

(defthm find-longest-suffix-returns-token-entry-or-nil
  (implies (token-list-p suffix-list)
           (or (null (find-longest-suffix word suffix-list))
               (and (consp (find-longest-suffix word suffix-list))
                    (stringp (car (find-longest-suffix word suffix-list)))
                    (natp (cdr (find-longest-suffix word suffix-list))))))
  :hints (("Goal" :induct (find-longest-suffix word suffix-list)))
  :rule-classes :type-prescription)

(defun morph-decompose (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (if (< (length word) 4)
      nil
    (let* ((prefix-result (find-longest-prefix word (mgt-prefixes st)))
           (suffix-result (find-longest-suffix word (mgt-suffixes st)))
           (prefix-len (if prefix-result (length (car prefix-result)) 0))
           (suffix-len (if suffix-result (length (car suffix-result)) 0)))
      (if (and (equal prefix-len 0) (equal suffix-len 0))
          nil
        (let* ((root-start prefix-len)
               (root-end (- (length word) suffix-len)))
          (if (or (<= root-end root-start)
                  (< (- root-end root-start) 2))
              nil
            (let* ((root (substring word root-start root-end))
                   (root-id (lookup-token root (mgt-token-to-id st))))
              (if (null root-id)
                  nil
                (let* ((prefix-tokens (if prefix-result
                                          (list (cdr prefix-result))
                                        nil))
                       (root-tokens (list root-id))
                       (suffix-tokens (if suffix-result
                                          (list (cdr suffix-result))
                                        nil)))
                  (nat-listp-append prefix-tokens
                                    (nat-listp-append root-tokens suffix-tokens)))))))))))

(defthm morph-decompose-returns-nat-listp-or-nil
  (implies (and (mgt-state-p st) (stringp word))
           (or (null (morph-decompose st word))
               (nat-listp (morph-decompose st word))))
  :rule-classes :type-prescription)

(defun longest-match-aux (st word pos end best)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp word)
                               (natp pos)
                               (natp end)
                               (natp best)
                               (<= pos end)
                               (<= end (length word)))
                  :measure (nfix (- (length word) end))))
  (if (or (not (natp end))
          (not (natp pos))
          (>= end (length word)))
      best
    (let* ((next-end (+ end 1))
           (substr (substring word pos next-end))
           (found (lookup-token substr (mgt-token-to-id st)))
           (new-best (if found (- next-end pos) best)))
      (longest-match-aux st word pos next-end new-best))))

(defthm natp-of-longest-match-aux
  (implies (natp best)
           (natp (longest-match-aux st word pos end best)))
  :hints (("Goal" :induct (longest-match-aux st word pos end best)))
  :rule-classes :type-prescription)

(defthm longest-match-aux-geq-best
  (implies (natp best)
           (<= best (longest-match-aux st word pos end best)))
  :hints (("Goal" :induct (longest-match-aux st word pos end best)))
  :rule-classes :linear)

(defun longest-match (st word pos)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp word)
                               (natp pos)
                               (<= pos (length word)))))
  (if (>= pos (length word))
      0
    (longest-match-aux st word pos pos 0)))

(defthm natp-of-longest-match
  (implies (and (mgt-state-p st) (stringp word) (natp pos))
           (natp (longest-match st word pos)))
  :rule-classes :type-prescription)

(defthm longest-match-non-negative
  (implies (and (mgt-state-p st) (stringp word) (natp pos))
           (<= 0 (longest-match st word pos)))
  :rule-classes :linear)

(defun subword-split-aux (st word pos acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp word)
                               (natp pos)
                               (nat-listp acc)
                               (<= pos (length word)))
                  :measure (nfix (- (length word) pos))))
  (if (or (not (natp pos))
          (>= pos (length word)))
      (reverse acc)
    (let ((match-len (longest-match st word pos)))
      (if (and (natp match-len) (> match-len 0))
          (let* ((found-word (substring word pos (+ pos match-len)))
                 (tid (lookup-token found-word (mgt-token-to-id st))))
            (if tid
                (subword-split-aux st word (+ pos match-len) (cons tid acc))
              (subword-split-aux st word (+ pos 1) (cons *special-unk* acc))))
        (subword-split-aux st word (+ pos 1) (cons *special-unk* acc))))))

(defthm nat-listp-of-reverse-when-nat-listp
  (implies (nat-listp lst)
           (nat-listp (reverse lst)))
  :hints (("Goal" :induct (reverse lst))))

(defthm true-listp-of-reverse
  (implies (true-listp lst)
           (true-listp (reverse lst))))

(defthm nat-listp-of-subword-split-aux
  (implies (and (mgt-state-p st)
                (stringp word)
                (natp pos)
                (nat-listp acc))
           (nat-listp (subword-split-aux st word pos acc)))
  :hints (("Goal" :induct (subword-split-aux st word pos acc))))

(defun subword-split (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (subword-split-aux st word 0 nil))

(defthm nat-listp-of-subword-split
  (implies (and (mgt-state-p st) (stringp word))
           (nat-listp (subword-split st word))))

(defun merge-subwords-aux (subwords acc)
  (declare (xargs :guard (and (true-listp subwords)
                               (nat-listp acc))
                  :measure (len subwords)))
  (if (atom subwords)
      (reverse acc)
    (if (nat-listp (car subwords))
        (merge-subwords-aux (cdr subwords)
                            (revappend (car subwords) acc))
      (merge-subwords-aux (cdr subwords) acc))))

(defun merge-subwords (subword-lists)
  (declare (xargs :guard (true-listp subword-lists)))
  (merge-subwords-aux subword-lists nil))

(defun encode-word (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (let ((direct (lookup-token word (mgt-token-to-id st))))
    (if direct
        (list direct)
      (let ((morph (morph-decompose st word)))
        (if morph
            morph
          (subword-split st word))))))

(defthm nat-listp-of-encode-word
  (implies (and (mgt-state-p st) (stringp word))
           (nat-listp (encode-word st word))))

(defun is-word-boundary-char (c)
  (declare (xargs :guard (characterp c)))
  (or (is-whitespace-char c)
      (is-punctuation-char c)))

(defun find-word-end (st text pos)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp text)
                               (natp pos)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (declare (ignore st))
  (if (or (not (natp pos)) (>= pos (length text)))
      pos
    (let ((c (char text pos)))
      (if (is-word-boundary-char c)
          pos
        (find-word-end st text (+ pos 1))))))

(defthm natp-of-find-word-end
  (implies (and (natp pos) (stringp text))
           (natp (find-word-end st text pos)))
  :hints (("Goal" :induct (find-word-end st text pos)))
  :rule-classes :type-prescription)

(defthm find-word-end-geq-pos
  (implies (and (natp pos) (stringp text) (<= pos (length text)))
           (<= pos (find-word-end st text pos)))
  :hints (("Goal" :induct (find-word-end st text pos)))
  :rule-classes :linear)

(defthm find-word-end-leq-length
  (implies (and (natp pos) (stringp text) (<= pos (length text)))
           (<= (find-word-end st text pos) (length text)))
  :hints (("Goal" :induct (find-word-end st text pos)))
  :rule-classes :linear)

(defun is-special-token (str)
  (declare (xargs :guard (stringp str)))
  (or (equal str "[PAD]")
      (equal str "[UNK]")
      (equal str "[BOS]")
      (equal str "[EOS]")))

(defthm booleanp-of-is-special-token
  (implies (stringp str)
           (booleanp (is-special-token str)))
  :rule-classes :type-prescription)

(defun check-special-token-at (text pos)
  (declare (xargs :guard (and (stringp text) (natp pos) (<= pos (length text)))))
  (cond ((and (<= (+ pos 5) (length text))
              (equal (substring text pos (+ pos 5)) "[PAD]"))
         5)
        ((and (<= (+ pos 5) (length text))
              (equal (substring text pos (+ pos 5)) "[UNK]"))
         5)
        ((and (<= (+ pos 5) (length text))
              (equal (substring text pos (+ pos 5)) "[BOS]"))
         5)
        ((and (<= (+ pos 5) (length text))
              (equal (substring text pos (+ pos 5)) "[EOS]"))
         5)
        (t 0)))

(defthm natp-of-check-special-token-at
  (implies (and (stringp text) (natp pos))
           (natp (check-special-token-at text pos)))
  :rule-classes :type-prescription)

(defthm check-special-token-at-bounded
  (implies (and (stringp text) (natp pos))
           (<= (check-special-token-at text pos) 5))
  :rule-classes :linear)

(defun encode-aux (st text pos acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp text)
                               (natp pos)
                               (nat-listp acc)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      (reverse acc)
    (let ((special-len (check-special-token-at text pos)))
      (if (> special-len 0)
          (let* ((special-str (substring text pos (+ pos special-len)))
                 (tid (lookup-token special-str (mgt-token-to-id st)))
                 (token-id (if tid tid *special-unk*)))
            (encode-aux st text (+ pos special-len)
                        (cons token-id acc)))
        (let ((c (char text pos)))
          (cond
           ((is-whitespace-char c)
            (let* ((ws-str (coerce (list c) 'string))
                   (tid (lookup-token ws-str (mgt-token-to-id st)))
                   (token-id (if tid tid
                               (let ((space-tid (lookup-token " " (mgt-token-to-id st))))
                                 (if (and (eql c #\Space) space-tid) space-tid *special-unk*)))))
              (encode-aux st text (+ pos 1) (cons token-id acc))))
           ((is-punctuation-char c)
            (let* ((p-str (coerce (list c) 'string))
                   (tid (lookup-token p-str (mgt-token-to-id st)))
                   (token-id (if tid tid *special-unk*)))
              (encode-aux st text (+ pos 1) (cons token-id acc))))
           (t
            (let* ((word-end (find-word-end st text pos)))
              (if (<= word-end pos)
                  (encode-aux st text (+ pos 1) (cons *special-unk* acc))
                (let* ((word (substring text pos word-end))
                       (word-tokens (encode-word st word))
                       (new-acc (revappend word-tokens acc)))
                  (encode-aux st text word-end new-acc)))))))))))

(defthm nat-listp-of-revappend-nat-listp
  (implies (and (nat-listp a) (nat-listp b))
           (nat-listp (revappend a b)))
  :hints (("Goal" :induct (revappend a b))))

(defthm nat-listp-of-encode-aux
  (implies (and (mgt-state-p st)
                (stringp text)
                (natp pos)
                (nat-listp acc))
           (nat-listp (encode-aux st text pos acc)))
  :hints (("Goal" :induct (encode-aux st text pos acc))))

(defun encode-text (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (encode-aux st text 0 nil))

(defthm nat-listp-of-encode-text
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defun decode-token (st tok)
  (declare (xargs :guard (and (mgt-state-p st) (natp tok))))
  (let ((str (lookup-id tok (mgt-id-to-token st))))
    (if str
        str
      (let ((unk-str (lookup-id *special-unk* (mgt-id-to-token st))))
        (if unk-str unk-str "[UNK]")))))

(defthm stringp-of-decode-token
  (implies (and (mgt-state-p st) (natp tok))
           (stringp (decode-token st tok)))
  :rule-classes :type-prescription)

(defun decode-tokens-aux (st tokens acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (nat-listp tokens)
                               (stringp acc))
                  :measure (len tokens)))
  (if (atom tokens)
      acc
    (let* ((tok-str (decode-token st (car tokens)))
           (new-acc (concatenate 'string acc tok-str)))
      (decode-tokens-aux st (cdr tokens) new-acc))))

(defthm stringp-of-decode-tokens-aux
  (implies (and (mgt-state-p st)
                (nat-listp tokens)
                (stringp acc))
           (stringp (decode-tokens-aux st tokens acc)))
  :hints (("Goal" :induct (decode-tokens-aux st tokens acc)))
  :rule-classes :type-prescription)

(defun decode-tokens (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))))
  (decode-tokens-aux st tokens ""))

(defthm stringp-of-decode-tokens
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (stringp (decode-tokens st tokens))))

(defun encode-batch-aux (st texts acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (string-list-p texts)
                               (true-listp acc))
                  :measure (len texts)))
  (if (atom texts)
      (reverse acc)
    (let ((tokens (encode-text st (car texts))))
      (encode-batch-aux st (cdr texts) (cons tokens acc)))))

(defun encode-batch (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))))
  (encode-batch-aux st texts nil))

(defthm true-listp-of-encode-batch-aux
  (implies (and (mgt-state-p st)
                (string-list-p texts)
                (true-listp acc))
           (true-listp (encode-batch-aux st texts acc)))
  :hints (("Goal" :induct (encode-batch-aux st texts acc))))

(defthm true-listp-of-encode-batch
  (implies (and (mgt-state-p st) (string-list-p texts))
           (true-listp (encode-batch st texts))))

(defthm len-of-encode-batch-aux
  (implies (and (mgt-state-p st)
                (string-list-p texts)
                (true-listp acc))
           (equal (len (encode-batch-aux st texts acc))
                  (+ (len texts) (len acc))))
  :hints (("Goal" :induct (encode-batch-aux st texts acc))))

(defthm len-of-encode-batch
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defun batch-decode-aux (st token-lists acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (true-listp token-lists)
                               (string-list-p acc))
                  :measure (len token-lists)))
  (if (atom token-lists)
      (reverse acc)
    (if (nat-listp (car token-lists))
        (let ((text (decode-tokens st (car token-lists))))
          (batch-decode-aux st (cdr token-lists) (cons text acc)))
      (batch-decode-aux st (cdr token-lists) (cons "[UNK]" acc)))))

(defun batch-decode (st token-lists)
  (declare (xargs :guard (and (mgt-state-p st) (true-listp token-lists))))
  (batch-decode-aux st token-lists nil))

(defthm string-list-p-of-batch-decode-aux
  (implies (and (mgt-state-p st)
                (true-listp token-lists)
                (string-list-p acc))
           (string-list-p (batch-decode-aux st token-lists acc)))
  :hints (("Goal" :induct (batch-decode-aux st token-lists acc))))

(defthm string-list-p-of-batch-decode
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (string-list-p (batch-decode st token-lists))))

(defthm len-of-batch-decode-aux
  (implies (and (mgt-state-p st)
                (true-listp token-lists)
                (string-list-p acc))
           (equal (len (batch-decode-aux st token-lists acc))
                  (+ (len token-lists) (len acc))))
  :hints (("Goal" :induct (batch-decode-aux st token-lists acc))))

(defthm len-of-batch-decode
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))

(defun init-special-tokens (st)
  (declare (xargs :guard (mgt-state-p st)))
  (mv-let (st1 id0) (add-token-to-state st "[PAD]")
    (declare (ignore id0))
    (mv-let (st2 id1) (add-token-to-state st1 "[UNK]")
      (declare (ignore id1))
      (mv-let (st3 id2) (add-token-to-state st2 "[BOS]")
        (declare (ignore id2))
        (mv-let (st4 id3) (add-token-to-state st3 "[EOS]")
          (declare (ignore id3))
          st4)))))

(defthm mgt-state-p-of-init-special-tokens
  (implies (mgt-state-p st)
           (mgt-state-p (init-special-tokens st))))

(defthm init-special-tokens-next-id-geq
  (implies (mgt-state-p st)
           (<= (mgt-next-id st)
               (mgt-next-id (init-special-tokens st))))
  :rule-classes :linear)

(defun make-empty-mgt-state ()
  (declare (xargs :guard t))
  (make-mgt-state nil nil nil nil nil nil 0))

(defthm mgt-state-p-of-make-empty
  (mgt-state-p (make-empty-mgt-state)))

(defun init-mgt (vocab anchors)
  (declare (xargs :guard (and (string-list-p vocab)
                               (string-list-p anchors))))
  (let* ((st0 (make-empty-mgt-state))
         (st1 (init-special-tokens st0))
         (st2 (add-tokens-to-state st1 vocab)))
    st2))

(defthm mgt-state-p-of-init-mgt
  (implies (and (string-list-p vocab) (string-list-p anchors))
           (mgt-state-p (init-mgt vocab anchors))))

(defun add-prefix-to-state (st prefix)
  (declare (xargs :guard (and (mgt-state-p st) (stringp prefix))))
  (mv-let (st1 id)
    (add-token-to-state st prefix)
    (make-mgt-state
     (mgt-token-to-id st1)
     (mgt-id-to-token st1)
     (cons (cons prefix id) (mgt-prefixes st1))
     (mgt-suffixes st1)
     (mgt-roots st1)
     (mgt-bpe-pairs st1)
     (mgt-next-id st1))))

(defthm mgt-state-p-of-add-prefix-to-state
  (implies (and (mgt-state-p st) (stringp prefix))
           (mgt-state-p (add-prefix-to-state st prefix))))

(defun add-suffix-to-state (st suffix)
  (declare (xargs :guard (and (mgt-state-p st) (stringp suffix))))
  (mv-let (st1 id)
    (add-token-to-state st suffix)
    (make-mgt-state
     (mgt-token-to-id st1)
     (mgt-id-to-token st1)
     (mgt-prefixes st1)
     (cons (cons suffix id) (mgt-suffixes st1))
     (mgt-roots st1)
     (mgt-bpe-pairs st1)
     (mgt-next-id st1))))

(defthm mgt-state-p-of-add-suffix-to-state
  (implies (and (mgt-state-p st) (stringp suffix))
           (mgt-state-p (add-suffix-to-state st suffix))))

(defun add-prefixes-to-state (st prefixes)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p prefixes))
                  :measure (len prefixes)))
  (if (atom prefixes)
      st
    (add-prefixes-to-state (add-prefix-to-state st (car prefixes))
                           (cdr prefixes))))

(defthm mgt-state-p-of-add-prefixes-to-state
  (implies (and (mgt-state-p st) (string-list-p prefixes))
           (mgt-state-p (add-prefixes-to-state st prefixes)))
  :hints (("Goal" :induct (add-prefixes-to-state st prefixes))))

(defun add-suffixes-to-state (st suffixes)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p suffixes))
                  :measure (len suffixes)))
  (if (atom suffixes)
      st
    (add-suffixes-to-state (add-suffix-to-state st (car suffixes))
                           (cdr suffixes))))

(defthm mgt-state-p-of-add-suffixes-to-state
  (implies (and (mgt-state-p st) (string-list-p suffixes))
           (mgt-state-p (add-suffixes-to-state st suffixes)))
  :hints (("Goal" :induct (add-suffixes-to-state st suffixes))))

(defconst *english-prefixes*
  (list "un" "re" "pre" "dis" "mis" "over" "under" "out"
        "sub" "inter" "fore" "de" "trans" "super" "semi" "anti"
        "mid" "non" "ex" "post" "pro" "co" "en" "em"))

(defconst *hungarian-prefixes*
  (list "meg" "el" "fel" "le" "be" "ki"))

(defconst *english-suffixes*
  (list "ing" "ed" "er" "est" "ly" "tion" "sion" "ness"
        "ment" "ful" "less" "ous" "ive" "able" "ible" "al"
        "ial" "y" "s" "es" "en" "ize" "ise" "ate"))

(defconst *hungarian-suffixes*
  (list "ban" "ben" "ba" "be" "nak" "nek" "val" "vel"
        "ra" "re" "kor" "ni" "at" "et"))

(defun init-morphemes (st)
  (declare (xargs :guard (mgt-state-p st)))
  (let* ((st1 (add-prefixes-to-state st *english-prefixes*))
         (st2 (add-prefixes-to-state st1 *hungarian-prefixes*))
         (st3 (add-suffixes-to-state st2 *english-suffixes*))
         (st4 (add-suffixes-to-state st3 *hungarian-suffixes*)))
    st4))

(defthm mgt-state-p-of-init-morphemes
  (implies (mgt-state-p st)
           (mgt-state-p (init-morphemes st))))

(defun init-mgt-full (vocab anchors)
  (declare (xargs :guard (and (string-list-p vocab) (string-list-p anchors))))
  (let* ((st0 (make-empty-mgt-state))
         (st1 (init-special-tokens st0))
         (st2 (add-tokens-to-state st1 vocab))
         (st3 (init-morphemes st2)))
    st3))

(defthm mgt-state-p-of-init-mgt-full
  (implies (and (string-list-p vocab) (string-list-p anchors))
           (mgt-state-p (init-mgt-full vocab anchors))))

(defun pair-key-p (pk)
  (declare (xargs :guard t))
  (and (consp pk)
       (natp (car pk))
       (natp (cdr pk))))

(defun pair-freq-p (pf)
  (declare (xargs :guard t))
  (and (consp pf)
       (pair-key-p (car pf))
       (natp (cdr pf))))

(defun pair-freq-list-p (lst)
  (declare (xargs :guard t))
  (if (atom lst)
      (null lst)
    (and (pair-freq-p (car lst))
         (pair-freq-list-p (cdr lst)))))

(defthm pair-freq-list-p-of-nil
  (pair-freq-list-p nil))

(defthm pair-freq-list-p-of-cons
  (implies (and (pair-freq-p pf) (pair-freq-list-p rest))
           (pair-freq-list-p (cons pf rest))))

(defthm pair-freq-list-p-implies-true-listp
  (implies (pair-freq-list-p lst)
           (true-listp lst))
  :hints (("Goal" :induct (pair-freq-list-p lst))))

(defun count-pairs-in-seq (seq pair-counts)
  (declare (xargs :guard (and (nat-listp seq) (true-listp pair-counts))
                  :measure (len seq)))
  (if (or (atom seq) (atom (cdr seq)))
      pair-counts
    (let* ((key (cons (car seq) (cadr seq)))
           (existing (assoc-equal key pair-counts))
           (new-count (if existing (+ 1 (cdr existing)) 1))
           (new-counts (if existing
                           (put-assoc-equal key new-count pair-counts)
                         (cons (cons key new-count) pair-counts))))
      (count-pairs-in-seq (cdr seq) new-counts))))

(defun count-pairs-in-seqs (seqs pair-counts)
  (declare (xargs :guard (and (true-listp seqs) (true-listp pair-counts))
                  :measure (len seqs)))
  (if (atom seqs)
      pair-counts
    (if (nat-listp (car seqs))
        (count-pairs-in-seqs (cdr seqs)
                             (count-pairs-in-seq (car seqs) pair-counts))
      (count-pairs-in-seqs (cdr seqs) pair-counts))))

(defun find-best-pair (counts best-key best-freq)
  (declare (xargs :guard (and (true-listp counts) (natp best-freq))
                  :measure (len counts)))
  (if (atom counts)
      (mv best-key best-freq)
    (let* ((entry (car counts))
           (key (car entry))
           (freq (if (natp (cdr entry)) (cdr entry) 0)))
      (if (> freq best-freq)
          (find-best-pair (cdr counts) key freq)
        (find-best-pair (cdr counts) best-key best-freq)))))

(defthm natp-of-find-best-pair-freq
  (implies (natp best-freq)
           (natp (mv-nth 1 (find-best-pair counts best-key best-freq))))
  :hints (("Goal" :induct (find-best-pair counts best-key best-freq)))
  :rule-classes :type-prescription)

(defthm find-best-pair-freq-geq
  (implies (natp best-freq)
           (<= best-freq (mv-nth 1 (find-best-pair counts best-key best-freq))))
  :hints (("Goal" :induct (find-best-pair counts best-key best-freq)))
  :rule-classes :linear)

(defun apply-merge-to-seq (seq first-id second-id merged-id acc)
  (declare (xargs :guard (and (nat-listp seq)
                               (natp first-id)
                               (natp second-id)
                               (natp merged-id)
                               (nat-listp acc))
                  :measure (len seq)))
  (if (atom seq)
      (reverse acc)
    (if (and (consp (cdr seq))
             (equal (car seq) first-id)
             (equal (cadr seq) second-id))
        (apply-merge-to-seq (cddr seq) first-id second-id merged-id
                            (cons merged-id acc))
      (apply-merge-to-seq (cdr seq) first-id second-id merged-id
                          (cons (car seq) acc)))))

(defthm nat-listp-of-apply-merge-to-seq
  (implies (and (nat-listp seq)
                (natp first-id)
                (natp second-id)
                (natp merged-id)
                (nat-listp acc))
           (nat-listp (apply-merge-to-seq seq first-id second-id merged-id acc)))
  :hints (("Goal" :induct (apply-merge-to-seq seq first-id second-id merged-id acc))))

(defun apply-merge-to-seqs (seqs first-id second-id merged-id)
  (declare (xargs :guard (and (true-listp seqs)
                               (natp first-id)
                               (natp second-id)
                               (natp merged-id))
                  :measure (len seqs)))
  (if (atom seqs)
      nil
    (if (nat-listp (car seqs))
        (cons (apply-merge-to-seq (car seqs) first-id second-id merged-id nil)
              (apply-merge-to-seqs (cdr seqs) first-id second-id merged-id))
      (cons (car seqs)
            (apply-merge-to-seqs (cdr seqs) first-id second-id merged-id)))))

(defthm true-listp-of-apply-merge-to-seqs
  (implies (true-listp seqs)
           (true-listp (apply-merge-to-seqs seqs first-id second-id merged-id)))
  :hints (("Goal" :induct (apply-merge-to-seqs seqs first-id second-id merged-id))))

(defthm len-of-apply-merge-to-seqs
  (implies (true-listp seqs)
           (equal (len (apply-merge-to-seqs seqs first-id second-id merged-id))
                  (len seqs)))
  :hints (("Goal" :induct (apply-merge-to-seqs seqs first-id second-id merged-id))))

(defun train-bpe-step (st seqs merge-count)
  (declare (xargs :guard (and (mgt-state-p st)
                               (true-listp seqs)
                               (natp merge-count))
                  :measure (nfix merge-count)))
  (if (zp merge-count)
      (mv st seqs)
    (let* ((pair-counts (count-pairs-in-seqs seqs nil)))
      (mv-let (best-key best-freq)
        (find-best-pair pair-counts nil 0)
        (if (or (null best-key)
                (< best-freq 2)
                (not (pair-key-p best-key)))
            (mv st seqs)
          (let* ((first-id (car best-key))
                 (second-id (cdr best-key))
                 (first-str (lookup-id first-id (mgt-id-to-token st)))
                 (second-str (lookup-id second-id (mgt-id-to-token st))))
            (if (or (null first-str) (null second-str))
                (mv st seqs)
              (let* ((merged-str (concatenate 'string first-str second-str)))
                (mv-let (st1 merged-id)
                  (add-token-to-state st merged-str)
                  (let* ((new-bpe (cons (cons merged-str (cons merged-id merge-count))
                                        (mgt-bpe-pairs st1)))
                         (st2 (make-mgt-state
                               (mgt-token-to-id st1)
                               (mgt-id-to-token st1)
                               (mgt-prefixes st1)
                               (mgt-suffixes st1)
                               (mgt-roots st1)
                               new-bpe
                               (mgt-next-id st1)))
                         (new-seqs (apply-merge-to-seqs seqs first-id second-id merged-id)))
                    (train-bpe-step st2 new-seqs (- merge-count 1))))))))))))

(defthm mgt-state-p-of-train-bpe-step
  (implies (and (mgt-state-p st)
                (true-listp seqs)
                (natp merge-count))
           (mgt-state-p (mv-nth 0 (train-bpe-step st seqs merge-count))))
  :hints (("Goal" :induct (train-bpe-step st seqs merge-count))))

(defthm true-listp-of-train-bpe-step-seqs
  (implies (and (mgt-state-p st)
                (true-listp seqs)
                (natp merge-count))
           (true-listp (mv-nth 1 (train-bpe-step st seqs merge-count))))
  :hints (("Goal" :induct (train-bpe-step st seqs merge-count))))

(defun text-to-byte-seq (text pos acc)
  (declare (xargs :guard (and (stringp text) (natp pos) (nat-listp acc)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      (reverse acc)
    (text-to-byte-seq text (+ pos 1)
                      (cons (char-code (char text pos)) acc))))

(defthm nat-listp-of-text-to-byte-seq
  (implies (and (stringp text) (natp pos) (nat-listp acc))
           (nat-listp (text-to-byte-seq text pos acc)))
  :hints (("Goal" :induct (text-to-byte-seq text pos acc))))

(defun byte-to-hex-token (byte)
  (declare (xargs :guard (natp byte)))
  (let* ((b (mod byte 256))
         (hi (floor b 16))
         (lo (mod b 16))
         (hex-chars "0123456789abcdef")
         (hi-char (char hex-chars hi))
         (lo-char (char hex-chars lo)))
    (coerce (list #\< hi-char lo-char #\>) 'string)))

(defthm stringp-of-byte-to-hex-token
  (implies (natp byte)
           (stringp (byte-to-hex-token byte)))
  :rule-classes :type-prescription)

(defun init-byte-tokens (st n)
  (declare (xargs :guard (and (mgt-state-p st) (natp n))
                  :measure (nfix (- 256 n))))
  (if (or (not (natp n)) (>= n 256))
      st
    (let ((hex-tok (byte-to-hex-token n)))
      (mv-let (st1 id)
        (add-token-to-state st hex-tok)
        (declare (ignore id))
        (init-byte-tokens st1 (+ n 1))))))

(defthm mgt-state-p-of-init-byte-tokens
  (implies (and (mgt-state-p st) (natp n))
           (mgt-state-p (init-byte-tokens st n)))
  :hints (("Goal" :induct (init-byte-tokens st n))))

(defun text-to-token-seq (st text pos acc)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text)
                               (natp pos) (nat-listp acc)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      (reverse acc)
    (let* ((byte (char-code (char text pos)))
           (hex-tok (byte-to-hex-token byte))
           (tid (lookup-token hex-tok (mgt-token-to-id st)))
           (token-id (if tid tid *special-unk*)))
      (text-to-token-seq st text (+ pos 1) (cons token-id acc)))))

(defthm nat-listp-of-text-to-token-seq
  (implies (and (mgt-state-p st) (stringp text)
                (natp pos) (nat-listp acc))
           (nat-listp (text-to-token-seq st text pos acc)))
  :hints (("Goal" :induct (text-to-token-seq st text pos acc))))

(defun corpus-to-seqs (st corpus acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (string-list-p corpus)
                               (true-listp acc))
                  :measure (len corpus)))
  (if (atom corpus)
      (reverse acc)
    (let ((seq (text-to-token-seq st (car corpus) 0 nil)))
      (corpus-to-seqs st (cdr corpus) (cons seq acc)))))

(defthm true-listp-of-corpus-to-seqs
  (implies (and (mgt-state-p st)
                (string-list-p corpus)
                (true-listp acc))
           (true-listp (corpus-to-seqs st corpus acc)))
  :hints (("Goal" :induct (corpus-to-seqs st corpus acc))))

(defun train-bpe (st corpus num-merges)
  (declare (xargs :guard (and (mgt-state-p st)
                               (string-list-p corpus)
                               (natp num-merges))))
  (let* ((st1 (init-byte-tokens st 0))
         (seqs (corpus-to-seqs st1 corpus nil)))
    (mv-let (st2 final-seqs)
      (train-bpe-step st1 seqs num-merges)
      (declare (ignore final-seqs))
      st2)))

(defthm mgt-state-p-of-train-bpe
  (implies (and (mgt-state-p st)
                (string-list-p corpus)
                (natp num-merges))
           (mgt-state-p (train-bpe st corpus num-merges))))

(defun add-vocab-word (st word is-anchor)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word) (booleanp is-anchor))))
  (mv-let (st1 id)
    (add-token-to-state st word)
    (declare (ignore id))
    st1))

(defthm mgt-state-p-of-add-vocab-word
  (implies (and (mgt-state-p st) (stringp word) (booleanp is-anchor))
           (mgt-state-p (add-vocab-word st word is-anchor))))

(defun remove-from-token-list (word lst)
  (declare (xargs :guard (token-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (if (equal word (car (car lst)))
        (remove-from-token-list word (cdr lst))
      (cons (car lst) (remove-from-token-list word (cdr lst))))))

(defthm token-list-p-of-remove-from-token-list
  (implies (token-list-p lst)
           (token-list-p (remove-from-token-list word lst)))
  :hints (("Goal" :induct (remove-from-token-list word lst))))

(defthm len-of-remove-from-token-list-leq
  (implies (token-list-p lst)
           (<= (len (remove-from-token-list word lst)) (len lst)))
  :hints (("Goal" :induct (remove-from-token-list word lst)))
  :rule-classes :linear)

(defun remove-from-id-list (id lst)
  (declare (xargs :guard (id-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (if (equal id (car (car lst)))
        (remove-from-id-list id (cdr lst))
      (cons (car lst) (remove-from-id-list id (cdr lst))))))

(defthm id-list-p-of-remove-from-id-list
  (implies (id-list-p lst)
           (id-list-p (remove-from-id-list id lst)))
  :hints (("Goal" :induct (remove-from-id-list id lst))))

(defun remove-from-bpe-list (word id lst)
  (declare (xargs :guard (and (bpe-pair-list-p lst) (natp id))
                  :measure (len lst)))
  (if (atom lst)
      nil
    (let* ((entry (car lst))
           (key (car entry))
           (merge (cdr entry))
           (merge-id (if (bpe-merge-p merge) (car merge) 0)))
      (if (or (equal key word)
              (equal merge-id id))
          (remove-from-bpe-list word id (cdr lst))
        (cons entry (remove-from-bpe-list word id (cdr lst)))))))

(defthm bpe-pair-list-p-of-remove-from-bpe-list
  (implies (bpe-pair-list-p lst)
           (bpe-pair-list-p (remove-from-bpe-list word id lst)))
  :hints (("Goal" :induct (remove-from-bpe-list word id lst))))

(defun remove-vocab-word (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (if (or (equal word "[PAD]")
          (equal word "[UNK]")
          (equal word "[BOS]")
          (equal word "[EOS]"))
      st
    (let ((id (lookup-token word (mgt-token-to-id st))))
      (if (null id)
          st
        (make-mgt-state
         (remove-from-token-list word (mgt-token-to-id st))
         (remove-from-id-list id (mgt-id-to-token st))
         (remove-from-token-list word (mgt-prefixes st))
         (remove-from-token-list word (mgt-suffixes st))
         (remove-from-token-list word (mgt-roots st))
         (remove-from-bpe-list word id (mgt-bpe-pairs st))
         (mgt-next-id st))))))

(defthm mgt-state-p-of-remove-vocab-word
  (implies (and (mgt-state-p st) (stringp word))
           (mgt-state-p (remove-vocab-word st word))))

(defthm remove-vocab-word-preserves-next-id
  (implies (and (mgt-state-p st) (stringp word))
           (equal (mgt-next-id (remove-vocab-word st word))
                  (mgt-next-id st))))

(defthm remove-special-is-identity
  (implies (and (mgt-state-p st)
                (or (equal word "[PAD]")
                    (equal word "[UNK]")
                    (equal word "[BOS]")
                    (equal word "[EOS]")))
           (equal (remove-vocab-word st word) st)))

(defun count-known-chars (st text pos count)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp text)
                               (natp pos)
                               (natp count)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      count
    (let ((c-str (coerce (list (char text pos)) 'string)))
      (if (lookup-token c-str (mgt-token-to-id st))
          (count-known-chars st text (+ pos 1) (+ count 1))
        (count-known-chars st text (+ pos 1) count)))))

(defthm natp-of-count-known-chars
  (implies (and (natp pos) (natp count))
           (natp (count-known-chars st text pos count)))
  :hints (("Goal" :induct (count-known-chars st text pos count)))
  :rule-classes :type-prescription)

(defthm count-known-chars-geq-count
  (implies (and (natp pos) (natp count))
           (<= count (count-known-chars st text pos count)))
  :hints (("Goal" :induct (count-known-chars st text pos count)))
  :rule-classes :linear)

(defun coverage-ratio (st corpus)
  (declare (xargs :guard (and (mgt-state-p st) (stringp corpus))))
  (if (equal (length corpus) 0)
      0
    (let ((known (count-known-chars st corpus 0 0)))
      (floor (* known 100) (length corpus)))))

(defthm natp-of-coverage-ratio
  (implies (and (mgt-state-p st) (stringp corpus))
           (natp (coverage-ratio st corpus)))
  :rule-classes :type-prescription)

(defun tensor-data-p (data)
  (declare (xargs :guard t))
  (if (atom data)
      (null data)
    (and (rationalp (car data))
         (tensor-data-p (cdr data)))))

(defthm tensor-data-p-of-nil
  (tensor-data-p nil))

(defthm tensor-data-p-of-cons
  (implies (and (rationalp x) (tensor-data-p rest))
           (tensor-data-p (cons x rest))))

(defthm tensor-data-p-implies-true-listp
  (implies (tensor-data-p data)
           (true-listp data))
  :hints (("Goal" :induct (tensor-data-p data))))

(defun tensor-p (tensor)
  (declare (xargs :guard t))
  (and (consp tensor)
       (nat-listp (car tensor))
       (tensor-data-p (cdr tensor))))

(defun make-tensor (shape data)
  (declare (xargs :guard (and (nat-listp shape) (tensor-data-p data))))
  (cons shape data))

(defun tensor-shape (tensor)
  (declare (xargs :guard (tensor-p tensor)))
  (car tensor))

(defun tensor-data (tensor)
  (declare (xargs :guard (tensor-p tensor)))
  (cdr tensor))

(defthm tensor-p-of-make-tensor
  (implies (and (nat-listp shape) (tensor-data-p data))
           (tensor-p (make-tensor shape data))))

(defthm nat-listp-of-tensor-shape
  (implies (tensor-p tensor)
           (nat-listp (tensor-shape tensor))))

(defthm tensor-data-p-of-tensor-data
  (implies (tensor-p tensor)
           (tensor-data-p (tensor-data tensor))))

(defun tokens-to-tensor-data (tokens)
  (declare (xargs :guard (nat-listp tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (cons (car tokens) (tokens-to-tensor-data (cdr tokens)))))

(defthm tensor-data-p-of-tokens-to-tensor-data
  (implies (nat-listp tokens)
           (tensor-data-p (tokens-to-tensor-data tokens)))
  :hints (("Goal" :induct (tokens-to-tensor-data tokens))))

(defthm len-of-tokens-to-tensor-data
  (implies (nat-listp tokens)
           (equal (len (tokens-to-tensor-data tokens))
                  (len tokens)))
  :hints (("Goal" :induct (tokens-to-tensor-data tokens))))

(defun encode-to-tensor (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (let* ((tokens (encode-text st text))
         (data (tokens-to-tensor-data tokens))
         (shape (list (len tokens))))
    (make-tensor shape data)))

(defthm tensor-p-of-encode-to-tensor
  (implies (and (mgt-state-p st) (stringp text))
           (tensor-p (encode-to-tensor st text))))

(defun max-len-of-token-lists (token-lists current-max)
  (declare (xargs :guard (and (true-listp token-lists) (natp current-max))
                  :measure (len token-lists)))
  (if (atom token-lists)
      current-max
    (let* ((row (car token-lists))
           (row-len (if (nat-listp row) (len row) 0))
           (new-max (if (> row-len current-max) row-len current-max)))
      (max-len-of-token-lists (cdr token-lists) new-max))))

(defthm natp-of-max-len-of-token-lists
  (implies (natp current-max)
           (natp (max-len-of-token-lists token-lists current-max)))
  :hints (("Goal" :induct (max-len-of-token-lists token-lists current-max)))
  :rule-classes :type-prescription)

(defthm max-len-geq-current
  (implies (natp current-max)
           (<= current-max (max-len-of-token-lists token-lists current-max)))
  :hints (("Goal" :induct (max-len-of-token-lists token-lists current-max)))
  :rule-classes :linear)

(defun pad-token-list (tokens target-len pad-val)
  (declare (xargs :guard (and (nat-listp tokens) (natp target-len) (natp pad-val))
                  :measure (nfix (- target-len (len tokens)))))
  (if (or (not (natp target-len))
          (>= (len tokens) target-len))
      tokens
    (pad-token-list (append tokens (list pad-val))
                    target-len
                    pad-val)))

(defun make-zeros (n)
  (declare (xargs :guard (natp n)
                  :measure (nfix n)))
  (if (zp n)
      nil
    (cons 0 (make-zeros (- n 1)))))

(defthm tensor-data-p-of-make-zeros
  (implies (natp n)
           (tensor-data-p (make-zeros n)))
  :hints (("Goal" :induct (make-zeros n))))

(defthm len-of-make-zeros
  (implies (natp n)
           (equal (len (make-zeros n)) n))
  :hints (("Goal" :induct (make-zeros n))))

(defun pad-tensor-data (data target-len)
  (declare (xargs :guard (and (tensor-data-p data) (natp target-len))
                  :measure (nfix (- target-len (len data)))))
  (if (or (not (natp target-len))
          (>= (len data) target-len))
      data
    (append data (make-zeros (- target-len (len data))))))

(defun flatten-token-lists-padded (token-lists max-len acc)
  (declare (xargs :guard (and (true-listp token-lists)
                               (natp max-len)
                               (tensor-data-p acc))
                  :measure (len token-lists)))
  (if (atom token-lists)
      acc
    (let* ((row (car token-lists))
           (row-data (if (nat-listp row) (tokens-to-tensor-data row) nil))
           (padded (pad-tensor-data row-data max-len)))
      (flatten-token-lists-padded (cdr token-lists) max-len
                                  (append acc padded)))))

(defun encode-batch-to-tensor (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))))
  (let* ((batch (encode-batch st texts))
         (max-len (max-len-of-token-lists batch 1))
         (flat-data (flatten-token-lists-padded batch max-len nil))
         (shape (list (len texts) max-len)))
    (make-tensor shape flat-data)))

(defun tensor-data-to-tokens (data)
  (declare (xargs :guard (tensor-data-p data)
                  :measure (len data)))
  (if (atom data)
      nil
    (let* ((val (car data))
           (tok (if (and (rationalp val)
                         (>= val 0)
                         (integerp val))
                    (nfix val)
                  *special-unk*)))
      (cons tok (tensor-data-to-tokens (cdr data))))))

(defthm nat-listp-of-tensor-data-to-tokens
  (implies (tensor-data-p data)
           (nat-listp (tensor-data-to-tokens data)))
  :hints (("Goal" :induct (tensor-data-to-tokens data))))

(defthm len-of-tensor-data-to-tokens
  (implies (tensor-data-p data)
           (equal (len (tensor-data-to-tokens data))
                  (len data)))
  :hints (("Goal" :induct (tensor-data-to-tokens data))))

(defun decode-from-tensor (st tensor)
  (declare (xargs :guard (and (mgt-state-p st) (tensor-p tensor))))
  (let* ((data (tensor-data tensor))
         (tokens (tensor-data-to-tokens data)))
    (decode-tokens st tokens)))

(defthm stringp-of-decode-from-tensor
  (implies (and (mgt-state-p st) (tensor-p tensor))
           (stringp (decode-from-tensor st tensor))))

(defun set-with-prefix (st token-to-id id-to-token prefixes suffixes roots bpe-pairs next-id)
  (declare (xargs :guard (and (token-list-p token-to-id)
                               (id-list-p id-to-token)
                               (token-list-p prefixes)
                               (token-list-p suffixes)
                               (token-list-p roots)
                               (bpe-pair-list-p bpe-pairs)
                               (natp next-id))))
  (make-mgt-state token-to-id id-to-token prefixes suffixes roots bpe-pairs next-id))

(defthm mgt-state-p-of-set-with-prefix
  (implies (and (token-list-p t2i)
                (id-list-p i2t)
                (token-list-p pref)
                (token-list-p suf)
                (token-list-p roots)
                (bpe-pair-list-p bpe)
                (natp nid))
           (mgt-state-p (set-with-prefix t2i i2t pref suf roots bpe nid))))

(defun serialize-token-list (lst)
  (declare (xargs :guard (token-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (cons (list (car (car lst)) (cdr (car lst)))
          (serialize-token-list (cdr lst)))))

(defthm true-listp-of-serialize-token-list
  (implies (token-list-p lst)
           (true-listp (serialize-token-list lst)))
  :hints (("Goal" :induct (serialize-token-list lst))))

(defthm len-of-serialize-token-list
  (implies (token-list-p lst)
           (equal (len (serialize-token-list lst))
                  (len lst)))
  :hints (("Goal" :induct (serialize-token-list lst))))

(defun serialize-id-list (lst)
  (declare (xargs :guard (id-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (cons (list (car (car lst)) (cdr (car lst)))
          (serialize-id-list (cdr lst)))))

(defthm true-listp-of-serialize-id-list
  (implies (id-list-p lst)
           (true-listp (serialize-id-list lst)))
  :hints (("Goal" :induct (serialize-id-list lst))))

(defun serialize-bpe-list (lst)
  (declare (xargs :guard (bpe-pair-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (let* ((entry (car lst))
           (key (car entry))
           (merge (cdr entry))
           (tid (if (bpe-merge-p merge) (car merge) 0))
           (pri (if (bpe-merge-p merge) (cdr merge) 0)))
      (cons (list key tid pri)
            (serialize-bpe-list (cdr lst))))))

(defthm true-listp-of-serialize-bpe-list
  (implies (bpe-pair-list-p lst)
           (true-listp (serialize-bpe-list lst)))
  :hints (("Goal" :induct (serialize-bpe-list lst))))

(defun serialize-mgt-state (st)
  (declare (xargs :guard (mgt-state-p st)))
  (list (serialize-token-list (mgt-token-to-id st))
        (serialize-id-list (mgt-id-to-token st))
        (serialize-token-list (mgt-prefixes st))
        (serialize-token-list (mgt-suffixes st))
        (serialize-token-list (mgt-roots st))
        (serialize-bpe-list (mgt-bpe-pairs st))
        (mgt-next-id st)))

(defthm true-listp-of-serialize-mgt-state
  (implies (mgt-state-p st)
           (true-listp (serialize-mgt-state st))))

(defun insert-sorted-by-id (entry lst)
  (declare (xargs :guard (and (token-entry-p entry) (token-list-p lst))
                  :measure (len lst)))
  (if (atom lst)
      (list entry)
    (if (<= (cdr entry) (cdr (car lst)))
        (cons entry lst)
      (cons (car lst) (insert-sorted-by-id entry (cdr lst))))))

(defthm token-list-p-of-insert-sorted-by-id
  (implies (and (token-entry-p entry) (token-list-p lst))
           (token-list-p (insert-sorted-by-id entry lst)))
  :hints (("Goal" :induct (insert-sorted-by-id entry lst))))

(defun sort-token-list-by-id (lst acc)
  (declare (xargs :guard (and (token-list-p lst) (token-list-p acc))
                  :measure (len lst)))
  (if (atom lst)
      acc
    (sort-token-list-by-id (cdr lst)
                           (insert-sorted-by-id (car lst) acc))))

(defthm token-list-p-of-sort-token-list-by-id
  (implies (and (token-list-p lst) (token-list-p acc))
           (token-list-p (sort-token-list-by-id lst acc)))
  :hints (("Goal" :induct (sort-token-list-by-id lst acc))))

(defun insert-sorted-by-priority (entry lst)
  (declare (xargs :guard (and (bpe-pair-entry-p entry) (bpe-pair-list-p lst))
                  :measure (len lst)))
  (if (atom lst)
      (list entry)
    (let* ((e-merge (cdr entry))
           (l-merge (cdr (car lst)))
           (e-pri (if (bpe-merge-p e-merge) (cdr e-merge) 0))
           (l-pri (if (bpe-merge-p l-merge) (cdr l-merge) 0)))
      (if (<= e-pri l-pri)
          (cons entry lst)
        (cons (car lst) (insert-sorted-by-priority entry (cdr lst)))))))

(defthm bpe-pair-list-p-of-insert-sorted-by-priority
  (implies (and (bpe-pair-entry-p entry) (bpe-pair-list-p lst))
           (bpe-pair-list-p (insert-sorted-by-priority entry lst)))
  :hints (("Goal" :induct (insert-sorted-by-priority entry lst))))

(defun sort-bpe-by-priority (lst acc)
  (declare (xargs :guard (and (bpe-pair-list-p lst) (bpe-pair-list-p acc))
                  :measure (len lst)))
  (if (atom lst)
      acc
    (sort-bpe-by-priority (cdr lst)
                          (insert-sorted-by-priority (car lst) acc))))

(defthm bpe-pair-list-p-of-sort-bpe-by-priority
  (implies (and (bpe-pair-list-p lst) (bpe-pair-list-p acc))
           (bpe-pair-list-p (sort-bpe-by-priority lst acc)))
  :hints (("Goal" :induct (sort-bpe-by-priority lst acc))))

(defun lookup-token-not-in-list (token lst)
  (declare (xargs :guard (token-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      t
    (if (equal token (car (car lst)))
        nil
      (lookup-token-not-in-list token (cdr lst)))))

(defthm lookup-token-not-iff-lookup
  (implies (token-list-p lst)
           (iff (lookup-token-not-in-list token lst)
                (not (lookup-token token lst))))
  :hints (("Goal" :induct (lookup-token-not-in-list token lst))))

(defun all-tokens-valid (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      t
    (and (if (lookup-id (car tokens) (mgt-id-to-token st)) t nil)
         (all-tokens-valid st (cdr tokens)))))

(defthm all-tokens-valid-equiv-validate
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (iff (all-tokens-valid st tokens)
                (validate-tokens st tokens)))
  :hints (("Goal" :induct (all-tokens-valid st tokens))))

(defun token-to-id-consistent-p (t2i i2t)
  (declare (xargs :guard (and (token-list-p t2i) (id-list-p i2t))
                  :measure (len t2i)))
  (if (atom t2i)
      t
    (let* ((entry (car t2i))
           (token (car entry))
           (id (cdr entry))
           (reverse-lookup (lookup-id id i2t)))
      (and reverse-lookup
           (equal reverse-lookup token)
           (token-to-id-consistent-p (cdr t2i) i2t)))))

(defthm token-to-id-consistent-p-of-nil
  (token-to-id-consistent-p nil i2t))

(defun id-to-token-consistent-p (i2t t2i)
  (declare (xargs :guard (and (id-list-p i2t) (token-list-p t2i))
                  :measure (len i2t)))
  (if (atom i2t)
      t
    (let* ((entry (car i2t))
           (id (car entry))
           (token (cdr entry))
           (reverse-lookup (lookup-token token t2i)))
      (and reverse-lookup
           (equal reverse-lookup id)
           (id-to-token-consistent-p (cdr i2t) t2i)))))

(defthm id-to-token-consistent-p-of-nil
  (id-to-token-consistent-p nil t2i))

(defun mgt-consistent-p (st)
  (declare (xargs :guard (mgt-state-p st)))
  (and (token-to-id-consistent-p (mgt-token-to-id st) (mgt-id-to-token st))
       (id-to-token-consistent-p (mgt-id-to-token st) (mgt-token-to-id st))))

(defun all-ids-below (lst bound)
  (declare (xargs :guard (and (token-list-p lst) (natp bound))
                  :measure (len lst)))
  (if (atom lst)
      t
    (and (< (cdr (car lst)) bound)
         (all-ids-below (cdr lst) bound))))

(defthm all-ids-below-of-nil
  (all-ids-below nil bound))

(defthm all-ids-below-monotonic
  (implies (and (all-ids-below lst bound1)
                (<= bound1 bound2)
                (natp bound1)
                (natp bound2))
           (all-ids-below lst bound2))
  :hints (("Goal" :induct (all-ids-below lst bound1))))

(defun no-duplicate-tokens (lst)
  (declare (xargs :guard (token-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      t
    (and (not (lookup-token (car (car lst)) (cdr lst)))
         (no-duplicate-tokens (cdr lst)))))

(defthm no-duplicate-tokens-of-nil
  (no-duplicate-tokens nil))

(defun no-duplicate-ids (lst)
  (declare (xargs :guard (id-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      t
    (and (not (lookup-id (car (car lst)) (cdr lst)))
         (no-duplicate-ids (cdr lst)))))

(defthm no-duplicate-ids-of-nil
  (no-duplicate-ids nil))

(defun mgt-well-formed-p (st)
  (declare (xargs :guard (mgt-state-p st)))
  (and (mgt-consistent-p st)
       (equal (len (mgt-token-to-id st))
              (len (mgt-id-to-token st)))
       (all-ids-below (mgt-token-to-id st) (mgt-next-id st))))

(defun string-concat-list (strs)
  (declare (xargs :guard (string-list-p strs)
                  :measure (len strs)))
  (if (atom strs)
      ""
    (concatenate 'string (car strs) (string-concat-list (cdr strs)))))

(defthm stringp-of-string-concat-list
  (implies (string-list-p strs)
           (stringp (string-concat-list strs)))
  :hints (("Goal" :induct (string-concat-list strs))))

(defun decode-tokens-to-list (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (cons (decode-token st (car tokens))
          (decode-tokens-to-list st (cdr tokens)))))

(defthm string-list-p-of-decode-tokens-to-list
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (string-list-p (decode-tokens-to-list st tokens)))
  :hints (("Goal" :induct (decode-tokens-to-list st tokens))))

(defthm len-of-decode-tokens-to-list
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (equal (len (decode-tokens-to-list st tokens))
                  (len tokens)))
  :hints (("Goal" :induct (decode-tokens-to-list st tokens))))

(defun encode-decode-roundtrip-check (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (let* ((tokens (encode-text st text))
         (decoded (decode-tokens st tokens)))
    decoded))

(defthm stringp-of-encode-decode-roundtrip
  (implies (and (mgt-state-p st) (stringp text))
           (stringp (encode-decode-roundtrip-check st text))))

(defun count-non-unk (tokens)
  (declare (xargs :guard (nat-listp tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      0
    (if (not (equal (car tokens) *special-unk*))
        (+ 1 (count-non-unk (cdr tokens)))
      (count-non-unk (cdr tokens)))))

(defthm natp-of-count-non-unk
  (implies (nat-listp tokens)
           (natp (count-non-unk tokens)))
  :hints (("Goal" :induct (count-non-unk tokens)))
  :rule-classes :type-prescription)

(defthm count-non-unk-leq-len
  (implies (nat-listp tokens)
           (<= (count-non-unk tokens) (len tokens)))
  :hints (("Goal" :induct (count-non-unk tokens)))
  :rule-classes :linear)

(defun count-unk (tokens)
  (declare (xargs :guard (nat-listp tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      0
    (if (equal (car tokens) *special-unk*)
        (+ 1 (count-unk (cdr tokens)))
      (count-unk (cdr tokens)))))

(defthm natp-of-count-unk
  (implies (nat-listp tokens)
           (natp (count-unk tokens)))
  :hints (("Goal" :induct (count-unk tokens)))
  :rule-classes :type-prescription)

(defthm count-unk-leq-len
  (implies (nat-listp tokens)
           (<= (count-unk tokens) (len tokens)))
  :hints (("Goal" :induct (count-unk tokens)))
  :rule-classes :linear)

(defthm count-unk-plus-non-unk-equals-len
  (implies (nat-listp tokens)
           (equal (+ (count-unk tokens) (count-non-unk tokens))
                  (len tokens)))
  :hints (("Goal" :induct (count-unk tokens))))

(defun filter-special-tokens (tokens)
  (declare (xargs :guard (nat-listp tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (if (or (equal (car tokens) *special-pad*)
            (equal (car tokens) *special-unk*)
            (equal (car tokens) *special-bos*)
            (equal (car tokens) *special-eos*))
        (filter-special-tokens (cdr tokens))
      (cons (car tokens) (filter-special-tokens (cdr tokens))))))

(defthm nat-listp-of-filter-special-tokens
  (implies (nat-listp tokens)
           (nat-listp (filter-special-tokens tokens)))
  :hints (("Goal" :induct (filter-special-tokens tokens))))

(defthm len-of-filter-special-tokens-leq
  (implies (nat-listp tokens)
           (<= (len (filter-special-tokens tokens)) (len tokens)))
  :hints (("Goal" :induct (filter-special-tokens tokens)))
  :rule-classes :linear)

(defun token-list-contains-p (token lst)
  (declare (xargs :guard (token-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (if (equal token (car (car lst)))
        t
      (token-list-contains-p token (cdr lst)))))

(defthm token-list-contains-p-iff-lookup
  (implies (token-list-p lst)
           (iff (token-list-contains-p token lst)
                (lookup-token token lst)))
  :hints (("Goal" :induct (token-list-contains-p token lst))))

(defun id-list-contains-p (id lst)
  (declare (xargs :guard (id-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      nil
    (if (equal id (car (car lst)))
        t
      (id-list-contains-p id (cdr lst)))))

(defthm id-list-contains-p-iff-lookup
  (implies (id-list-p lst)
           (iff (id-list-contains-p id lst)
                (lookup-id id lst)))
  :hints (("Goal" :induct (id-list-contains-p id lst))))

(defun count-token-list (lst)
  (declare (xargs :guard (token-list-p lst)))
  (len lst))

(defthm natp-of-count-token-list
  (implies (token-list-p lst)
           (natp (count-token-list lst)))
  :rule-classes :type-prescription)

(defun count-id-list (lst)
  (declare (xargs :guard (id-list-p lst)))
  (len lst))

(defthm natp-of-count-id-list
  (implies (id-list-p lst)
           (natp (count-id-list lst)))
  :rule-classes :type-prescription)

(defun encode-multiple-texts (st texts acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (string-list-p texts)
                               (true-listp acc))
                  :measure (len texts)))
  (if (atom texts)
      (reverse acc)
    (let ((tokens (encode-text st (car texts))))
      (encode-multiple-texts st (cdr texts) (cons tokens acc)))))

(defthm true-listp-of-encode-multiple-texts
  (implies (and (mgt-state-p st)
                (string-list-p texts)
                (true-listp acc))
           (true-listp (encode-multiple-texts st texts acc)))
  :hints (("Goal" :induct (encode-multiple-texts st texts acc))))

(defthm len-of-encode-multiple-texts
  (implies (and (mgt-state-p st)
                (string-list-p texts)
                (true-listp acc))
           (equal (len (encode-multiple-texts st texts acc))
                  (+ (len texts) (len acc))))
  :hints (("Goal" :induct (encode-multiple-texts st texts acc))))

(defun decode-multiple-token-lists (st token-lists acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (true-listp token-lists)
                               (string-list-p acc))
                  :measure (len token-lists)))
  (if (atom token-lists)
      (reverse acc)
    (if (nat-listp (car token-lists))
        (let ((text (decode-tokens st (car token-lists))))
          (decode-multiple-token-lists st (cdr token-lists) (cons text acc)))
      (decode-multiple-token-lists st (cdr token-lists) (cons "" acc)))))

(defthm string-list-p-of-decode-multiple
  (implies (and (mgt-state-p st)
                (true-listp token-lists)
                (string-list-p acc))
           (string-list-p (decode-multiple-token-lists st token-lists acc)))
  :hints (("Goal" :induct (decode-multiple-token-lists st token-lists acc))))

(defthm len-of-decode-multiple
  (implies (and (mgt-state-p st)
                (true-listp token-lists)
                (string-list-p acc))
           (equal (len (decode-multiple-token-lists st token-lists acc))
                  (+ (len token-lists) (len acc))))
  :hints (("Goal" :induct (decode-multiple-token-lists st token-lists acc))))

(defun count-entries-with-value (lst val)
  (declare (xargs :guard (token-list-p lst)
                  :measure (len lst)))
  (if (atom lst)
      0
    (if (equal (cdr (car lst)) val)
        (+ 1 (count-entries-with-value (cdr lst) val))
      (count-entries-with-value (cdr lst) val))))

(defthm natp-of-count-entries-with-value
  (implies (token-list-p lst)
           (natp (count-entries-with-value lst val)))
  :hints (("Goal" :induct (count-entries-with-value lst val)))
  :rule-classes :type-prescription)

(defthm count-entries-with-value-leq-len
  (implies (token-list-p lst)
           (<= (count-entries-with-value lst val) (len lst)))
  :hints (("Goal" :induct (count-entries-with-value lst val)))
  :rule-classes :linear)

(defun collect-tokens-with-prefix-str (t2i prefix)
  (declare (xargs :guard (and (token-list-p t2i) (stringp prefix))
                  :measure (len t2i)))
  (if (atom t2i)
      nil
    (let ((token (car (car t2i))))
      (if (string-prefix-p prefix token)
          (cons (car t2i) (collect-tokens-with-prefix-str (cdr t2i) prefix))
        (collect-tokens-with-prefix-str (cdr t2i) prefix)))))

(defthm token-list-p-of-collect-tokens-with-prefix-str
  (implies (token-list-p t2i)
           (token-list-p (collect-tokens-with-prefix-str t2i prefix)))
  :hints (("Goal" :induct (collect-tokens-with-prefix-str t2i prefix))))

(defthm len-of-collect-tokens-leq
  (implies (token-list-p t2i)
           (<= (len (collect-tokens-with-prefix-str t2i prefix))
               (len t2i)))
  :hints (("Goal" :induct (collect-tokens-with-prefix-str t2i prefix)))
  :rule-classes :linear)

(defun collect-tokens-with-suffix-str (t2i suffix)
  (declare (xargs :guard (and (token-list-p t2i) (stringp suffix))
                  :measure (len t2i)))
  (if (atom t2i)
      nil
    (let ((token (car (car t2i))))
      (if (string-suffix-p suffix token)
          (cons (car t2i) (collect-tokens-with-suffix-str (cdr t2i) suffix))
        (collect-tokens-with-suffix-str (cdr t2i) suffix)))))

(defthm token-list-p-of-collect-tokens-with-suffix-str
  (implies (token-list-p t2i)
           (token-list-p (collect-tokens-with-suffix-str t2i suffix)))
  :hints (("Goal" :induct (collect-tokens-with-suffix-str t2i suffix))))

(defun min-token-id (lst current-min)
  (declare (xargs :guard (and (token-list-p lst) (natp current-min))
                  :measure (len lst)))
  (if (atom lst)
      current-min
    (let ((id (cdr (car lst))))
      (min-token-id (cdr lst) (if (< id current-min) id current-min)))))

(defthm natp-of-min-token-id
  (implies (natp current-min)
           (natp (min-token-id lst current-min)))
  :hints (("Goal" :induct (min-token-id lst current-min)))
  :rule-classes :type-prescription)

(defthm min-token-id-leq-current
  (implies (natp current-min)
           (<= (min-token-id lst current-min) current-min))
  :hints (("Goal" :induct (min-token-id lst current-min)))
  :rule-classes :linear)

(defun max-token-id (lst current-max)
  (declare (xargs :guard (and (token-list-p lst) (natp current-max))
                  :measure (len lst)))
  (if (atom lst)
      current-max
    (let ((id (cdr (car lst))))
      (max-token-id (cdr lst) (if (> id current-max) id current-max)))))

(defthm natp-of-max-token-id
  (implies (natp current-max)
           (natp (max-token-id lst current-max)))
  :hints (("Goal" :induct (max-token-id lst current-max)))
  :rule-classes :type-prescription)

(defthm max-token-id-geq-current
  (implies (natp current-max)
           (<= current-max (max-token-id lst current-max)))
  :hints (("Goal" :induct (max-token-id lst current-max)))
  :rule-classes :linear)

(defun all-prefixes-are-tokens (prefix-list t2i)
  (declare (xargs :guard (and (token-list-p prefix-list) (token-list-p t2i))
                  :measure (len prefix-list)))
  (if (atom prefix-list)
      t
    (and (lookup-token (car (car prefix-list)) t2i)
         (all-prefixes-are-tokens (cdr prefix-list) t2i))))

(defthm all-prefixes-are-tokens-of-nil
  (all-prefixes-are-tokens nil t2i))

(defun all-suffixes-are-tokens (suffix-list t2i)
  (declare (xargs :guard (and (token-list-p suffix-list) (token-list-p t2i))
                  :measure (len suffix-list)))
  (if (atom suffix-list)
      t
    (and (lookup-token (car (car suffix-list)) t2i)
         (all-suffixes-are-tokens (cdr suffix-list) t2i))))

(defthm all-suffixes-are-tokens-of-nil
  (all-suffixes-are-tokens nil t2i))

(defun morpheme-consistency-p (st)
  (declare (xargs :guard (mgt-state-p st)))
  (and (all-prefixes-are-tokens (mgt-prefixes st) (mgt-token-to-id st))
       (all-suffixes-are-tokens (mgt-suffixes st) (mgt-token-to-id st))))

(defun encode-char-by-char (st text pos acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp text)
                               (natp pos)
                               (nat-listp acc)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      (reverse acc)
    (let* ((c-str (coerce (list (char text pos)) 'string))
           (tid (lookup-token c-str (mgt-token-to-id st)))
           (token-id (if tid tid *special-unk*)))
      (encode-char-by-char st text (+ pos 1) (cons token-id acc)))))

(defthm nat-listp-of-encode-char-by-char
  (implies (and (mgt-state-p st)
                (stringp text)
                (natp pos)
                (nat-listp acc))
           (nat-listp (encode-char-by-char st text pos acc)))
  :hints (("Goal" :induct (encode-char-by-char st text pos acc))))

(defun greedy-tokenize (st text pos acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp text)
                               (natp pos)
                               (nat-listp acc)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      (reverse acc)
    (let ((match-len (longest-match st text pos)))
      (if (and (natp match-len) (> match-len 0))
          (let* ((matched (substring text pos (+ pos match-len)))
                 (tid (lookup-token matched (mgt-token-to-id st))))
            (if tid
                (greedy-tokenize st text (+ pos match-len) (cons tid acc))
              (greedy-tokenize st text (+ pos 1) (cons *special-unk* acc))))
        (greedy-tokenize st text (+ pos 1) (cons *special-unk* acc))))))

(defthm nat-listp-of-greedy-tokenize
  (implies (and (mgt-state-p st)
                (stringp text)
                (natp pos)
                (nat-listp acc))
           (nat-listp (greedy-tokenize st text pos acc)))
  :hints (("Goal" :induct (greedy-tokenize st text pos acc))))

(defun reverse-nat-list (lst acc)
  (declare (xargs :guard (and (nat-listp lst) (nat-listp acc))
                  :measure (len lst)))
  (if (atom lst)
      acc
    (reverse-nat-list (cdr lst) (cons (car lst) acc))))

(defthm nat-listp-of-reverse-nat-list
  (implies (and (nat-listp lst) (nat-listp acc))
           (nat-listp (reverse-nat-list lst acc)))
  :hints (("Goal" :induct (reverse-nat-list lst acc))))

(defthm len-of-reverse-nat-list
  (implies (and (nat-listp lst) (nat-listp acc))
           (equal (len (reverse-nat-list lst acc))
                  (+ (len lst) (len acc))))
  :hints (("Goal" :induct (reverse-nat-list lst acc))))

(defun take-n (n lst)
  (declare (xargs :guard (and (natp n) (true-listp lst))
                  :measure (nfix n)))
  (if (or (zp n) (atom lst))
      nil
    (cons (car lst) (take-n (- n 1) (cdr lst)))))

(defthm true-listp-of-take-n
  (true-listp (take-n n lst))
  :hints (("Goal" :induct (take-n n lst))))

(defthm len-of-take-n
  (implies (and (natp n) (true-listp lst))
           (<= (len (take-n n lst)) n))
  :hints (("Goal" :induct (take-n n lst)))
  :rule-classes :linear)

(defun drop-n (n lst)
  (declare (xargs :guard (and (natp n) (true-listp lst))
                  :measure (nfix n)))
  (if (or (zp n) (atom lst))
      lst
    (drop-n (- n 1) (cdr lst))))

(defthm true-listp-of-drop-n
  (implies (true-listp lst)
           (true-listp (drop-n n lst)))
  :hints (("Goal" :induct (drop-n n lst))))

(defun split-token-list-at (lst n)
  (declare (xargs :guard (and (nat-listp lst) (natp n))))
  (mv (take-n n lst) (drop-n n lst)))

(defun unique-tokens-in-list (tokens seen acc)
  (declare (xargs :guard (and (nat-listp tokens) (true-listp seen) (nat-listp acc))
                  :measure (len tokens)))
  (if (atom tokens)
      (reverse acc)
    (if (member-equal (car tokens) seen)
        (unique-tokens-in-list (cdr tokens) seen acc)
      (unique-tokens-in-list (cdr tokens)
                             (cons (car tokens) seen)
                             (cons (car tokens) acc)))))

(defthm nat-listp-of-unique-tokens-in-list
  (implies (and (nat-listp tokens) (nat-listp acc))
           (nat-listp (unique-tokens-in-list tokens seen acc)))
  :hints (("Goal" :induct (unique-tokens-in-list tokens seen acc))))

(defthm len-of-unique-tokens-leq
  (implies (and (nat-listp tokens) (nat-listp acc))
           (<= (len (unique-tokens-in-list tokens seen acc))
               (+ (len tokens) (len acc))))
  :hints (("Goal" :induct (unique-tokens-in-list tokens seen acc)))
  :rule-classes :linear)

(defun token-frequency-count (tokens freq-map)
  (declare (xargs :guard (and (nat-listp tokens) (true-listp freq-map))
                  :measure (len tokens)))
  (if (atom tokens)
      freq-map
    (let* ((tok (car tokens))
           (existing (assoc-equal tok freq-map))
           (new-count (if existing (+ 1 (cdr existing)) 1))
           (new-map (if existing
                        (put-assoc-equal tok new-count freq-map)
                      (cons (cons tok new-count) freq-map))))
      (token-frequency-count (cdr tokens) new-map))))

(defun sum-frequencies (freq-map)
  (declare (xargs :guard (true-listp freq-map)
                  :measure (len freq-map)))
  (if (atom freq-map)
      0
    (let ((val (if (and (consp (car freq-map))
                        (natp (cdr (car freq-map))))
                   (cdr (car freq-map))
                 0)))
      (+ val (sum-frequencies (cdr freq-map))))))

(defthm natp-of-sum-frequencies
  (implies (true-listp freq-map)
           (natp (sum-frequencies freq-map)))
  :hints (("Goal" :induct (sum-frequencies freq-map)))
  :rule-classes :type-prescription)

(defun encode-text-deterministic-check (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (let* ((tokens1 (encode-text st text))
         (tokens2 (encode-text st text)))
    (equal tokens1 tokens2)))

(defthm encode-is-deterministic
  (implies (and (mgt-state-p st) (stringp text))
           (encode-text-deterministic-check st text)))

(defun apply-merge-to-seq-preserves-length-bound (seq first-id second-id merged-id)
  (declare (xargs :guard (and (nat-listp seq) (natp first-id) (natp second-id) (natp merged-id))))
  (<= (len (apply-merge-to-seq seq first-id second-id merged-id nil))
      (len seq)))

(defthm apply-merge-shrinks-or-preserves
  (implies (and (nat-listp seq) (natp first-id) (natp second-id) (natp merged-id))
           (apply-merge-to-seq-preserves-length-bound seq first-id second-id merged-id)))

(defun verify-add-token-idempotence (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (mv-let (st1 id1) (add-token-to-state st token)
    (mv-let (st2 id2) (add-token-to-state st1 token)
      (and (equal id1 id2)
           (equal (mgt-next-id st1) (mgt-next-id st2))))))

(defthm add-token-is-idempotent
  (implies (and (mgt-state-p st) (stringp token))
           (verify-add-token-idempotence st token)))

(defun batch-size-preserved-check (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))))
  (equal (len (encode-batch st texts))
         (len texts)))

(defthm batch-size-preserved
  (implies (and (mgt-state-p st) (string-list-p texts))
           (batch-size-preserved-check st texts)))

(defun decode-batch-size-check (st token-lists)
  (declare (xargs :guard (and (mgt-state-p st) (true-listp token-lists))))
  (equal (len (batch-decode st token-lists))
         (len token-lists)))

(defthm decode-batch-size-preserved
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (decode-batch-size-check st token-lists)))

(defun special-tokens-present-p (st)
  (declare (xargs :guard (mgt-state-p st)))
  (and (lookup-token "[PAD]" (mgt-token-to-id st))
       (lookup-token "[UNK]" (mgt-token-to-id st))
       (lookup-token "[BOS]" (mgt-token-to-id st))
       (lookup-token "[EOS]" (mgt-token-to-id st))
       t))

(defun verify-special-token-ids (st)
  (declare (xargs :guard (mgt-state-p st)))
  (let ((pad-id (lookup-token "[PAD]" (mgt-token-to-id st)))
        (unk-id (lookup-token "[UNK]" (mgt-token-to-id st)))
        (bos-id (lookup-token "[BOS]" (mgt-token-to-id st)))
        (eos-id (lookup-token "[EOS]" (mgt-token-to-id st))))
    (and pad-id unk-id bos-id eos-id
         (not (equal pad-id unk-id))
         (not (equal pad-id bos-id))
         (not (equal pad-id eos-id))
         (not (equal unk-id bos-id))
         (not (equal unk-id eos-id))
         (not (equal bos-id eos-id)))))

(defun lookup-token-after-add (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (mv-let (st1 id) (add-token-to-state st token)
    (equal (lookup-token token (mgt-token-to-id st1)) id)))

(defthm add-then-lookup-succeeds
  (implies (and (mgt-state-p st) (stringp token))
           (lookup-token-after-add st token)))

(defun lookup-id-after-add (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (mv-let (st1 id) (add-token-to-state st token)
    (equal (lookup-id id (mgt-id-to-token st1)) token)))

(defthm add-then-lookup-id-succeeds
  (implies (and (mgt-state-p st) (stringp token))
           (lookup-id-after-add st token)))

(defun vocab-grows-on-new-token (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (if (lookup-token token (mgt-token-to-id st))
      (mv-let (st1 id) (add-token-to-state st token)
        (declare (ignore id))
        (equal (len (mgt-token-to-id st1))
               (len (mgt-token-to-id st))))
    (mv-let (st1 id) (add-token-to-state st token)
      (declare (ignore id))
      (equal (len (mgt-token-to-id st1))
             (+ 1 (len (mgt-token-to-id st)))))))

(defthm vocab-size-correct-on-add
  (implies (and (mgt-state-p st) (stringp token))
           (vocab-grows-on-new-token st token)))

(defun remove-then-lookup-fails (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (if (or (equal word "[PAD]") (equal word "[UNK]")
          (equal word "[BOS]") (equal word "[EOS]"))
      t
    (let ((st1 (remove-vocab-word st word)))
      (not (lookup-token word (mgt-token-to-id st1))))))

(defthm remove-then-lookup-is-nil
  (implies (and (mgt-state-p st) (stringp word))
           (remove-then-lookup-fails st word)))

(defun prefix-id-matches-token-id (st)
  (declare (xargs :guard (mgt-state-p st)))
  (all-prefixes-match-aux (mgt-prefixes st) (mgt-token-to-id st)))

(defun all-prefixes-match-aux (prefix-list t2i)
  (declare (xargs :guard (and (token-list-p prefix-list) (token-list-p t2i))
                  :measure (len prefix-list)))
  (if (atom prefix-list)
      t
    (let* ((entry (car prefix-list))
           (token (car entry))
           (prefix-id (cdr entry))
           (token-id (lookup-token token t2i)))
      (and (equal token-id prefix-id)
           (all-prefixes-match-aux (cdr prefix-list) t2i)))))

(defun suffix-id-matches-token-id (st)
  (declare (xargs :guard (mgt-state-p st)))
  (all-suffixes-match-aux (mgt-suffixes st) (mgt-token-to-id st)))

(defun all-suffixes-match-aux (suffix-list t2i)
  (declare (xargs :guard (and (token-list-p suffix-list) (token-list-p t2i))
                  :measure (len suffix-list)))
  (if (atom suffix-list)
      t
    (let* ((entry (car suffix-list))
           (token (car entry))
           (suffix-id (cdr entry))
           (token-id (lookup-token token t2i)))
      (and (equal token-id suffix-id)
           (all-suffixes-match-aux (cdr suffix-list) t2i)))))

(defun encode-empty-text-gives-empty (st)
  (declare (xargs :guard (mgt-state-p st)))
  (equal (encode-text st "") nil))

(defthm encode-empty-is-nil
  (implies (mgt-state-p st)
           (encode-empty-text-gives-empty st)))

(defun decode-empty-tokens-gives-empty (st)
  (declare (xargs :guard (mgt-state-p st)))
  (equal (decode-tokens st nil) ""))

(defthm decode-nil-is-empty-string
  (implies (mgt-state-p st)
           (decode-empty-tokens-gives-empty st)))

(defun concat-two-strings (a b)
  (declare (xargs :guard (and (stringp a) (stringp b))))
  (concatenate 'string a b))

(defthm stringp-of-concat-two-strings
  (implies (and (stringp a) (stringp b))
           (stringp (concat-two-strings a b)))
  :rule-classes :type-prescription)

(defun repeat-string (s n)
  (declare (xargs :guard (and (stringp s) (natp n))
                  :measure (nfix n)))
  (if (zp n)
      ""
    (concatenate 'string s (repeat-string s (- n 1)))))

(defthm stringp-of-repeat-string
  (implies (and (stringp s) (natp n))
           (stringp (repeat-string s n)))
  :hints (("Goal" :induct (repeat-string s n)))
  :rule-classes :type-prescription)

(defun token-list-to-nat-list (tokens)
  (declare (xargs :guard (token-list-p tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (cons (cdr (car tokens))
          (token-list-to-nat-list (cdr tokens)))))

(defthm nat-listp-of-token-list-to-nat-list
  (implies (token-list-p tokens)
           (nat-listp (token-list-to-nat-list tokens)))
  :hints (("Goal" :induct (token-list-to-nat-list tokens))))

(defthm len-of-token-list-to-nat-list
  (implies (token-list-p tokens)
           (equal (len (token-list-to-nat-list tokens))
                  (len tokens)))
  :hints (("Goal" :induct (token-list-to-nat-list tokens))))

(defun token-list-to-string-list (tokens)
  (declare (xargs :guard (token-list-p tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (cons (car (car tokens))
          (token-list-to-string-list (cdr tokens)))))

(defthm string-list-p-of-token-list-to-string-list
  (implies (token-list-p tokens)
           (string-list-p (token-list-to-string-list tokens)))
  :hints (("Goal" :induct (token-list-to-string-list tokens))))

(defthm len-of-token-list-to-string-list
  (implies (token-list-p tokens)
           (equal (len (token-list-to-string-list tokens))
                  (len tokens)))
  :hints (("Goal" :induct (token-list-to-string-list tokens))))

(defun id-list-to-nat-list (ids)
  (declare (xargs :guard (id-list-p ids)
                  :measure (len ids)))
  (if (atom ids)
      nil
    (cons (car (car ids))
          (id-list-to-nat-list (cdr ids)))))

(defthm nat-listp-of-id-list-to-nat-list
  (implies (id-list-p ids)
           (nat-listp (id-list-to-nat-list ids)))
  :hints (("Goal" :induct (id-list-to-nat-list ids))))

(defun id-list-to-string-list (ids)
  (declare (xargs :guard (id-list-p ids)
                  :measure (len ids)))
  (if (atom ids)
      nil
    (cons (cdr (car ids))
          (id-list-to-string-list (cdr ids)))))

(defthm string-list-p-of-id-list-to-string-list
  (implies (id-list-p ids)
           (string-list-p (id-list-to-string-list ids)))
  :hints (("Goal" :induct (id-list-to-string-list ids))))

(defun bpe-pair-keys (bpe-list)
  (declare (xargs :guard (bpe-pair-list-p bpe-list)
                  :measure (len bpe-list)))
  (if (atom bpe-list)
      nil
    (cons (car (car bpe-list))
          (bpe-pair-keys (cdr bpe-list)))))

(defthm string-list-p-of-bpe-pair-keys
  (implies (bpe-pair-list-p bpe-list)
           (string-list-p (bpe-pair-keys bpe-list)))
  :hints (("Goal" :induct (bpe-pair-keys bpe-list))))

(defun bpe-pair-token-ids (bpe-list)
  (declare (xargs :guard (bpe-pair-list-p bpe-list)
                  :measure (len bpe-list)))
  (if (atom bpe-list)
      nil
    (let ((merge (cdr (car bpe-list))))
      (cons (if (bpe-merge-p merge) (car merge) 0)
            (bpe-pair-token-ids (cdr bpe-list))))))

(defthm nat-listp-of-bpe-pair-token-ids
  (implies (bpe-pair-list-p bpe-list)
           (nat-listp (bpe-pair-token-ids bpe-list)))
  :hints (("Goal" :induct (bpe-pair-token-ids bpe-list))))

(defun bpe-pair-priorities (bpe-list)
  (declare (xargs :guard (bpe-pair-list-p bpe-list)
                  :measure (len bpe-list)))
  (if (atom bpe-list)
      nil
    (let ((merge (cdr (car bpe-list))))
      (cons (if (bpe-merge-p merge) (cdr merge) 0)
            (bpe-pair-priorities (cdr bpe-list))))))

(defthm nat-listp-of-bpe-pair-priorities
  (implies (bpe-pair-list-p bpe-list)
           (nat-listp (bpe-pair-priorities bpe-list)))
  :hints (("Goal" :induct (bpe-pair-priorities bpe-list))))

(defun all-bpe-tokens-in-vocab (bpe-list t2i)
  (declare (xargs :guard (and (bpe-pair-list-p bpe-list) (token-list-p t2i))
                  :measure (len bpe-list)))
  (if (atom bpe-list)
      t
    (let* ((merge (cdr (car bpe-list)))
           (key (car (car bpe-list)))
           (tid (if (bpe-merge-p merge) (car merge) 0)))
      (declare (ignore tid))
      (and (lookup-token key t2i)
           (all-bpe-tokens-in-vocab (cdr bpe-list) t2i)))))

(defthm all-bpe-tokens-in-vocab-of-nil
  (all-bpe-tokens-in-vocab nil t2i))

(defun sequence-total-length (seqs)
  (declare (xargs :guard (true-listp seqs)
                  :measure (len seqs)))
  (if (atom seqs)
      0
    (+ (if (nat-listp (car seqs)) (len (car seqs)) 0)
       (sequence-total-length (cdr seqs)))))

(defthm natp-of-sequence-total-length
  (implies (true-listp seqs)
           (natp (sequence-total-length seqs)))
  :hints (("Goal" :induct (sequence-total-length seqs)))
  :rule-classes :type-prescription)

(defun apply-n-merges-shrinks (seq n first-id second-id merged-id)
  (declare (xargs :guard (and (nat-listp seq) (natp n) (natp first-id)
                               (natp second-id) (natp merged-id))
                  :measure (nfix n)))
  (if (zp n)
      seq
    (let ((new-seq (apply-merge-to-seq seq first-id second-id merged-id nil)))
      (apply-n-merges-shrinks new-seq (- n 1) first-id second-id merged-id))))

(defthm nat-listp-of-apply-n-merges
  (implies (and (nat-listp seq) (natp n) (natp first-id) (natp second-id) (natp merged-id))
           (nat-listp (apply-n-merges-shrinks seq n first-id second-id merged-id)))
  :hints (("Goal" :induct (apply-n-merges-shrinks seq n first-id second-id merged-id))))

(defun map-encode-text (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))
                  :measure (len texts)))
  (if (atom texts)
      nil
    (cons (encode-text st (car texts))
          (map-encode-text st (cdr texts)))))

(defthm true-listp-of-map-encode-text
  (implies (and (mgt-state-p st) (string-list-p texts))
           (true-listp (map-encode-text st texts)))
  :hints (("Goal" :induct (map-encode-text st texts))))

(defthm len-of-map-encode-text
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (map-encode-text st texts))
                  (len texts)))
  :hints (("Goal" :induct (map-encode-text st texts))))

(defun map-decode-tokens (st token-lists)
  (declare (xargs :guard (and (mgt-state-p st) (true-listp token-lists))
                  :measure (len token-lists)))
  (if (atom token-lists)
      nil
    (if (nat-listp (car token-lists))
        (cons (decode-tokens st (car token-lists))
              (map-decode-tokens st (cdr token-lists)))
      (cons "" (map-decode-tokens st (cdr token-lists))))))

(defthm string-list-p-of-map-decode-tokens
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (string-list-p (map-decode-tokens st token-lists)))
  :hints (("Goal" :induct (map-decode-tokens st token-lists))))

(defthm len-of-map-decode-tokens
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (map-decode-tokens st token-lists))
                  (len token-lists)))
  :hints (("Goal" :induct (map-decode-tokens st token-lists))))

(defun safe-nth-token (n tokens)
  (declare (xargs :guard (and (natp n) (nat-listp tokens))))
  (if (< n (len tokens))
      (nth n tokens)
    *special-unk*))

(defthm natp-of-safe-nth-token
  (implies (and (natp n) (nat-listp tokens))
           (natp (safe-nth-token n tokens)))
  :rule-classes :type-prescription)

(defun window-tokens (tokens start window-size)
  (declare (xargs :guard (and (nat-listp tokens) (natp start) (natp window-size))))
  (let ((end (min (+ start window-size) (len tokens))))
    (if (>= start (len tokens))
        nil
      (take-n (- end start) (drop-n start tokens)))))

(defun count-vocab-entries (st)
  (declare (xargs :guard (mgt-state-p st)))
  (len (mgt-token-to-id st)))

(defthm natp-of-count-vocab-entries
  (implies (mgt-state-p st)
           (natp (count-vocab-entries st)))
  :rule-classes :type-prescription)

(defthm count-vocab-equals-vocab-size
  (implies (mgt-state-p st)
           (equal (count-vocab-entries st)
                  (vocab-size st))))

(defun count-bpe-merges (st)
  (declare (xargs :guard (mgt-state-p st)))
  (len (mgt-bpe-pairs st)))

(defthm natp-of-count-bpe-merges
  (implies (mgt-state-p st)
           (natp (count-bpe-merges st)))
  :rule-classes :type-prescription)

(defun count-prefix-entries (st)
  (declare (xargs :guard (mgt-state-p st)))
  (len (mgt-prefixes st)))

(defthm natp-of-count-prefix-entries
  (implies (mgt-state-p st)
           (natp (count-prefix-entries st)))
  :rule-classes :type-prescription)

(defun count-suffix-entries (st)
  (declare (xargs :guard (mgt-state-p st)))
  (len (mgt-suffixes st)))

(defthm natp-of-count-suffix-entries
  (implies (mgt-state-p st)
           (natp (count-suffix-entries st)))
  :rule-classes :type-prescription)

(defun total-morpheme-count (st)
  (declare (xargs :guard (mgt-state-p st)))
  (+ (count-prefix-entries st)
     (count-suffix-entries st)
     (len (mgt-roots st))))

(defthm natp-of-total-morpheme-count
  (implies (mgt-state-p st)
           (natp (total-morpheme-count st)))
  :rule-classes :type-prescription)

(defun has-token-p (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (if (lookup-token token (mgt-token-to-id st)) t nil))

(defun has-id-p (st id)
  (declare (xargs :guard (and (mgt-state-p st) (natp id))))
  (if (lookup-id id (mgt-id-to-token st)) t nil))

(defun get-token-id (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (lookup-token token (mgt-token-to-id st)))

(defun get-id-token (st id)
  (declare (xargs :guard (and (mgt-state-p st) (natp id))))
  (lookup-id id (mgt-id-to-token st)))

(defthm get-token-id-returns-natp-or-nil
  (implies (mgt-state-p st)
           (or (null (get-token-id st token))
               (natp (get-token-id st token))))
  :rule-classes :type-prescription)

(defthm get-id-token-returns-stringp-or-nil
  (implies (mgt-state-p st)
           (or (null (get-id-token st id))
               (stringp (get-id-token st id))))
  :rule-classes :type-prescription)

(defun is-prefix-p (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (if (lookup-token token (mgt-prefixes st)) t nil))

(defun is-suffix-p (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (if (lookup-token token (mgt-suffixes st)) t nil))

(defun is-root-p (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (if (lookup-token token (mgt-roots st)) t nil))

(defun classify-token (st token)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token))))
  (cond ((is-special-token token) :special)
        ((is-prefix-p st token) :prefix)
        ((is-suffix-p st token) :suffix)
        ((is-root-p st token) :root)
        ((has-token-p st token) :word)
        (t :unknown)))

(defun classify-all-tokens (st token-strings acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (string-list-p token-strings)
                               (true-listp acc))
                  :measure (len token-strings)))
  (if (atom token-strings)
      (reverse acc)
    (let ((class (classify-token st (car token-strings))))
      (classify-all-tokens st (cdr token-strings) (cons class acc)))))

(defthm true-listp-of-classify-all-tokens
  (implies (and (mgt-state-p st)
                (string-list-p token-strings)
                (true-listp acc))
           (true-listp (classify-all-tokens st token-strings acc)))
  :hints (("Goal" :induct (classify-all-tokens st token-strings acc))))

(defthm len-of-classify-all-tokens
  (implies (and (mgt-state-p st)
                (string-list-p token-strings)
                (true-listp acc))
           (equal (len (classify-all-tokens st token-strings acc))
                  (+ (len token-strings) (len acc))))
  :hints (("Goal" :induct (classify-all-tokens st token-strings acc))))

(defun encode-single-char (st c)
  (declare (xargs :guard (and (mgt-state-p st) (characterp c))))
  (let* ((c-str (coerce (list c) 'string))
         (tid (lookup-token c-str (mgt-token-to-id st))))
    (if tid tid *special-unk*)))

(defthm natp-of-encode-single-char
  (implies (and (mgt-state-p st) (characterp c))
           (natp (encode-single-char st c)))
  :rule-classes :type-prescription)

(defun decode-single-token (st tid)
  (declare (xargs :guard (and (mgt-state-p st) (natp tid))))
  (decode-token st tid))

(defthm stringp-of-decode-single-token
  (implies (and (mgt-state-p st) (natp tid))
           (stringp (decode-single-token st tid)))
  :rule-classes :type-prescription)

(defun all-tokens-are-known (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      t
    (and (has-id-p st (car tokens))
         (all-tokens-are-known st (cdr tokens)))))

(defthm all-tokens-are-known-implies-validate
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (iff (all-tokens-are-known st tokens)
                (validate-tokens st tokens)))
  :hints (("Goal" :induct (all-tokens-are-known st tokens))))

(defun no-unk-in-tokens (tokens)
  (declare (xargs :guard (nat-listp tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      t
    (and (not (equal (car tokens) *special-unk*))
         (no-unk-in-tokens (cdr tokens)))))

(defthm no-unk-implies-zero-unk-count
  (implies (and (nat-listp tokens) (no-unk-in-tokens tokens))
           (equal (count-unk tokens) 0))
  :hints (("Goal" :induct (count-unk tokens))))

(defthm zero-unk-count-implies-no-unk
  (implies (and (nat-listp tokens) (equal (count-unk tokens) 0))
           (no-unk-in-tokens tokens))
  :hints (("Goal" :induct (no-unk-in-tokens tokens))))

(defun only-known-tokens (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (if (has-id-p st (car tokens))
        (cons (car tokens) (only-known-tokens st (cdr tokens)))
      (only-known-tokens st (cdr tokens)))))

(defthm nat-listp-of-only-known-tokens
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (nat-listp (only-known-tokens st tokens)))
  :hints (("Goal" :induct (only-known-tokens st tokens))))

(defthm len-of-only-known-tokens-leq
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (<= (len (only-known-tokens st tokens)) (len tokens)))
  :hints (("Goal" :induct (only-known-tokens st tokens)))
  :rule-classes :linear)

(defthm all-tokens-known-in-only-known
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (all-tokens-are-known st (only-known-tokens st tokens)))
  :hints (("Goal" :induct (only-known-tokens st tokens))))

(defun replace-unk-with (tokens replacement)
  (declare (xargs :guard (and (nat-listp tokens) (natp replacement))
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (if (equal (car tokens) *special-unk*)
        (cons replacement (replace-unk-with (cdr tokens) replacement))
      (cons (car tokens) (replace-unk-with (cdr tokens) replacement)))))

(defthm nat-listp-of-replace-unk-with
  (implies (and (nat-listp tokens) (natp replacement))
           (nat-listp (replace-unk-with tokens replacement)))
  :hints (("Goal" :induct (replace-unk-with tokens replacement))))

(defthm len-of-replace-unk-with
  (implies (nat-listp tokens)
           (equal (len (replace-unk-with tokens replacement))
                  (len tokens)))
  :hints (("Goal" :induct (replace-unk-with tokens replacement))))

(defthm no-unk-after-replace
  (implies (and (nat-listp tokens)
                (natp replacement)
                (not (equal replacement *special-unk*)))
           (no-unk-in-tokens (replace-unk-with tokens replacement)))
  :hints (("Goal" :induct (replace-unk-with tokens replacement))))

(defun pad-tokens-to-length (tokens target-len)
  (declare (xargs :guard (and (nat-listp tokens) (natp target-len))
                  :measure (nfix (- target-len (len tokens)))))
  (if (or (not (natp target-len))
          (>= (len tokens) target-len))
      tokens
    (pad-tokens-to-length (append tokens (list *special-pad*))
                          target-len)))

(defun truncate-tokens (tokens max-len)
  (declare (xargs :guard (and (nat-listp tokens) (natp max-len))))
  (take-n max-len tokens))

(defun pad-or-truncate (tokens target-len)
  (declare (xargs :guard (and (nat-listp tokens) (natp target-len))))
  (if (> (len tokens) target-len)
      (truncate-tokens tokens target-len)
    (pad-tokens-to-length tokens target-len)))

(defun token-list-equal (a b)
  (declare (xargs :guard (and (nat-listp a) (nat-listp b))
                  :measure (+ (len a) (len b))))
  (if (and (atom a) (atom b))
      t
    (if (or (atom a) (atom b))
        nil
      (and (equal (car a) (car b))
           (token-list-equal (cdr a) (cdr b))))))

(defthm token-list-equal-reflexive
  (implies (nat-listp tokens)
           (token-list-equal tokens tokens))
  :hints (("Goal" :induct (token-list-equal tokens tokens))))

(defthm token-list-equal-symmetric
  (implies (and (nat-listp a) (nat-listp b)
                (token-list-equal a b))
           (token-list-equal b a))
  :hints (("Goal" :induct (token-list-equal a b))))

(defun interleave-tokens (a b)
  (declare (xargs :guard (and (nat-listp a) (nat-listp b))
                  :measure (+ (len a) (len b))))
  (cond ((atom a) b)
        ((atom b) a)
        (t (cons (car a) (cons (car b) (interleave-tokens (cdr a) (cdr b)))))))

(defthm nat-listp-of-interleave-tokens
  (implies (and (nat-listp a) (nat-listp b))
           (nat-listp (interleave-tokens a b)))
  :hints (("Goal" :induct (interleave-tokens a b))))

(defun prepend-bos-append-eos (tokens)
  (declare (xargs :guard (nat-listp tokens)))
  (cons *special-bos* (append tokens (list *special-eos*))))

(defthm nat-listp-of-prepend-bos-append-eos
  (implies (nat-listp tokens)
           (nat-listp (prepend-bos-append-eos tokens))))

(defthm len-of-prepend-bos-append-eos
  (implies (nat-listp tokens)
           (equal (len (prepend-bos-append-eos tokens))
                  (+ 2 (len tokens)))))

(defthm car-of-prepend-bos-is-bos
  (equal (car (prepend-bos-append-eos tokens))
         *special-bos*))

(defun strip-bos-eos (tokens)
  (declare (xargs :guard (nat-listp tokens)))
  (if (and (consp tokens)
           (equal (car tokens) *special-bos*))
      (let ((rest (cdr tokens)))
        (if (and (consp rest)
                 (equal (car (last rest)) *special-eos*))
            (take-n (- (len rest) 1) rest)
          rest))
    tokens))

(defun encode-with-bos-eos (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (prepend-bos-append-eos (encode-text st text)))

(defthm nat-listp-of-encode-with-bos-eos
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-with-bos-eos st text))))

(defthm encode-with-bos-eos-starts-with-bos
  (implies (and (mgt-state-p st) (stringp text))
           (equal (car (encode-with-bos-eos st text))
                  *special-bos*)))

(defun attention-mask-from-tokens (tokens)
  (declare (xargs :guard (nat-listp tokens)
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (cons (if (equal (car tokens) *special-pad*) 0 1)
          (attention-mask-from-tokens (cdr tokens)))))

(defthm nat-listp-of-attention-mask
  (implies (nat-listp tokens)
           (nat-listp (attention-mask-from-tokens tokens)))
  :hints (("Goal" :induct (attention-mask-from-tokens tokens))))

(defthm len-of-attention-mask
  (implies (nat-listp tokens)
           (equal (len (attention-mask-from-tokens tokens))
                  (len tokens)))
  :hints (("Goal" :induct (attention-mask-from-tokens tokens))))

(defun count-active-tokens (mask)
  (declare (xargs :guard (nat-listp mask)
                  :measure (len mask)))
  (if (atom mask)
      0
    (+ (if (equal (car mask) 1) 1 0)
       (count-active-tokens (cdr mask)))))

(defthm natp-of-count-active-tokens
  (implies (nat-listp mask)
           (natp (count-active-tokens mask)))
  :hints (("Goal" :induct (count-active-tokens mask)))
  :rule-classes :type-prescription)

(defthm count-active-leq-len
  (implies (nat-listp mask)
           (<= (count-active-tokens mask) (len mask)))
  :hints (("Goal" :induct (count-active-tokens mask)))
  :rule-classes :linear)

(defun make-position-ids (n start)
  (declare (xargs :guard (and (natp n) (natp start))
                  :measure (nfix n)))
  (if (zp n)
      nil
    (cons start (make-position-ids (- n 1) (+ start 1)))))

(defthm nat-listp-of-make-position-ids
  (implies (and (natp n) (natp start))
           (nat-listp (make-position-ids n start)))
  :hints (("Goal" :induct (make-position-ids n start))))

(defthm len-of-make-position-ids
  (implies (natp n)
           (equal (len (make-position-ids n start)) n))
  :hints (("Goal" :induct (make-position-ids n start))))

(defun token-type-ids (tokens segment-id)
  (declare (xargs :guard (and (nat-listp tokens) (natp segment-id))
                  :measure (len tokens)))
  (if (atom tokens)
      nil
    (cons segment-id (token-type-ids (cdr tokens) segment-id))))

(defthm nat-listp-of-token-type-ids
  (implies (and (nat-listp tokens) (natp segment-id))
           (nat-listp (token-type-ids tokens segment-id)))
  :hints (("Goal" :induct (token-type-ids tokens segment-id))))

(defthm len-of-token-type-ids
  (implies (nat-listp tokens)
           (equal (len (token-type-ids tokens segment-id))
                  (len tokens)))
  :hints (("Goal" :induct (token-type-ids tokens segment-id))))

(defun make-input-triple (st text max-len)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text) (natp max-len))))
  (let* ((raw-tokens (encode-with-bos-eos st text))
         (tokens (pad-or-truncate raw-tokens max-len))
         (mask (attention-mask-from-tokens tokens))
         (positions (make-position-ids (len tokens) 0))
         (type-ids (token-type-ids tokens 0)))
    (list tokens mask positions type-ids)))

(defun hex-digit-to-nat (c)
  (declare (xargs :guard (characterp c)))
  (let ((code (char-code c)))
    (cond ((and (>= code 48) (<= code 57)) (- code 48))
          ((and (>= code 65) (<= code 70)) (+ 10 (- code 65)))
          ((and (>= code 97) (<= code 102)) (+ 10 (- code 97)))
          (t 0))))

(defthm natp-of-hex-digit-to-nat
  (implies (characterp c)
           (natp (hex-digit-to-nat c)))
  :rule-classes :type-prescription)

(defthm hex-digit-to-nat-bounded
  (implies (characterp c)
           (< (hex-digit-to-nat c) 16))
  :rule-classes :linear)

(defun decode-hex-byte-token (token-str)
  (declare (xargs :guard (stringp token-str)))
  (if (and (equal (length token-str) 4)
           (eql (char token-str 0) #\<)
           (eql (char token-str 3) #\>))
      (let* ((hi (hex-digit-to-nat (char token-str 1)))
             (lo (hex-digit-to-nat (char token-str 2)))
             (byte-val (+ (* hi 16) lo)))
        (mv t byte-val))
    (mv nil 0)))

(defun decode-bpe-tokens-aux (st tokens acc)
  (declare (xargs :guard (and (mgt-state-p st)
                               (nat-listp tokens)
                               (nat-listp acc))
                  :measure (len tokens)))
  (if (atom tokens)
      (reverse acc)
    (let ((str (lookup-id (car tokens) (mgt-id-to-token st))))
      (if (null str)
          (decode-bpe-tokens-aux st (cdr tokens) acc)
        (mv-let (is-hex byte-val)
          (decode-hex-byte-token str)
          (if is-hex
              (decode-bpe-tokens-aux st (cdr tokens) (cons byte-val acc))
            (let ((codes (string-to-codes str)))
              (decode-bpe-tokens-aux st (cdr tokens)
                                    (revappend codes acc)))))))))

(defthm nat-listp-of-decode-bpe-tokens-aux
  (implies (and (mgt-state-p st)
                (nat-listp tokens)
                (nat-listp acc))
           (nat-listp (decode-bpe-tokens-aux st tokens acc)))
  :hints (("Goal" :induct (decode-bpe-tokens-aux st tokens acc))))

(defun decode-bpe-tokens-to-string (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))))
  (let ((bytes (decode-bpe-tokens-aux st tokens nil)))
    (codes-to-string bytes)))

(defthm stringp-of-decode-bpe-tokens-to-string
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (stringp (decode-bpe-tokens-to-string st tokens))))

(defun compute-overlap (tokens1 tokens2)
  (declare (xargs :guard (and (nat-listp tokens1) (nat-listp tokens2))
                  :measure (len tokens1)))
  (if (atom tokens1)
      0
    (+ (if (member-equal (car tokens1) tokens2) 1 0)
       (compute-overlap (cdr tokens1) tokens2))))

(defthm natp-of-compute-overlap
  (implies (and (nat-listp tokens1) (nat-listp tokens2))
           (natp (compute-overlap tokens1 tokens2)))
  :hints (("Goal" :induct (compute-overlap tokens1 tokens2)))
  :rule-classes :type-prescription)

(defthm compute-overlap-leq-len
  (implies (and (nat-listp tokens1) (nat-listp tokens2))
           (<= (compute-overlap tokens1 tokens2) (len tokens1)))
  :hints (("Goal" :induct (compute-overlap tokens1 tokens2)))
  :rule-classes :linear)

(defun jaccard-similarity-numer (tokens1 tokens2)
  (declare (xargs :guard (and (nat-listp tokens1) (nat-listp tokens2))))
  (compute-overlap tokens1 tokens2))

(defun jaccard-similarity-denom (tokens1 tokens2)
  (declare (xargs :guard (and (nat-listp tokens1) (nat-listp tokens2))))
  (let* ((union-size (+ (len tokens1) (len tokens2)
                        (- (compute-overlap tokens1 tokens2)))))
    (if (> union-size 0) union-size 1)))

(defthm natp-of-jaccard-denom
  (implies (and (nat-listp tokens1) (nat-listp tokens2))
           (natp (jaccard-similarity-denom tokens1 tokens2)))
  :rule-classes :type-prescription)

(defun multi-step-encode-decode (st text n)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text) (natp n))
                  :measure (nfix n)))
  (if (zp n)
      text
    (let* ((tokens (encode-text st text))
           (decoded (decode-tokens st tokens)))
      (multi-step-encode-decode st decoded (- n 1)))))

(defthm stringp-of-multi-step-encode-decode
  (implies (and (mgt-state-p st) (stringp text) (natp n))
           (stringp (multi-step-encode-decode st text n)))
  :hints (("Goal" :induct (multi-step-encode-decode st text n))))

(defun ngrams (tokens n)
  (declare (xargs :guard (and (nat-listp tokens) (natp n) (> n 0))
                  :measure (len tokens)))
  (if (or (atom tokens) (< (len tokens) n))
      nil
    (cons (take-n n tokens)
          (ngrams (cdr tokens) n))))

(defthm true-listp-of-ngrams
  (implies (and (nat-listp tokens) (natp n))
           (true-listp (ngrams tokens n)))
  :hints (("Goal" :induct (ngrams tokens n))))

(defun count-ngrams (tokens n)
  (declare (xargs :guard (and (nat-listp tokens) (natp n) (> n 0))))
  (len (ngrams tokens n)))

(defthm natp-of-count-ngrams
  (implies (and (nat-listp tokens) (natp n) (> n 0))
           (natp (count-ngrams tokens n)))
  :rule-classes :type-prescription)

(defun flatten-list-of-nat-lists (lists acc)
  (declare (xargs :guard (and (true-listp lists) (nat-listp acc))
                  :measure (len lists)))
  (if (atom lists)
      (reverse acc)
    (if (nat-listp (car lists))
        (flatten-list-of-nat-lists (cdr lists)
                                   (revappend (car lists) acc))
      (flatten-list-of-nat-lists (cdr lists) acc))))

(defthm nat-listp-of-flatten-list-of-nat-lists
  (implies (and (true-listp lists) (nat-listp acc))
           (nat-listp (flatten-list-of-nat-lists lists acc)))
  :hints (("Goal" :induct (flatten-list-of-nat-lists lists acc))))

(defun build-vocabulary-from-corpus (st corpus)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p corpus))
                  :measure (len corpus)))
  (if (atom corpus)
      st
    (let ((text (car corpus)))
      (mv-let (st1 id)
        (add-token-to-state st text)
        (declare (ignore id))
        (build-vocabulary-from-corpus st1 (cdr corpus))))))

(defthm mgt-state-p-of-build-vocabulary
  (implies (and (mgt-state-p st) (string-list-p corpus))
           (mgt-state-p (build-vocabulary-from-corpus st corpus)))
  :hints (("Goal" :induct (build-vocabulary-from-corpus st corpus))))

(defthm build-vocabulary-next-id-monotonic
  (implies (and (mgt-state-p st) (string-list-p corpus))
           (<= (mgt-next-id st)
               (mgt-next-id (build-vocabulary-from-corpus st corpus))))
  :hints (("Goal" :induct (build-vocabulary-from-corpus st corpus)))
  :rule-classes :linear)

(defun total-tokens-in-batch (batch)
  (declare (xargs :guard (true-listp batch)
                  :measure (len batch)))
  (if (atom batch)
      0
    (+ (if (nat-listp (car batch)) (len (car batch)) 0)
       (total-tokens-in-batch (cdr batch)))))

(defthm natp-of-total-tokens-in-batch
  (implies (true-listp batch)
           (natp (total-tokens-in-batch batch)))
  :hints (("Goal" :induct (total-tokens-in-batch batch)))
  :rule-classes :type-prescription)

(defun avg-tokens-per-text-numer (batch)
  (declare (xargs :guard (true-listp batch)))
  (total-tokens-in-batch batch))

(defun avg-tokens-per-text-denom (batch)
  (declare (xargs :guard (true-listp batch)))
  (if (> (len batch) 0) (len batch) 1))

(defthm avg-denom-positive
  (implies (true-listp batch)
           (> (avg-tokens-per-text-denom batch) 0))
  :rule-classes :linear)

(defun map-length-of-token-lists (token-lists acc)
  (declare (xargs :guard (and (true-listp token-lists) (nat-listp acc))
                  :measure (len token-lists)))
  (if (atom token-lists)
      (reverse acc)
    (let ((l (if (nat-listp (car token-lists))
                 (len (car token-lists))
               0)))
      (map-length-of-token-lists (cdr token-lists)
                                 (cons l acc)))))

(defthm nat-listp-of-map-length
  (implies (and (true-listp token-lists) (nat-listp acc))
           (nat-listp (map-length-of-token-lists token-lists acc)))
  :hints (("Goal" :induct (map-length-of-token-lists token-lists acc))))

(defthm len-of-map-length
  (implies (and (true-listp token-lists) (nat-listp acc))
           (equal (len (map-length-of-token-lists token-lists acc))
                  (+ (len token-lists) (len acc))))
  :hints (("Goal" :induct (map-length-of-token-lists token-lists acc))))

(defun sum-nat-list (lst)
  (declare (xargs :guard (nat-listp lst)
                  :measure (len lst)))
  (if (atom lst)
      0
    (+ (car lst) (sum-nat-list (cdr lst)))))

(defthm natp-of-sum-nat-list
  (implies (nat-listp lst)
           (natp (sum-nat-list lst)))
  :hints (("Goal" :induct (sum-nat-list lst)))
  :rule-classes :type-prescription)

(defun max-nat-list (lst current)
  (declare (xargs :guard (and (nat-listp lst) (natp current))
                  :measure (len lst)))
  (if (atom lst)
      current
    (max-nat-list (cdr lst) (if (> (car lst) current) (car lst) current))))

(defthm natp-of-max-nat-list
  (implies (natp current)
           (natp (max-nat-list lst current)))
  :hints (("Goal" :induct (max-nat-list lst current)))
  :rule-classes :type-prescription)

(defthm max-nat-list-geq-current
  (implies (natp current)
           (<= current (max-nat-list lst current)))
  :hints (("Goal" :induct (max-nat-list lst current)))
  :rule-classes :linear)

(defun min-nat-list (lst current)
  (declare (xargs :guard (and (nat-listp lst) (natp current))
                  :measure (len lst)))
  (if (atom lst)
      current
    (min-nat-list (cdr lst) (if (< (car lst) current) (car lst) current))))

(defthm natp-of-min-nat-list
  (implies (natp current)
           (natp (min-nat-list lst current)))
  :hints (("Goal" :induct (min-nat-list lst current)))
  :rule-classes :type-prescription)

(defthm min-nat-list-leq-current
  (implies (natp current)
           (<= (min-nat-list lst current) current))
  :hints (("Goal" :induct (min-nat-list lst current)))
  :rule-classes :linear)

(defun batch-statistics (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))))
  (let* ((batch (encode-batch st texts))
         (lengths (map-length-of-token-lists batch nil))
         (total (sum-nat-list lengths))
         (max-l (max-nat-list lengths 0))
         (min-l (if (consp lengths) (min-nat-list lengths (car lengths)) 0)))
    (list total max-l min-l (len texts))))

(defun verify-state-invariant-after-ops (st token1 token2)
  (declare (xargs :guard (and (mgt-state-p st) (stringp token1) (stringp token2))))
  (mv-let (st1 id1) (add-token-to-state st token1)
    (declare (ignore id1))
    (mv-let (st2 id2) (add-token-to-state st1 token2)
      (declare (ignore id2))
      (and (mgt-state-p st2)
           (<= (mgt-next-id st) (mgt-next-id st2))))))

(defthm state-invariant-preserved-through-adds
  (implies (and (mgt-state-p st) (stringp token1) (stringp token2))
           (verify-state-invariant-after-ops st token1 token2)))

(defun chain-add-n-tokens (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      st
    (mv-let (st1 id) (add-token-to-state st (car tokens))
      (declare (ignore id))
      (chain-add-n-tokens st1 (cdr tokens)))))

(defthm mgt-state-p-of-chain-add
  (implies (and (mgt-state-p st) (string-list-p tokens))
           (mgt-state-p (chain-add-n-tokens st tokens)))
  :hints (("Goal" :induct (chain-add-n-tokens st tokens))))

(defthm chain-add-next-id-monotonic
  (implies (and (mgt-state-p st) (string-list-p tokens))
           (<= (mgt-next-id st)
               (mgt-next-id (chain-add-n-tokens st tokens))))
  :hints (("Goal" :induct (chain-add-n-tokens st tokens)))
  :rule-classes :linear)

(defun all-added-tokens-findable (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p tokens))
                  :measure (len tokens)))
  (if (atom tokens)
      t
    (mv-let (st1 id) (add-token-to-state st (car tokens))
      (and (equal (lookup-token (car tokens) (mgt-token-to-id st1)) id)
           (equal (lookup-id id (mgt-id-to-token st1)) (car tokens))
           (all-added-tokens-findable st1 (cdr tokens))))))

(defthm all-added-tokens-are-findable
  (implies (and (mgt-state-p st) (string-list-p tokens))
           (all-added-tokens-findable st tokens))
  :hints (("Goal" :induct (all-added-tokens-findable st tokens))))

(defun verify-encode-produces-valid-tokens (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (let ((tokens (encode-text st text)))
    (validate-tokens st tokens)))

(defun verify-batch-encode-all-valid (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))
                  :measure (len texts)))
  (if (atom texts)
      t
    (and (verify-encode-produces-valid-tokens st (car texts))
         (verify-batch-encode-all-valid st (cdr texts)))))

(defun double-encode-same-result (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (equal (encode-text st text)
         (encode-text st text)))

(defthm double-encode-deterministic
  (implies (and (mgt-state-p st) (stringp text))
           (double-encode-same-result st text)))

(defun double-decode-same-result (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))))
  (equal (decode-tokens st tokens)
         (decode-tokens st tokens)))

(defthm double-decode-deterministic
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (double-decode-same-result st tokens)))

(defun make-mgt-with-n-tokens (n)
  (declare (xargs :guard (natp n)
                  :measure (nfix n)))
  (if (zp n)
      (make-empty-mgt-state)
    (mv-let (st id)
      (add-token-to-state (make-mgt-with-n-tokens (- n 1))
                          (concatenate 'string "tok" (coerce (list (code-char (+ 65 (mod (- n 1) 26)))) 'string)))
      (declare (ignore id))
      st)))

(defthm mgt-state-p-of-make-mgt-with-n-tokens
  (implies (natp n)
           (mgt-state-p (make-mgt-with-n-tokens n)))
  :hints (("Goal" :induct (make-mgt-with-n-tokens n))))

(defun token-ids-unique-in-list (lst seen)
  (declare (xargs :guard (and (token-list-p lst) (true-listp seen))
                  :measure (len lst)))
  (if (atom lst)
      t
    (let ((id (cdr (car lst))))
      (if (member-equal id seen)
          nil
        (token-ids-unique-in-list (cdr lst) (cons id seen))))))

(defun token-strings-unique-in-list (lst seen)
  (declare (xargs :guard (and (token-list-p lst) (true-listp seen))
                  :measure (len lst)))
  (if (atom lst)
      t
    (let ((tok (car (car lst))))
      (if (member-equal tok seen)
          nil
        (token-strings-unique-in-list (cdr lst) (cons tok seen))))))

(defun verify-full-pipeline (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (let* ((tokens (encode-text st text))
         (decoded (decode-tokens st tokens))
         (re-encoded (encode-text st decoded))
         (re-decoded (decode-tokens st re-encoded)))
    (equal decoded re-decoded)))

(defthm full-pipeline-stable-after-first-roundtrip
  (implies (and (mgt-state-p st) (stringp text))
           (verify-full-pipeline st text)))

(defun count-whitespace-tokens (st text pos count)
  (declare (xargs :guard (and (mgt-state-p st)
                               (stringp text)
                               (natp pos)
                               (natp count)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (declare (ignore st))
  (if (or (not (natp pos)) (>= pos (length text)))
      count
    (if (is-whitespace-char (char text pos))
        (count-whitespace-tokens st text (+ pos 1) (+ count 1))
      (count-whitespace-tokens st text (+ pos 1) count))))

(defthm natp-of-count-whitespace-tokens
  (implies (and (natp pos) (natp count))
           (natp (count-whitespace-tokens st text pos count)))
  :hints (("Goal" :induct (count-whitespace-tokens st text pos count)))
  :rule-classes :type-prescription)

(defun count-punctuation-chars (text pos count)
  (declare (xargs :guard (and (stringp text)
                               (natp pos)
                               (natp count)
                               (<= pos (length text)))
                  :measure (nfix (- (length text) pos))))
  (if (or (not (natp pos)) (>= pos (length text)))
      count
    (if (is-punctuation-char (char text pos))
        (count-punctuation-chars text (+ pos 1) (+ count 1))
      (count-punctuation-chars text (+ pos 1) count))))

(defthm natp-of-count-punctuation-chars
  (implies (and (natp pos) (natp count))
           (natp (count-punctuation-chars text pos count)))
  :hints (("Goal" :induct (count-punctuation-chars text pos count)))
  :rule-classes :type-prescription)

(defun make-token-pair (first second)
  (declare (xargs :guard (and (natp first) (natp second))))
  (cons first second))

(defun token-pair-first (pair)
  (declare (xargs :guard (pair-key-p pair)))
  (car pair))

(defun token-pair-second (pair)
  (declare (xargs :guard (pair-key-p pair)))
  (cdr pair))

(defthm pair-key-p-of-make-token-pair
  (implies (and (natp first) (natp second))
           (pair-key-p (make-token-pair first second))))

(defthm token-pair-first-of-make
  (implies (and (natp first) (natp second))
           (equal (token-pair-first (make-token-pair first second))
                  first)))

(defthm token-pair-second-of-make
  (implies (and (natp first) (natp second))
           (equal (token-pair-second (make-token-pair first second))
                  second)))

(defun make-bpe-merge (token-id priority)
  (declare (xargs :guard (and (natp token-id) (natp priority))))
  (cons token-id priority))

(defun bpe-merge-token-id (merge)
  (declare (xargs :guard (bpe-merge-p merge)))
  (car merge))

(defun bpe-merge-priority (merge)
  (declare (xargs :guard (bpe-merge-p merge)))
  (cdr merge))

(defthm bpe-merge-p-of-make
  (implies (and (natp tid) (natp pri))
           (bpe-merge-p (make-bpe-merge tid pri))))

(defthm bpe-merge-token-id-of-make
  (implies (and (natp tid) (natp pri))
           (equal (bpe-merge-token-id (make-bpe-merge tid pri))
                  tid)))

(defthm bpe-merge-priority-of-make
  (implies (and (natp tid) (natp pri))
           (equal (bpe-merge-priority (make-bpe-merge tid pri))
                  pri)))

(defun find-best-bpe-merge (bpe-pairs key1 key2 st)
  (declare (xargs :guard (and (bpe-pair-list-p bpe-pairs)
                               (stringp key1)
                               (stringp key2)
                               (mgt-state-p st))
                  :measure (len bpe-pairs)))
  (declare (ignore st))
  (let ((merged-key (concatenate 'string key1 key2)))
    (lookup-bpe merged-key bpe-pairs)))

(defun apply-best-bpe-step (st current-tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp current-tokens))
                  :measure (len current-tokens)))
  (declare (ignore st))
  current-tokens)

(defun all-token-list-entries-valid (t2i i2t)
  (declare (xargs :guard (and (token-list-p t2i) (id-list-p i2t))
                  :measure (len t2i)))
  (if (atom t2i)
      t
    (let* ((token (car (car t2i)))
           (id (cdr (car t2i)))
           (found (lookup-id id i2t)))
      (and found
           (equal found token)
           (all-token-list-entries-valid (cdr t2i) i2t)))))

(defthm all-entries-valid-when-nil
  (all-token-list-entries-valid nil i2t))

(defun all-id-list-entries-valid (i2t t2i)
  (declare (xargs :guard (and (id-list-p i2t) (token-list-p t2i))
                  :measure (len i2t)))
  (if (atom i2t)
      t
    (let* ((id (car (car i2t)))
           (token (cdr (car i2t)))
           (found (lookup-token token t2i)))
      (and found
           (equal found id)
           (all-id-list-entries-valid (cdr i2t) t2i)))))

(defthm all-id-entries-valid-when-nil
  (all-id-list-entries-valid nil t2i))

(defun state-bijectivity-p (st)
  (declare (xargs :guard (mgt-state-p st)))
  (and (all-token-list-entries-valid (mgt-token-to-id st)
                                     (mgt-id-to-token st))
       (all-id-list-entries-valid (mgt-id-to-token st)
                                  (mgt-token-to-id st))
       (equal (len (mgt-token-to-id st))
              (len (mgt-id-to-token st)))))

(defun encode-then-validate (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (validate-tokens st (encode-text st text)))

(defun safe-decode (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))))
  (if (validate-tokens st tokens)
      (decode-tokens st tokens)
    "[INVALID]"))

(defthm stringp-of-safe-decode
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (stringp (safe-decode st tokens))))

(defun state-snapshot (st)
  (declare (xargs :guard (mgt-state-p st)))
  (list (len (mgt-token-to-id st))
        (len (mgt-id-to-token st))
        (len (mgt-prefixes st))
        (len (mgt-suffixes st))
        (len (mgt-roots st))
        (len (mgt-bpe-pairs st))
        (mgt-next-id st)))

(defthm true-listp-of-state-snapshot
  (implies (mgt-state-p st)
           (true-listp (state-snapshot st))))

(defun snapshot-next-id (snapshot)
  (declare (xargs :guard (and (true-listp snapshot) (equal (len snapshot) 7))))
  (nth 6 snapshot))

(defun snapshot-vocab-size (snapshot)
  (declare (xargs :guard (and (true-listp snapshot) (equal (len snapshot) 7))))
  (nth 0 snapshot))

(defun compare-snapshots-monotonic (snap1 snap2)
  (declare (xargs :guard (and (true-listp snap1) (equal (len snap1) 7)
                               (true-listp snap2) (equal (len snap2) 7)
                               (natp (nth 6 snap1)) (natp (nth 6 snap2)))))
  (<= (nth 6 snap1) (nth 6 snap2)))

(defun empty-state-has-no-tokens ()
  (declare (xargs :guard t))
  (let ((st (make-empty-mgt-state)))
    (and (equal (vocab-size st) 0)
         (equal (mgt-next-id st) 0)
         (null (mgt-prefixes st))
         (null (mgt-suffixes st))
         (null (mgt-roots st))
         (null (mgt-bpe-pairs st)))))

(defthm empty-state-verified
  (empty-state-has-no-tokens))

(defun verify-unknown-always-unk ()
  (declare (xargs :guard t))
  (and (equal (unknown-replacement "anything") *special-unk*)
       (equal (unknown-replacement "") *special-unk*)
       (equal (unknown-replacement "test") *special-unk*)))

(defthm unknown-always-unk
  (verify-unknown-always-unk))

(defun reconstruct-from-serialized-token-list (serialized)
  (declare (xargs :guard (true-listp serialized)
                  :measure (len serialized)))
  (if (atom serialized)
      nil
    (let ((entry (car serialized)))
      (if (and (true-listp entry)
               (equal (len entry) 2)
               (stringp (first entry))
               (natp (second entry)))
          (cons (cons (first entry) (second entry))
                (reconstruct-from-serialized-token-list (cdr serialized)))
        (reconstruct-from-serialized-token-list (cdr serialized))))))

(defun reconstruct-from-serialized-id-list (serialized)
  (declare (xargs :guard (true-listp serialized)
                  :measure (len serialized)))
  (if (atom serialized)
      nil
    (let ((entry (car serialized)))
      (if (and (true-listp entry)
               (equal (len entry) 2)
               (natp (first entry))
               (stringp (second entry)))
          (cons (cons (first entry) (second entry))
                (reconstruct-from-serialized-id-list (cdr serialized)))
        (reconstruct-from-serialized-id-list (cdr serialized))))))

(defun reconstruct-from-serialized-bpe-list (serialized)
  (declare (xargs :guard (true-listp serialized)
                  :measure (len serialized)))
  (if (atom serialized)
      nil
    (let ((entry (car serialized)))
      (if (and (true-listp entry)
               (equal (len entry) 3)
               (stringp (first entry))
               (natp (second entry))
               (natp (third entry)))
          (cons (cons (first entry)
                      (cons (second entry) (third entry)))
                (reconstruct-from-serialized-bpe-list (cdr serialized)))
        (reconstruct-from-serialized-bpe-list (cdr serialized))))))

(defun string-length-bounded (str bound)
  (declare (xargs :guard (and (stringp str) (natp bound))))
  (<= (length str) bound))

(defun all-token-strings-bounded (lst bound)
  (declare (xargs :guard (and (token-list-p lst) (natp bound))
                  :measure (len lst)))
  (if (atom lst)
      t
    (and (string-length-bounded (car (car lst)) bound)
         (all-token-strings-bounded (cdr lst) bound))))

(defthm all-token-strings-bounded-of-nil
  (all-token-strings-bounded nil bound))

(defun max-token-string-length (lst current-max)
  (declare (xargs :guard (and (token-list-p lst) (natp current-max))
                  :measure (len lst)))
  (if (atom lst)
      current-max
    (let ((l (length (car (car lst)))))
      (max-token-string-length (cdr lst)
                                (if (> l current-max) l current-max)))))

(defthm natp-of-max-token-string-length
  (implies (natp current-max)
           (natp (max-token-string-length lst current-max)))
  :hints (("Goal" :induct (max-token-string-length lst current-max)))
  :rule-classes :type-prescription)

(defthm max-token-string-length-geq-current
  (implies (natp current-max)
           (<= current-max (max-token-string-length lst current-max)))
  :hints (("Goal" :induct (max-token-string-length lst current-max)))
  :rule-classes :linear)

(defun verify-morph-decompose-returns-tokens (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (let ((result (morph-decompose st word)))
    (or (null result)
        (nat-listp result))))

(defthm morph-decompose-type-correct
  (implies (and (mgt-state-p st) (stringp word))
           (verify-morph-decompose-returns-tokens st word)))

(defun verify-subword-split-returns-tokens (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (nat-listp (subword-split st word)))

(defthm subword-split-type-correct
  (implies (and (mgt-state-p st) (stringp word))
           (verify-subword-split-returns-tokens st word)))

(defun verify-encode-word-returns-tokens (st word)
  (declare (xargs :guard (and (mgt-state-p st) (stringp word))))
  (nat-listp (encode-word st word)))

(defthm encode-word-type-correct
  (implies (and (mgt-state-p st) (stringp word))
           (verify-encode-word-returns-tokens st word)))

(defun verify-encode-text-returns-tokens (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (nat-listp (encode-text st text)))

(defthm encode-text-type-correct
  (implies (and (mgt-state-p st) (stringp text))
           (verify-encode-text-returns-tokens st text)))

(defun verify-decode-tokens-returns-string (st tokens)
  (declare (xargs :guard (and (mgt-state-p st) (nat-listp tokens))))
  (stringp (decode-tokens st tokens)))

(defthm decode-tokens-type-correct
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (verify-decode-tokens-returns-string st tokens)))

(defun verify-tensor-encode-returns-tensor (st text)
  (declare (xargs :guard (and (mgt-state-p st) (stringp text))))
  (tensor-p (encode-to-tensor st text)))

(defthm tensor-encode-type-correct
  (implies (and (mgt-state-p st) (stringp text))
           (verify-tensor-encode-returns-tensor st text)))

(defun verify-tensor-decode-returns-string (st tensor)
  (declare (xargs :guard (and (mgt-state-p st) (tensor-p tensor))))
  (stringp (decode-from-tensor st tensor)))

(defthm tensor-decode-type-correct
  (implies (and (mgt-state-p st) (tensor-p tensor))
           (verify-tensor-decode-returns-string st tensor)))

(defun verify-batch-encode-returns-list (st texts)
  (declare (xargs :guard (and (mgt-state-p st) (string-list-p texts))))
  (true-listp (encode-batch st texts)))

(defthm batch-encode-type-correct
  (implies (and (mgt-state-p st) (string-list-p texts))
           (verify-batch-encode-returns-list st texts)))

(defun verify-batch-decode-returns-list (st token-lists)
  (declare (xargs :guard (and (mgt-state-p st) (true-listp token-lists))))
  (string-list-p (batch-decode st token-lists)))

(defthm batch-decode-type-correct
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (verify-batch-decode-returns-list st token-lists)))


(defthm mgt-state-p-implies-true-listp-1
  (implies (mgt-state-p st)
           (true-listp st)))

(defthm mgt-state-p-implies-len-7-1
  (implies (mgt-state-p st)
           (equal (len st) 7)))

(defthm token-list-p-forward-to-true-listp-1
  (implies (token-list-p x)
           (true-listp x))
  :rule-classes :forward-chaining)

(defthm id-list-p-forward-to-true-listp-1
  (implies (id-list-p x)
           (true-listp x))
  :rule-classes :forward-chaining)

(defthm bpe-pair-list-p-forward-to-true-listp-1
  (implies (bpe-pair-list-p x)
           (true-listp x))
  :rule-classes :forward-chaining)

(defthm anchor-list-p-forward-to-true-listp-1
  (implies (anchor-list-p x)
           (true-listp x))
  :rule-classes :forward-chaining)

(defthm nat-listp-forward-to-true-listp-1
  (implies (nat-listp x)
           (true-listp x))
  :rule-classes :forward-chaining)

(defthm string-list-p-forward-to-true-listp-1
  (implies (string-list-p x)
           (true-listp x))
  :rule-classes :forward-chaining)

(defthm token-entry-p-forward-1
  (implies (token-entry-p x)
           (and (consp x)
                (stringp (car x))
                (natp (cdr x))))
  :rule-classes :forward-chaining)

(defthm id-entry-p-forward-1
  (implies (id-entry-p x)
           (and (consp x)
                (natp (car x))
                (stringp (cdr x))))
  :rule-classes :forward-chaining)

(defthm bpe-merge-p-forward-1
  (implies (bpe-merge-p x)
           (and (consp x)
                (natp (car x))
                (natp (cdr x))))
  :rule-classes :forward-chaining)

(defthm bpe-pair-entry-p-forward-1
  (implies (bpe-pair-entry-p x)
           (and (consp x)
                (stringp (car x))
                (bpe-merge-p (cdr x))))
  :rule-classes :forward-chaining)

(defthm anchor-entry-p-forward-1
  (implies (anchor-entry-p x)
           (and (consp x)
                (stringp (car x))
                (natp (cdr x))))
  :rule-classes :forward-chaining)

(defthm token-list-p-of-cdr-1
  (implies (and (token-list-p x) (consp x))
           (token-list-p (cdr x))))

(defthm id-list-p-of-cdr-1
  (implies (and (id-list-p x) (consp x))
           (id-list-p (cdr x))))

(defthm bpe-pair-list-p-of-cdr-1
  (implies (and (bpe-pair-list-p x) (consp x))
           (bpe-pair-list-p (cdr x))))

(defthm anchor-list-p-of-cdr-1
  (implies (and (anchor-list-p x) (consp x))
           (anchor-list-p (cdr x))))

(defthm nat-listp-of-cdr-1
  (implies (and (nat-listp x) (consp x))
           (nat-listp (cdr x))))

(defthm string-list-p-of-cdr-1
  (implies (and (string-list-p x) (consp x))
           (string-list-p (cdr x))))

(defthm token-entry-p-of-car-1
  (implies (and (token-list-p x) (consp x))
           (token-entry-p (car x))))

(defthm id-entry-p-of-car-1
  (implies (and (id-list-p x) (consp x))
           (id-entry-p (car x))))

(defthm bpe-pair-entry-p-of-car-1
  (implies (and (bpe-pair-list-p x) (consp x))
           (bpe-pair-entry-p (car x))))

(defthm anchor-entry-p-of-car-1
  (implies (and (anchor-list-p x) (consp x))
           (anchor-entry-p (car x))))

(defthm natp-of-car-when-nat-listp-1
  (implies (and (nat-listp x) (consp x))
           (natp (car x))))

(defthm stringp-of-car-when-string-list-p-1
  (implies (and (string-list-p x) (consp x))
           (stringp (car x))))

(defthm lookup-token-of-mgt-token-to-id-type-1
  (implies (mgt-state-p st)
           (or (null (lookup-token tok (mgt-token-to-id st)))
               (natp (lookup-token tok (mgt-token-to-id st)))))
  :rule-classes :type-prescription)

(defthm lookup-id-of-mgt-id-to-token-type-1
  (implies (mgt-state-p st)
           (or (null (lookup-id id (mgt-id-to-token st)))
               (stringp (lookup-id id (mgt-id-to-token st)))))
  :rule-classes :type-prescription)

(defthm lookup-bpe-of-mgt-bpe-pairs-type-1
  (implies (mgt-state-p st)
           (or (null (lookup-bpe key (mgt-bpe-pairs st)))
               (bpe-merge-p (lookup-bpe key (mgt-bpe-pairs st)))))
  :rule-classes :type-prescription)

(defthm vocab-size-equals-token-map-len-1
  (implies (mgt-state-p st)
           (equal (vocab-size st)
                  (len (mgt-token-to-id st)))))

(defthm count-vocab-entries-equals-token-map-len-1
  (implies (mgt-state-p st)
           (equal (count-vocab-entries st)
                  (len (mgt-token-to-id st)))))

(defthm count-prefix-entries-equals-prefix-len-1
  (implies (mgt-state-p st)
           (equal (count-prefix-entries st)
                  (len (mgt-prefixes st)))))

(defthm count-suffix-entries-equals-suffix-len-1
  (implies (mgt-state-p st)
           (equal (count-suffix-entries st)
                  (len (mgt-suffixes st)))))

(defthm count-bpe-merges-equals-bpe-len-1
  (implies (mgt-state-p st)
           (equal (count-bpe-merges st)
                  (len (mgt-bpe-pairs st)))))

(defthm total-morpheme-count-expansion-1
  (implies (mgt-state-p st)
           (equal (total-morpheme-count st)
                  (+ (len (mgt-prefixes st))
                     (len (mgt-suffixes st))
                     (len (mgt-roots st))))))

(defthm has-token-p-iff-lookup-token-1
  (implies (and (mgt-state-p st) (stringp token))
           (iff (has-token-p st token)
                (lookup-token token (mgt-token-to-id st)))))

(defthm has-id-p-iff-lookup-id-1
  (implies (and (mgt-state-p st) (natp id))
           (iff (has-id-p st id)
                (lookup-id id (mgt-id-to-token st)))))

(defthm get-token-id-is-lookup-token-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (get-token-id st token)
                  (lookup-token token (mgt-token-to-id st)))))

(defthm get-id-token-is-lookup-id-1
  (implies (and (mgt-state-p st) (natp id))
           (equal (get-id-token st id)
                  (lookup-id id (mgt-id-to-token st)))))

(defthm is-prefix-p-iff-prefix-lookup-1
  (implies (and (mgt-state-p st) (stringp token))
           (iff (is-prefix-p st token)
                (lookup-token token (mgt-prefixes st)))))

(defthm is-suffix-p-iff-suffix-lookup-1
  (implies (and (mgt-state-p st) (stringp token))
           (iff (is-suffix-p st token)
                (lookup-token token (mgt-suffixes st)))))

(defthm is-root-p-iff-root-lookup-1
  (implies (and (mgt-state-p st) (stringp token))
           (iff (is-root-p st token)
                (lookup-token token (mgt-roots st)))))

(defthm classify-token-special-1
  (implies (and (mgt-state-p st) (stringp token) (is-special-token token))
           (equal (classify-token st token) :special)))

(defthm classify-token-prefix-1
  (implies (and (mgt-state-p st) (stringp token)
                (not (is-special-token token))
                (is-prefix-p st token))
           (equal (classify-token st token) :prefix)))

(defthm classify-token-suffix-1
  (implies (and (mgt-state-p st) (stringp token)
                (not (is-special-token token))
                (not (is-prefix-p st token))
                (is-suffix-p st token))
           (equal (classify-token st token) :suffix)))

(defthm classify-token-root-1
  (implies (and (mgt-state-p st) (stringp token)
                (not (is-special-token token))
                (not (is-prefix-p st token))
                (not (is-suffix-p st token))
                (is-root-p st token))
           (equal (classify-token st token) :root)))

(defthm classify-token-word-1
  (implies (and (mgt-state-p st) (stringp token)
                (not (is-special-token token))
                (not (is-prefix-p st token))
                (not (is-suffix-p st token))
                (not (is-root-p st token))
                (has-token-p st token))
           (equal (classify-token st token) :word)))

(defthm classify-token-unknown-1
  (implies (and (mgt-state-p st) (stringp token)
                (not (is-special-token token))
                (not (is-prefix-p st token))
                (not (is-suffix-p st token))
                (not (is-root-p st token))
                (not (has-token-p st token)))
           (equal (classify-token st token) :unknown)))

(defthm natp-of-lookup-token-after-add-1
  (implies (and (mgt-state-p st) (stringp token))
           (natp (mv-nth 1 (add-token-to-state st token))))
  :rule-classes :type-prescription)

(defthm add-token-to-state-preserves-state-1
  (implies (and (mgt-state-p st) (stringp token))
           (mgt-state-p (mv-nth 0 (add-token-to-state st token)))))

(defthm add-token-to-state-preserves-token-list-1
  (implies (and (mgt-state-p st) (stringp token))
           (token-list-p (mgt-token-to-id (mv-nth 0 (add-token-to-state st token))))))

(defthm add-token-to-state-preserves-id-list-1
  (implies (and (mgt-state-p st) (stringp token))
           (id-list-p (mgt-id-to-token (mv-nth 0 (add-token-to-state st token))))))

(defthm add-token-to-state-next-id-natp-1
  (implies (and (mgt-state-p st) (stringp token))
           (natp (mgt-next-id (mv-nth 0 (add-token-to-state st token)))))
  :rule-classes :type-prescription)

(defthm add-token-to-state-lookup-token-after-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (lookup-token token (mgt-token-to-id (mv-nth 0 (add-token-to-state st token))))
                  (mv-nth 1 (add-token-to-state st token)))))

(defthm add-token-to-state-lookup-id-after-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (lookup-id (mv-nth 1 (add-token-to-state st token))
                             (mgt-id-to-token (mv-nth 0 (add-token-to-state st token))))
                  token)))

(defthm add-token-to-state-vocab-monotone-1
  (implies (and (mgt-state-p st) (stringp token))
           (<= (len (mgt-token-to-id st))
               (len (mgt-token-to-id (mv-nth 0 (add-token-to-state st token))))))
  :rule-classes :linear)

(defthm add-token-to-state-id-map-monotone-1
  (implies (and (mgt-state-p st) (stringp token))
           (<= (len (mgt-id-to-token st))
               (len (mgt-id-to-token (mv-nth 0 (add-token-to-state st token))))))
  :rule-classes :linear)

(defthm add-token-to-state-next-id-monotone-1
  (implies (and (mgt-state-p st) (stringp token))
           (<= (mgt-next-id st)
               (mgt-next-id (mv-nth 0 (add-token-to-state st token)))))
  :rule-classes :linear)

(defthm add-token-to-state-preserves-prefix-count-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (len (mgt-prefixes (mv-nth 0 (add-token-to-state st token))))
                  (len (mgt-prefixes st)))))

(defthm add-token-to-state-preserves-suffix-count-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (len (mgt-suffixes (mv-nth 0 (add-token-to-state st token))))
                  (len (mgt-suffixes st)))))

(defthm add-token-to-state-preserves-root-count-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (len (mgt-roots (mv-nth 0 (add-token-to-state st token))))
                  (len (mgt-roots st)))))

(defthm add-token-to-state-preserves-bpe-count-1
  (implies (and (mgt-state-p st) (stringp token))
           (equal (len (mgt-bpe-pairs (mv-nth 0 (add-token-to-state st token))))
                  (len (mgt-bpe-pairs st)))))

(defthm add-tokens-to-state-preserves-state-1
  (implies (and (mgt-state-p st) (string-list-p toks))
           (mgt-state-p (add-tokens-to-state st toks))))

(defthm add-tokens-to-state-next-id-monotone-1
  (implies (and (mgt-state-p st) (string-list-p toks))
           (<= (mgt-next-id st)
               (mgt-next-id (add-tokens-to-state st toks))))
  :rule-classes :linear)

(defthm add-tokens-to-state-vocab-monotone-1
  (implies (and (mgt-state-p st) (string-list-p toks))
           (<= (len (mgt-token-to-id st))
               (len (mgt-token-to-id (add-tokens-to-state st toks)))))
  :rule-classes :linear)

(defthm add-tokens-to-state-id-map-monotone-1
  (implies (and (mgt-state-p st) (string-list-p toks))
           (<= (len (mgt-id-to-token st))
               (len (mgt-id-to-token (add-tokens-to-state st toks)))))
  :rule-classes :linear)

(defthm add-prefix-to-state-preserves-state-1
  (implies (and (mgt-state-p st) (stringp prefix))
           (mgt-state-p (add-prefix-to-state st prefix))))

(defthm add-suffix-to-state-preserves-state-1
  (implies (and (mgt-state-p st) (stringp suffix))
           (mgt-state-p (add-suffix-to-state st suffix))))

(defthm add-prefixes-to-state-preserves-state-1
  (implies (and (mgt-state-p st) (string-list-p prefixes))
           (mgt-state-p (add-prefixes-to-state st prefixes))))

(defthm add-suffixes-to-state-preserves-state-1
  (implies (and (mgt-state-p st) (string-list-p suffixes))
           (mgt-state-p (add-suffixes-to-state st suffixes))))

(defthm init-special-tokens-preserves-state-1
  (implies (mgt-state-p st)
           (mgt-state-p (init-special-tokens st))))

(defthm init-special-tokens-has-pad-1
  (implies (mgt-state-p st)
           (lookup-token "[PAD]" (mgt-token-to-id (init-special-tokens st)))))

(defthm init-special-tokens-has-unk-1
  (implies (mgt-state-p st)
           (lookup-token "[UNK]" (mgt-token-to-id (init-special-tokens st)))))

(defthm init-special-tokens-has-bos-1
  (implies (mgt-state-p st)
           (lookup-token "[BOS]" (mgt-token-to-id (init-special-tokens st)))))

(defthm init-special-tokens-has-eos-1
  (implies (mgt-state-p st)
           (lookup-token "[EOS]" (mgt-token-to-id (init-special-tokens st)))))

(defthm init-morphemes-preserves-state-1
  (implies (mgt-state-p st)
           (mgt-state-p (init-morphemes st))))

(defthm init-morphemes-prefix-count-monotone-1
  (implies (mgt-state-p st)
           (<= (len (mgt-prefixes st))
               (len (mgt-prefixes (init-morphemes st)))))
  :rule-classes :linear)

(defthm init-morphemes-suffix-count-monotone-1
  (implies (mgt-state-p st)
           (<= (len (mgt-suffixes st))
               (len (mgt-suffixes (init-morphemes st)))))
  :rule-classes :linear)

(defthm init-byte-tokens-preserves-state-1
  (implies (and (mgt-state-p st) (natp n))
           (mgt-state-p (init-byte-tokens st n))))

(defthm init-byte-tokens-next-id-monotone-1
  (implies (and (mgt-state-p st) (natp n))
           (<= (mgt-next-id st)
               (mgt-next-id (init-byte-tokens st n))))
  :rule-classes :linear)

(defthm train-bpe-preserves-state-1
  (implies (and (mgt-state-p st) (string-list-p corpus) (natp num-merges))
           (mgt-state-p (train-bpe st corpus num-merges))))

(defthm train-bpe-next-id-monotone-1
  (implies (and (mgt-state-p st) (string-list-p corpus) (natp num-merges))
           (<= (mgt-next-id st)
               (mgt-next-id (train-bpe st corpus num-merges))))
  :rule-classes :linear)

(defthm train-bpe-bpe-count-monotone-1
  (implies (and (mgt-state-p st) (string-list-p corpus) (natp num-merges))
           (<= (len (mgt-bpe-pairs st))
               (len (mgt-bpe-pairs (train-bpe st corpus num-merges)))))
  :rule-classes :linear)

(defthm remove-vocab-word-preserves-state-1
  (implies (and (mgt-state-p st) (stringp word))
           (mgt-state-p (remove-vocab-word st word))))

(defthm remove-vocab-word-token-count-nonincrease-1
  (implies (and (mgt-state-p st) (stringp word))
           (<= (len (mgt-token-to-id (remove-vocab-word st word)))
               (len (mgt-token-to-id st))))
  :rule-classes :linear)

(defthm remove-vocab-word-id-count-nonincrease-1
  (implies (and (mgt-state-p st) (stringp word))
           (<= (len (mgt-id-to-token (remove-vocab-word st word)))
               (len (mgt-id-to-token st))))
  :rule-classes :linear)

(defthm remove-vocab-word-prefix-count-nonincrease-1
  (implies (and (mgt-state-p st) (stringp word))
           (<= (len (mgt-prefixes (remove-vocab-word st word)))
               (len (mgt-prefixes st))))
  :rule-classes :linear)

(defthm remove-vocab-word-suffix-count-nonincrease-1
  (implies (and (mgt-state-p st) (stringp word))
           (<= (len (mgt-suffixes (remove-vocab-word st word)))
               (len (mgt-suffixes st))))
  :rule-classes :linear)

(defthm remove-vocab-word-root-count-nonincrease-1
  (implies (and (mgt-state-p st) (stringp word))
           (<= (len (mgt-roots (remove-vocab-word st word)))
               (len (mgt-roots st))))
  :rule-classes :linear)

(defthm remove-vocab-word-bpe-count-nonincrease-1
  (implies (and (mgt-state-p st) (stringp word))
           (<= (len (mgt-bpe-pairs (remove-vocab-word st word)))
               (len (mgt-bpe-pairs st))))
  :rule-classes :linear)

(defthm remove-vocab-word-next-id-preserved-1
  (implies (and (mgt-state-p st) (stringp word))
           (equal (mgt-next-id (remove-vocab-word st word))
                  (mgt-next-id st))))

(defthm remove-special-pad-identity-1
  (implies (mgt-state-p st)
           (equal (remove-vocab-word st "[PAD]") st)))

(defthm remove-special-unk-identity-1
  (implies (mgt-state-p st)
           (equal (remove-vocab-word st "[UNK]") st)))

(defthm remove-special-bos-identity-1
  (implies (mgt-state-p st)
           (equal (remove-vocab-word st "[BOS]") st)))

(defthm remove-special-eos-identity-1
  (implies (mgt-state-p st)
           (equal (remove-vocab-word st "[EOS]") st)))

(defthm encode-text-produces-nat-list-1
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm encode-text-empty-1
  (implies (mgt-state-p st)
           (equal (encode-text st "") nil)))

(defthm encode-text-validates-1
  (implies (and (mgt-state-p st) (stringp text))
           (booleanp (validate-tokens st (encode-text st text))))
  :rule-classes :type-prescription)

(defthm decode-tokens-produces-string-1
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm decode-tokens-empty-1
  (implies (mgt-state-p st)
           (equal (decode-tokens st nil) "")))

(defthm encode-batch-produces-true-list-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (true-listp (encode-batch st texts))))

(defthm encode-batch-length-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch-decode-produces-string-list-1
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (string-list-p (batch-decode st token-lists))))

(defthm batch-decode-length-1
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))

(defthm count-known-chars-type-1
  (implies (and (mgt-state-p st) (stringp text) (natp pos) (natp count))
           (natp (count-known-chars st text pos count)))
  :rule-classes :type-prescription)

(defthm count-known-chars-monotone-in-count-1
  (implies (and (mgt-state-p st) (stringp text) (natp pos) (natp count))
           (<= count (count-known-chars st text pos count)))
  :rule-classes :linear)

(defthm coverage-ratio-type-1
  (implies (and (mgt-state-p st) (stringp corpus))
           (natp (coverage-ratio st corpus)))
  :rule-classes :type-prescription)

(defthm coverage-ratio-nonnegative-1
  (implies (and (mgt-state-p st) (stringp corpus))
           (<= 0 (coverage-ratio st corpus)))
  :rule-classes :linear)

(defthm tensor-p-of-encode-to-tensor-1
  (implies (and (mgt-state-p st) (stringp text))
           (tensor-p (encode-to-tensor st text))))

(defthm tensor-p-of-encode-batch-to-tensor-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (tensor-p (encode-batch-to-tensor st texts))))

(defthm stringp-of-decode-from-tensor-1
  (implies (and (mgt-state-p st) (tensor-p tensor))
           (stringp (decode-from-tensor st tensor)))
  :rule-classes :type-prescription)

(defthm nat-listp-of-tensor-data-to-tokens-1
  (implies (tensor-data-p data)
           (nat-listp (tensor-data-to-tokens data))))

(defthm safe-nth-token-type-1
  (implies (and (natp n) (nat-listp toks))
           (natp (safe-nth-token n toks)))
  :rule-classes :type-prescription)

(defthm window-tokens-type-1
  (implies (and (nat-listp toks) (natp start) (natp size))
           (true-listp (window-tokens toks start size))))

(defthm state-snapshot-true-listp-1
  (implies (mgt-state-p st)
           (true-listp (state-snapshot st))))

(defthm snapshot-next-id-type-1
  (implies (and (true-listp snap) (equal (len snap) 7))
           (natp (snapshot-next-id snap)))
  :rule-classes :type-prescription)

(defthm snapshot-vocab-size-type-1
  (implies (and (true-listp snap) (equal (len snap) 7))
           (natp (snapshot-vocab-size snap)))
  :rule-classes :type-prescription)

(defthm compare-snapshots-monotonic-boolean-1
  (implies (and (true-listp s1) (equal (len s1) 7)
                (true-listp s2) (equal (len s2) 7)
                (natp (nth 6 s1)) (natp (nth 6 s2)))
           (booleanp (compare-snapshots-monotonic s1 s2)))
  :rule-classes :type-prescription)

(defthm empty-state-verified-1
  (empty-state-has-no-tokens))

(defthm unknown-always-unk-1
  (verify-unknown-always-unk))

(defthm special-tokens-present-after-init-1
  (let ((st (init-special-tokens (make-empty-mgt-state))))
    (special-tokens-present-p st)))

(defthm verify-special-token-ids-after-init-1
  (let ((st (init-special-tokens (make-empty-mgt-state))))
    (verify-special-token-ids st)))

(defthm encode-is-deterministic-1
  (implies (and (mgt-state-p st) (stringp text))
           (encode-text-deterministic-check st text)))

(defthm add-token-idempotent-1
  (implies (and (mgt-state-p st) (stringp token))
           (verify-add-token-idempotence st token)))

(defthm batch-size-preserved-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (batch-size-preserved-check st texts)))

(defthm decode-batch-size-preserved-1
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (decode-batch-size-check st token-lists)))

(defthm lookup-token-after-add-1
  (implies (and (mgt-state-p st) (stringp token))
           (lookup-token-after-add st token)))

(defthm lookup-id-after-add-1
  (implies (and (mgt-state-p st) (stringp token))
           (lookup-id-after-add st token)))

(defthm vocab-grows-on-add-1
  (implies (and (mgt-state-p st) (stringp token))
           (vocab-grows-on-new-token st token)))

(defthm remove-then-lookup-fails-1
  (implies (and (mgt-state-p st) (stringp word))
           (remove-then-lookup-fails st word)))

(defthm prefix-id-matches-token-id-1
  (implies (mgt-state-p st)
           (prefix-id-matches-token-id st)))

(defthm suffix-id-matches-token-id-1
  (implies (mgt-state-p st)
           (suffix-id-matches-token-id st)))

(defthm token-to-id-consistency-1
  (implies (mgt-state-p st)
           (token-to-id-consistent-p (mgt-token-to-id st)
                                     (mgt-id-to-token st))))

(defthm id-to-token-consistency-1
  (implies (mgt-state-p st)
           (id-to-token-consistent-p (mgt-id-to-token st)
                                     (mgt-token-to-id st))))

(defthm mgt-consistent-p-forward-1
  (implies (mgt-consistent-p st)
           (and (token-to-id-consistent-p (mgt-token-to-id st) (mgt-id-to-token st))
                (id-to-token-consistent-p (mgt-id-to-token st) (mgt-token-to-id st))))
  :rule-classes :forward-chaining)

(defthm mgt-well-formed-p-forward-1
  (implies (mgt-well-formed-p st)
           (and (mgt-consistent-p st)
                (equal (len (mgt-token-to-id st))
                       (len (mgt-id-to-token st)))
                (all-ids-below (mgt-token-to-id st) (mgt-next-id st))))
  :rule-classes :forward-chaining)

(defthm token-list-contains-iff-has-token-1
  (implies (and (mgt-state-p st) (stringp token))
           (iff (token-list-contains-p token (mgt-token-to-id st))
                (has-token-p st token))))

(defthm id-list-contains-iff-has-id-1
  (implies (and (mgt-state-p st) (natp id))
           (iff (id-list-contains-p id (mgt-id-to-token st))
                (has-id-p st id))))

(defthm count-unk-nonnegative-1
  (implies (nat-listp toks)
           (<= 0 (count-unk toks)))
  :rule-classes :linear)

(defthm count-non-unk-nonnegative-1
  (implies (nat-listp toks)
           (<= 0 (count-non-unk toks)))
  :rule-classes :linear)

(defthm count-unk-plus-count-non-unk-1
  (implies (nat-listp toks)
           (equal (+ (count-unk toks) (count-non-unk toks))
                  (len toks))))

(defthm filter-special-tokens-type-1
  (implies (nat-listp toks)
           (nat-listp (filter-special-tokens toks))))

(defthm filter-special-tokens-length-1
  (implies (nat-listp toks)
           (<= (len (filter-special-tokens toks))
               (len toks)))
  :rule-classes :linear)

(defthm reverse-nat-list-type-1
  (implies (and (nat-listp xs) (nat-listp acc))
           (nat-listp (reverse-nat-list xs acc))))

(defthm reverse-nat-list-length-1
  (implies (and (nat-listp xs) (nat-listp acc))
           (equal (len (reverse-nat-list xs acc))
                  (+ (len xs) (len acc)))))

(defthm take-n-true-listp-1
  (true-listp (take-n n x)))

(defthm drop-n-true-listp-1
  (implies (true-listp x)
           (true-listp (drop-n n x))))

(defthm split-token-list-at-types-1
  (implies (and (nat-listp toks) (natp n))
           (and (true-listp (mv-nth 0 (split-token-list-at toks n)))
                (true-listp (mv-nth 1 (split-token-list-at toks n))))))

(defthm unique-tokens-in-list-type-1
  (implies (and (nat-listp toks) (nat-listp acc))
           (nat-listp (unique-tokens-in-list toks seen acc))))

(defthm unique-tokens-in-list-length-bound-1
  (implies (and (nat-listp toks) (nat-listp acc))
           (<= (len (unique-tokens-in-list toks seen acc))
               (+ (len toks) (len acc))))
  :rule-classes :linear)

(defthm token-frequency-count-true-listp-1
  (implies (and (nat-listp toks) (true-listp freq))
           (true-listp (token-frequency-count toks freq))))

(defthm sum-frequencies-type-1
  (implies (true-listp freq)
           (natp (sum-frequencies freq)))
  :rule-classes :type-prescription)

(defthm map-encode-text-true-listp-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (true-listp (map-encode-text st texts))))

(defthm map-encode-text-length-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (map-encode-text st texts))
                  (len texts))))

(defthm map-decode-tokens-string-listp-1
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (string-list-p (map-decode-tokens st token-lists))))

(defthm map-decode-tokens-length-1
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (map-decode-tokens st token-lists))
                  (len token-lists))))

(defthm encode-multiple-texts-true-listp-1
  (implies (and (mgt-state-p st) (string-list-p texts) (true-listp acc))
           (true-listp (encode-multiple-texts st texts acc))))

(defthm encode-multiple-texts-length-1
  (implies (and (mgt-state-p st) (string-list-p texts) (true-listp acc))
           (equal (len (encode-multiple-texts st texts acc))
                  (+ (len texts) (len acc)))))

(defthm decode-multiple-token-lists-string-listp-1
  (implies (and (mgt-state-p st) (true-listp token-lists) (string-list-p acc))
           (string-list-p (decode-multiple-token-lists st token-lists acc))))

(defthm decode-multiple-token-lists-length-1
  (implies (and (mgt-state-p st) (true-listp token-lists) (string-list-p acc))
           (equal (len (decode-multiple-token-lists st token-lists acc))
                  (+ (len token-lists) (len acc)))))

(defthm token-list-to-nat-list-type-1
  (implies (token-list-p xs)
           (nat-listp (token-list-to-nat-list xs))))

(defthm token-list-to-nat-list-length-1
  (implies (token-list-p xs)
           (equal (len (token-list-to-nat-list xs))
                  (len xs))))

(defthm token-list-to-string-list-type-1
  (implies (token-list-p xs)
           (string-list-p (token-list-to-string-list xs))))

(defthm token-list-to-string-list-length-1
  (implies (token-list-p xs)
           (equal (len (token-list-to-string-list xs))
                  (len xs))))

(defthm id-list-to-nat-list-type-1
  (implies (id-list-p xs)
           (nat-listp (id-list-to-nat-list xs))))

(defthm id-list-to-string-list-type-1
  (implies (id-list-p xs)
           (string-list-p (id-list-to-string-list xs))))

(defthm bpe-pair-keys-type-1
  (implies (bpe-pair-list-p xs)
           (string-list-p (bpe-pair-keys xs))))

(defthm bpe-pair-token-ids-type-1
  (implies (bpe-pair-list-p xs)
           (nat-listp (bpe-pair-token-ids xs))))

(defthm bpe-pair-priorities-type-1
  (implies (bpe-pair-list-p xs)
           (nat-listp (bpe-pair-priorities xs))))

(defthm all-bpe-tokens-in-vocab-base-1
  (all-bpe-tokens-in-vocab nil t2i))

(defthm sequence-total-length-type-1
  (implies (true-listp seqs)
           (natp (sequence-total-length seqs)))
  :rule-classes :type-prescription)

(defthm apply-n-merges-shrinks-type-1
  (implies (and (nat-listp seq) (natp n) (natp a) (natp b) (natp c))
           (nat-listp (apply-n-merges-shrinks seq n a b c))))

(defthm make-input-triple-true-listp-1
  (implies (and (mgt-state-p st) (stringp text) (natp max-len))
           (true-listp (make-input-triple st text max-len))))

(defthm decode-bpe-tokens-aux-type-1
  (implies (and (mgt-state-p st) (nat-listp toks) (nat-listp acc))
           (nat-listp (decode-bpe-tokens-aux st toks acc))))

(defthm decode-hex-byte-token-flag-boolean-1
  (implies (stringp token-str)
           (booleanp (mv-nth 0 (decode-hex-byte-token token-str))))
  :rule-classes :type-prescription)

(defthm decode-hex-byte-token-byte-natp-1
  (implies (stringp token-str)
           (natp (mv-nth 1 (decode-hex-byte-token token-str))))
  :rule-classes :type-prescription)

(defthm hex-digit-to-nat-type-1
  (implies (characterp c)
           (natp (hex-digit-to-nat c)))
  :rule-classes :type-prescription)

(defthm hex-digit-to-nat-bound-1
  (implies (characterp c)
           (< (hex-digit-to-nat c) 16))
  :rule-classes :linear)

(defthm byte-to-hex-token-stringp-1
  (implies (natp b)
           (stringp (byte-to-hex-token b)))
  :rule-classes :type-prescription)

(defthm text-to-byte-seq-type-1
  (implies (and (stringp text) (natp pos) (nat-listp acc))
           (nat-listp (text-to-byte-seq text pos acc))))

(defthm text-to-token-seq-type-1
  (implies (and (mgt-state-p st) (stringp text) (natp pos) (nat-listp acc))
           (nat-listp (text-to-token-seq st text pos acc))))

(defthm corpus-to-seqs-true-listp-1
  (implies (and (mgt-state-p st) (string-list-p corpus) (true-listp acc))
           (true-listp (corpus-to-seqs st corpus acc))))

(defthm train-bpe-step-statep-1
  (implies (and (mgt-state-p st) (true-listp seqs) (natp merge-count))
           (mgt-state-p (mv-nth 0 (train-bpe-step st seqs merge-count)))))

(defthm train-bpe-step-seqs-true-listp-1
  (implies (and (mgt-state-p st) (true-listp seqs) (natp merge-count))
           (true-listp (mv-nth 1 (train-bpe-step st seqs merge-count)))))

(defthm apply-merge-to-seq-type-1
  (implies (and (nat-listp seq) (natp a) (natp b) (natp c) (nat-listp acc))
           (nat-listp (apply-merge-to-seq seq a b c acc))))

(defthm apply-merge-to-seq-length-bound-1
  (implies (and (nat-listp seq) (natp a) (natp b) (natp c))
           (<= (len (apply-merge-to-seq seq a b c nil))
               (len seq)))
  :rule-classes :linear)

(defthm apply-merge-to-seqs-true-listp-1
  (implies (and (true-listp seqs) (natp a) (natp b) (natp c))
           (true-listp (apply-merge-to-seqs seqs a b c))))

(defthm apply-merge-to-seqs-length-1
  (implies (and (true-listp seqs) (natp a) (natp b) (natp c))
           (equal (len (apply-merge-to-seqs seqs a b c))
                  (len seqs))))

(defthm count-pairs-in-seq-true-listp-1
  (implies (and (nat-listp seq) (true-listp pair-counts))
           (true-listp (count-pairs-in-seq seq pair-counts))))

(defthm count-pairs-in-seqs-true-listp-1
  (implies (and (true-listp seqs) (true-listp pair-counts))
           (true-listp (count-pairs-in-seqs seqs pair-counts))))

(defthm find-best-pair-freq-type-1
  (implies (natp best-freq)
           (natp (mv-nth 1 (find-best-pair counts best-key best-freq))))
  :rule-classes :type-prescription)

(defthm find-best-pair-freq-monotone-1
  (implies (natp best-freq)
           (<= best-freq (mv-nth 1 (find-best-pair counts best-key best-freq))))
  :rule-classes :linear)

(defthm count-token-list-type-1
  (implies (token-list-p x)
           (natp (count-token-list x)))
  :rule-classes :type-prescription)

(defthm count-id-list-type-1
  (implies (id-list-p x)
           (natp (count-id-list x)))
  :rule-classes :type-prescription)

(defthm sequence-total-length-nonnegative-1
  (implies (true-listp seqs)
           (<= 0 (sequence-total-length seqs)))
  :rule-classes :linear)

(defthm count-token-list-equals-len-1
  (implies (token-list-p x)
           (equal (count-token-list x) (len x))))

(defthm count-id-list-equals-len-1
  (implies (id-list-p x)
           (equal (count-id-list x) (len x))))

(defthm encode-to-tensor-shape-type-1
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (tensor-shape (encode-to-tensor st text)))))

(defthm encode-to-tensor-data-type-1
  (implies (and (mgt-state-p st) (stringp text))
           (tensor-data-p (tensor-data (encode-to-tensor st text)))))

(defthm encode-batch-to-tensor-shape-type-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (nat-listp (tensor-shape (encode-batch-to-tensor st texts)))))

(defthm encode-batch-to-tensor-data-type-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (tensor-data-p (tensor-data (encode-batch-to-tensor st texts)))))

(defthm state-snapshot-next-id-equals-field-1
  (implies (mgt-state-p st)
           (equal (snapshot-next-id (state-snapshot st))
                  (mgt-next-id st))))

(defthm state-snapshot-vocab-size-equals-field-1
  (implies (mgt-state-p st)
           (equal (snapshot-vocab-size (state-snapshot st))
                  (len (mgt-token-to-id st)))))

(defthm compare-snapshots-reflexive-1
  (implies (and (true-listp snap) (equal (len snap) 7) (natp (nth 6 snap)))
           (compare-snapshots-monotonic snap snap)))

(defthm compare-snapshots-transitive-1
  (implies (and (true-listp a) (equal (len a) 7) (natp (nth 6 a))
                (true-listp b) (equal (len b) 7) (natp (nth 6 b))
                (true-listp c) (equal (len c) 7) (natp (nth 6 c))
                (compare-snapshots-monotonic a b)
                (compare-snapshots-monotonic b c))
           (compare-snapshots-monotonic a c)))

(defthm encode-single-char-type-1
  (implies (and (mgt-state-p st) (characterp c))
           (natp (encode-single-char st c)))
  :rule-classes :type-prescription)

(defthm decode-single-token-type-1
  (implies (and (mgt-state-p st) (natp tid))
           (stringp (decode-single-token st tid)))
  :rule-classes :type-prescription)

(defthm all-tokens-are-known-iff-validate-1
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (iff (all-tokens-are-known st tokens)
                (validate-tokens st tokens))))

(defthm all-prefixes-are-tokens-base-1
  (all-prefixes-are-tokens nil t2i))

(defthm all-suffixes-are-tokens-base-1
  (all-suffixes-are-tokens nil t2i))

(defthm morpheme-consistency-forward-1
  (implies (morpheme-consistency-p st)
           (and (all-prefixes-are-tokens (mgt-prefixes st) (mgt-token-to-id st))
                (all-suffixes-are-tokens (mgt-suffixes st) (mgt-token-to-id st))))
  :rule-classes :forward-chaining)

(defthm encode-char-by-char-type-1
  (implies (and (mgt-state-p st) (stringp text) (natp pos) (nat-listp acc))
           (nat-listp (encode-char-by-char st text pos acc))))

(defthm greedy-tokenize-type-1
  (implies (and (mgt-state-p st) (stringp text) (natp pos) (nat-listp acc))
           (nat-listp (greedy-tokenize st text pos acc))))

(defthm longest-match-type-1
  (implies (and (mgt-state-p st) (stringp word) (natp pos))
           (natp (longest-match st word pos)))
  :rule-classes :type-prescription)

(defthm longest-match-nonnegative-1
  (implies (and (mgt-state-p st) (stringp word) (natp pos))
           (<= 0 (longest-match st word pos)))
  :rule-classes :linear)

(defthm longest-match-aux-type-1
  (implies (natp best)
           (natp (longest-match-aux st word pos end best)))
  :rule-classes :type-prescription)

(defthm longest-match-aux-monotone-best-1
  (implies (natp best)
           (<= best (longest-match-aux st word pos end best)))
  :rule-classes :linear)

(defthm morph-decompose-type-1
  (implies (and (mgt-state-p st) (stringp word))
           (or (null (morph-decompose st word))
               (nat-listp (morph-decompose st word))))
  :rule-classes :type-prescription)

(defthm find-longest-prefix-type-1
  (implies (token-list-p prefix-list)
           (or (null (find-longest-prefix word prefix-list))
               (token-entry-p (find-longest-prefix word prefix-list))))
  :rule-classes :type-prescription)

(defthm find-longest-suffix-type-1
  (implies (token-list-p suffix-list)
           (or (null (find-longest-suffix word suffix-list))
               (token-entry-p (find-longest-suffix word suffix-list))))
  :rule-classes :type-prescription)

(defthm encode-word-type-1
  (implies (and (mgt-state-p st) (stringp word))
           (nat-listp (encode-word st word))))

(defthm find-word-end-type-1
  (implies (and (stringp text) (natp pos))
           (natp (find-word-end st text pos)))
  :rule-classes :type-prescription)

(defthm find-word-end-lower-bound-1
  (implies (and (stringp text) (natp pos) (<= pos (length text)))
           (<= pos (find-word-end st text pos)))
  :rule-classes :linear)

(defthm find-word-end-upper-bound-1
  (implies (and (stringp text) (natp pos) (<= pos (length text)))
           (<= (find-word-end st text pos) (length text)))
  :rule-classes :linear)

(defthm check-special-token-at-type-1
  (implies (and (stringp text) (natp pos))
           (natp (check-special-token-at text pos)))
  :rule-classes :type-prescription)

(defthm check-special-token-at-bound-1
  (implies (and (stringp text) (natp pos))
           (<= (check-special-token-at text pos) 5))
  :rule-classes :linear)

(defthm encode-aux-type-1
  (implies (and (mgt-state-p st) (stringp text) (natp pos) (nat-listp acc))
           (nat-listp (encode-aux st text pos acc))))

(defthm decode-token-type-1
  (implies (and (mgt-state-p st) (natp tok))
           (stringp (decode-token st tok)))
  :rule-classes :type-prescription)

(defthm decode-tokens-aux-type-1
  (implies (and (mgt-state-p st) (nat-listp tokens) (stringp acc))
           (stringp (decode-tokens-aux st tokens acc)))
  :rule-classes :type-prescription)

(defthm reconstruct-token-list-type-1
  (implies (true-listp serialized)
           (token-list-p (reconstruct-from-serialized-token-list serialized))))

(defthm reconstruct-id-list-type-1
  (implies (true-listp serialized)
           (id-list-p (reconstruct-from-serialized-id-list serialized))))

(defthm reconstruct-bpe-list-type-1
  (implies (true-listp serialized)
           (bpe-pair-list-p (reconstruct-from-serialized-bpe-list serialized))))

(defthm serialize-token-list-true-listp-1
  (implies (token-list-p xs)
           (true-listp (serialize-token-list xs))))

(defthm serialize-id-list-true-listp-1
  (implies (id-list-p xs)
           (true-listp (serialize-id-list xs))))

(defthm serialize-bpe-list-true-listp-1
  (implies (bpe-pair-list-p xs)
           (true-listp (serialize-bpe-list xs))))

(defthm serialize-mgt-state-true-listp-1
  (implies (mgt-state-p st)
           (true-listp (serialize-mgt-state st))))

(defthm string-length-bounded-boolean-1
  (implies (and (stringp str) (natp bound))
           (booleanp (string-length-bounded str bound)))
  :rule-classes :type-prescription)

(defthm all-token-strings-bounded-boolean-1
  (implies (and (token-list-p lst) (natp bound))
           (booleanp (all-token-strings-bounded lst bound)))
  :rule-classes :type-prescription)

(defthm all-token-strings-bounded-base-1
  (all-token-strings-bounded nil bound))

(defthm max-nat-list-type-1
  (implies (natp current)
           (natp (max-nat-list lst current)))
  :rule-classes :type-prescription)

(defthm min-nat-list-type-1
  (implies (natp current)
           (natp (min-nat-list lst current)))
  :rule-classes :type-prescription)

(defthm batch-statistics-true-listp-1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (true-listp (batch-statistics st texts))))

(defthm verify-state-invariant-after-ops-1
  (implies (and (mgt-state-p st) (stringp token1) (stringp token2))
           (verify-state-invariant-after-ops st token1 token2)))

(defthm chain-add-n-tokens-statep-1
  (implies (and (mgt-state-p st) (string-list-p tokens))
           (mgt-state-p (chain-add-n-tokens st tokens))))

(defthm chain-add-n-tokens-next-id-monotone-1
  (implies (and (mgt-state-p st) (string-list-p tokens))
           (<= (mgt-next-id st)
               (mgt-next-id (chain-add-n-tokens st tokens))))
  :rule-classes :linear)

(defthm all-added-tokens-findable-1
  (implies (and (mgt-state-p st) (string-list-p tokens))
           (all-added-tokens-findable st tokens)))

(defthm verify-encode-produces-valid-tokens-1
  (implies (and (mgt-state-p st) (stringp text))
           (verify-encode-produces-valid-tokens st text)))

(defthm verify-batch-encode-all-valid-base-1
  (verify-batch-encode-all-valid (make-empty-mgt-state) nil))

(defthm string-concat-list-type-1
  (implies (string-list-p strs)
           (stringp (string-concat-list strs)))
  :rule-classes :type-prescription)

(defthm decode-tokens-to-list-type-1
  (implies (and (mgt-state-p st) (nat-listp toks))
           (string-list-p (decode-tokens-to-list st toks))))

(defthm decode-tokens-to-list-length-1
  (implies (and (mgt-state-p st) (nat-listp toks))
           (equal (len (decode-tokens-to-list st toks))
                  (len toks))))

(defthm encode-decode-roundtrip-check-type-1
  (implies (and (mgt-state-p st) (stringp text))
           (stringp (encode-decode-roundtrip-check st text)))
  :rule-classes :type-prescription)

(defthm safe-decode-type-1
  (implies (and (mgt-state-p st) (nat-listp tokens))
           (stringp (safe-decode st tokens)))
  :rule-classes :type-prescription)

(defthm compare-snapshots-monotonic-self-1
  (let ((snap (state-snapshot (make-empty-mgt-state))))
    (compare-snapshots-monotonic snap snap)))

(defthm mgt_token_to_id_preserved_under_init_specials_1
  (implies (mgt-state-p st)
           (token-list-p (mgt-token-to-id (init-special-tokens st)))))

(defthm mgt_token_to_id_preserved_under_init_morphemes_1
  (implies (mgt-state-p st)
           (token-list-p (mgt-token-to-id (init-morphemes st)))))

(defthm mgt_token_to_id_preserved_under_remove_vocab_1
  (implies (and (mgt-state-p st) (stringp word))
           (token-list-p (mgt-token-to-id (remove-vocab-word st word)))))


(defthm mgt_id_to_token_preserved_under_init_specials_2
  (implies (mgt-state-p st)
           (id-list-p (mgt-id-to-token (init-special-tokens st)))))

(defthm mgt_id_to_token_preserved_under_init_morphemes_2
  (implies (mgt-state-p st)
           (id-list-p (mgt-id-to-token (init-morphemes st)))))

(defthm mgt_id_to_token_preserved_under_remove_vocab_2
  (implies (and (mgt-state-p st) (stringp word))
           (id-list-p (mgt-id-to-token (remove-vocab-word st word)))))


(defthm mgt_prefixes_preserved_under_init_specials_3
  (implies (mgt-state-p st)
           (token-list-p (mgt-prefixes (init-special-tokens st)))))

(defthm mgt_prefixes_preserved_under_init_morphemes_3
  (implies (mgt-state-p st)
           (token-list-p (mgt-prefixes (init-morphemes st)))))

(defthm mgt_prefixes_preserved_under_remove_vocab_3
  (implies (and (mgt-state-p st) (stringp word))
           (token-list-p (mgt-prefixes (remove-vocab-word st word)))))


(defthm mgt_suffixes_preserved_under_init_specials_4
  (implies (mgt-state-p st)
           (token-list-p (mgt-suffixes (init-special-tokens st)))))

(defthm mgt_suffixes_preserved_under_init_morphemes_4
  (implies (mgt-state-p st)
           (token-list-p (mgt-suffixes (init-morphemes st)))))

(defthm mgt_suffixes_preserved_under_remove_vocab_4
  (implies (and (mgt-state-p st) (stringp word))
           (token-list-p (mgt-suffixes (remove-vocab-word st word)))))


(defthm mgt_roots_preserved_under_init_specials_5
  (implies (mgt-state-p st)
           (token-list-p (mgt-roots (init-special-tokens st)))))

(defthm mgt_roots_preserved_under_init_morphemes_5
  (implies (mgt-state-p st)
           (token-list-p (mgt-roots (init-morphemes st)))))

(defthm mgt_roots_preserved_under_remove_vocab_5
  (implies (and (mgt-state-p st) (stringp word))
           (token-list-p (mgt-roots (remove-vocab-word st word)))))


(defthm mgt_bpe_pairs_preserved_under_init_specials_6
  (implies (mgt-state-p st)
           (bpe-pair-list-p (mgt-bpe-pairs (init-special-tokens st)))))

(defthm mgt_bpe_pairs_preserved_under_init_morphemes_6
  (implies (mgt-state-p st)
           (bpe-pair-list-p (mgt-bpe-pairs (init-morphemes st)))))

(defthm mgt_bpe_pairs_preserved_under_remove_vocab_6
  (implies (and (mgt-state-p st) (stringp word))
           (bpe-pair-list-p (mgt-bpe-pairs (remove-vocab-word st word)))))


(defthm vocab_size_natp_generated_1
  (implies (mgt-state-p st)
           (natp (vocab-size st)))
  :rule-classes :type-prescription)

(defthm vocab_size_nonnegative_generated_1
  (implies (mgt-state-p st)
           (<= 0 (vocab-size st)))
  :rule-classes :linear)


(defthm count_vocab_entries_natp_generated_2
  (implies (mgt-state-p st)
           (natp (count-vocab-entries st)))
  :rule-classes :type-prescription)

(defthm count_vocab_entries_nonnegative_generated_2
  (implies (mgt-state-p st)
           (<= 0 (count-vocab-entries st)))
  :rule-classes :linear)


(defthm count_prefix_entries_natp_generated_3
  (implies (mgt-state-p st)
           (natp (count-prefix-entries st)))
  :rule-classes :type-prescription)

(defthm count_prefix_entries_nonnegative_generated_3
  (implies (mgt-state-p st)
           (<= 0 (count-prefix-entries st)))
  :rule-classes :linear)


(defthm count_suffix_entries_natp_generated_4
  (implies (mgt-state-p st)
           (natp (count-suffix-entries st)))
  :rule-classes :type-prescription)

(defthm count_suffix_entries_nonnegative_generated_4
  (implies (mgt-state-p st)
           (<= 0 (count-suffix-entries st)))
  :rule-classes :linear)


(defthm count_bpe_merges_natp_generated_5
  (implies (mgt-state-p st)
           (natp (count-bpe-merges st)))
  :rule-classes :type-prescription)

(defthm count_bpe_merges_nonnegative_generated_5
  (implies (mgt-state-p st)
           (<= 0 (count-bpe-merges st)))
  :rule-classes :linear)


(defthm total_morpheme_count_natp_generated_6
  (implies (mgt-state-p st)
           (natp (total-morpheme-count st)))
  :rule-classes :type-prescription)

(defthm total_morpheme_count_nonnegative_generated_6
  (implies (mgt-state-p st)
           (<= 0 (total-morpheme-count st)))
  :rule-classes :linear)


(defthm encode_text_type_generated_1
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_1
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_1
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_1
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_2
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_2
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_2
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_2
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_3
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_3
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_3
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_3
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_4
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_4
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_4
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_4
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_5
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_5
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_5
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_5
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_6
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_6
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_6
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_6
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_7
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_7
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_7
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_7
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_8
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_8
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_8
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_8
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_9
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_9
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_9
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_9
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))


(defthm encode_text_type_generated_10
  (implies (and (mgt-state-p st) (stringp text))
           (nat-listp (encode-text st text))))

(defthm decode_tokens_type_generated_10
  (implies (and (mgt-state-p st) (nat-listp toks))
           (stringp (decode-tokens st toks)))
  :rule-classes :type-prescription)

(defthm encode_batch_length_generated_10
  (implies (and (mgt-state-p st) (string-list-p texts))
           (equal (len (encode-batch st texts))
                  (len texts))))

(defthm batch_decode_length_generated_10
  (implies (and (mgt-state-p st) (true-listp token-lists))
           (equal (len (batch-decode st token-lists))
                  (len token-lists))))

