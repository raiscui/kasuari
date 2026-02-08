use kasuari::WeightedRelation::*;
use kasuari::{
    merge_semantic_constraints, semantic_constraints, semantic_constraints_with_mode,
    ConstraintFingerprint, DuplicateFingerprintMode, Strength, Variable,
};

#[test]
fn constraint_fingerprint_should_be_semantic_only() {
    let x = Variable::new();

    // 同语义,不同对象(指针身份不同)
    let c0_a = x | EQ(Strength::MEDIUM) | 0.0;
    let c0_b = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(c0_a, c0_b, "Constraint Eq should be pointer identity");

    assert_eq!(
        ConstraintFingerprint::new(&c0_a),
        ConstraintFingerprint::new(&c0_b),
        "ConstraintFingerprint should ignore pointer identity"
    );
}

#[test]
fn semantic_constraints_should_dedup_by_fingerprint_keep_first() {
    let x = Variable::new();

    let first = x | EQ(Strength::MEDIUM) | 0.0;
    let second = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(first, second);

    let set = semantic_constraints([&first, &second]);
    assert_eq!(set.len(), 1);

    let fp = ConstraintFingerprint::new(&first);
    let kept = set.get(&fp).expect("should keep the fingerprint entry");
    assert_eq!(kept, &first, "should keep first constraint object");
    assert_ne!(kept, &second, "should not replace with later duplicate");
}

#[test]
fn semantic_constraints_error_mode_should_report_duplicate() {
    let x = Variable::new();

    let c0_a = x | EQ(Strength::MEDIUM) | 0.0;
    let c0_b = x | EQ(Strength::MEDIUM) | 0.0;
    let fp = ConstraintFingerprint::new(&c0_a);

    let err = semantic_constraints_with_mode([&c0_a, &c0_b], DuplicateFingerprintMode::Error)
        .expect_err("should error on duplicate fingerprints");

    assert_eq!(err.fingerprint, fp);
    assert_eq!(err.count, 2);
}

#[test]
fn merge_semantic_constraints_should_reuse_previous_object() {
    let x = Variable::new();

    let old = x | EQ(Strength::MEDIUM) | 0.0;
    let new = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(old, new);

    let previous = semantic_constraints([&old]);
    let newest = semantic_constraints([&new]);

    let merged = merge_semantic_constraints(&previous, &newest);

    let fp = ConstraintFingerprint::new(&old);
    let merged_c = merged.get(&fp).expect("merged should contain fingerprint");

    assert_eq!(
        merged_c, &old,
        "merged should reuse previous constraint object"
    );
    assert_ne!(merged_c, &new, "merged should not use newly created object");
}
