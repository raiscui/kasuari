use std::collections::BTreeMap;

use kasuari::WeightedRelation::*;
use kasuari::{
    semantic_constraints, ConstraintFingerprint, ConstraintId, SemanticReconcileResult, Solver,
    Strength, Variable,
};

#[test]
fn reconcile_should_keep_constraint_id_when_semantics_same_but_object_rebuilt() {
    // 同语义,但对象实例重建(指针身份不同)时,应该最大化复用旧 ConstraintId.
    let x = Variable::new();

    let old = x | EQ(Strength::MEDIUM) | 0.0;
    let rebuilt = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(old, rebuilt, "Constraint Eq 应该是 pointer identity");

    let fp = ConstraintFingerprint::new(&old);

    let mut solver = Solver::new();
    let old_id = solver.add_constraint(old.clone()).unwrap();

    // 上层状态: fingerprint -> ConstraintId(上一轮观测)
    let mut last_observation: BTreeMap<ConstraintFingerprint, ConstraintId> = BTreeMap::new();
    last_observation.insert(fp.clone(), old_id);

    // newest: 本轮输入全集(语义集合),包含“重建后的对象”
    let newest = semantic_constraints([&rebuilt]);

    let report: SemanticReconcileResult = solver
        .reconcile_semantic_constraints(&mut last_observation, &newest)
        .expect("reconcile should succeed");

    // 不应发生 remove/add,因此不算 update
    assert!(!report.did_update);
    assert!(
        report.removed_fingerprints.is_empty(),
        "semantics unchanged should not remove: removed={:?}",
        report.removed_fingerprints
    );
    assert!(
        report.added_fingerprints.is_empty(),
        "semantics unchanged should not add: added={:?}",
        report.added_fingerprints
    );
    assert!(
        report.repaired_fingerprints.is_empty(),
        "semantics unchanged should not repair: repaired={:?}",
        report.repaired_fingerprints
    );
    assert!(
        report.skipped_adds.is_empty(),
        "semantics unchanged should not skip adds: skipped={:?}",
        report.skipped_adds
    );
    assert_eq!(last_observation.get(&fp), Some(&old_id));
    assert!(solver.has_constraint(old_id));
}

#[test]
fn reconcile_should_remove_constraints_not_in_newest() {
    // newest 缺失某个 fingerprint 时,应从 solver 与 map 中移除对应 ConstraintId.
    let x = Variable::new();
    let y = Variable::new();

    let c1 = x | EQ(Strength::MEDIUM) | 0.0;
    let c2 = y | EQ(Strength::MEDIUM) | 0.0;

    let fp1 = ConstraintFingerprint::new(&c1);
    let fp2 = ConstraintFingerprint::new(&c2);

    let mut solver = Solver::new();
    let c1_id = solver.add_constraint(c1.clone()).unwrap();
    let c2_id = solver.add_constraint(c2.clone()).unwrap();

    let mut last_observation: BTreeMap<ConstraintFingerprint, ConstraintId> = BTreeMap::new();
    last_observation.insert(fp1.clone(), c1_id);
    last_observation.insert(fp2.clone(), c2_id);

    // newest 只保留 c1
    let newest = semantic_constraints([&c1]);
    let report = solver
        .reconcile_semantic_constraints(&mut last_observation, &newest)
        .expect("reconcile should succeed");

    assert!(report.did_update);
    assert_eq!(
        report.removed_fingerprints,
        vec![fp2.clone()],
        "should report removed fingerprint"
    );
    assert!(
        report.added_fingerprints.is_empty(),
        "remove-only reconcile should not add: added={:?}",
        report.added_fingerprints
    );
    assert!(
        report.repaired_fingerprints.is_empty(),
        "remove-only reconcile should not repair: repaired={:?}",
        report.repaired_fingerprints
    );
    assert!(solver.has_constraint(c1_id));
    assert!(!solver.has_constraint(c2_id));
    assert!(last_observation.contains_key(&fp1));
    assert!(!last_observation.contains_key(&fp2));
}

#[test]
fn reconcile_should_add_new_constraints_and_update_mapping() {
    // newest 增加新 fingerprint 时,应 add 并写回新的 ConstraintId.
    let x = Variable::new();
    let y = Variable::new();

    let c1 = x | EQ(Strength::MEDIUM) | 0.0;
    let c2 = y | EQ(Strength::MEDIUM) | 0.0;

    let fp1 = ConstraintFingerprint::new(&c1);
    let fp2 = ConstraintFingerprint::new(&c2);

    let mut solver = Solver::new();
    let c1_id = solver.add_constraint(c1.clone()).unwrap();

    let mut last_observation: BTreeMap<ConstraintFingerprint, ConstraintId> = BTreeMap::new();
    last_observation.insert(fp1.clone(), c1_id);

    // newest 包含 c1 与 c2
    let newest = semantic_constraints([&c1, &c2]);
    let report = solver
        .reconcile_semantic_constraints(&mut last_observation, &newest)
        .expect("reconcile should succeed");

    assert!(report.did_update);
    assert!(
        report.removed_fingerprints.is_empty(),
        "add-only reconcile should not remove: removed={:?}",
        report.removed_fingerprints
    );
    assert!(
        report.repaired_fingerprints.is_empty(),
        "add-only reconcile should not repair: repaired={:?}",
        report.repaired_fingerprints
    );
    assert_eq!(last_observation.get(&fp1), Some(&c1_id));

    let c2_id = *last_observation.get(&fp2).expect("should insert new id");
    assert!(solver.has_constraint(c2_id));
    assert_ne!(c1_id, c2_id);

    assert_eq!(
        report.added_fingerprints,
        vec![(fp2.clone(), c2_id)],
        "should report added fingerprint and id"
    );
}

#[test]
fn reconcile_should_repair_dirty_mapping_when_id_missing_in_solver() {
    ////////////////////////////////////////////////////////////////////////////////
    // 回归测试(repair 语义):
    //
    // - 上层 map 可能因为 bug/重置/跨帧状态错位,出现“fingerprint 存在但 id 已不在 solver”的脏数据。
    // - reconcile 应当:
    //   1) 通过 add 修复该项(并分配新 id)
    //   2) 用新 id 覆盖 map
    //   3) 在报告里把它记为 repaired(而不是普通 added)
    ////////////////////////////////////////////////////////////////////////////////
    let x = Variable::new();

    let c1 = x | EQ(Strength::MEDIUM) | 0.0;
    let fp1 = ConstraintFingerprint::new(&c1);

    let mut solver = Solver::new();
    let old_id = solver.add_constraint(c1.clone()).unwrap();

    // 人为制造脏数据:solver 里移除了约束,但 map 仍保留旧 id。
    solver
        .remove_constraint(old_id)
        .expect("manual removal should succeed");
    assert!(!solver.has_constraint(old_id));

    let mut last_observation: BTreeMap<ConstraintFingerprint, ConstraintId> = BTreeMap::new();
    last_observation.insert(fp1.clone(), old_id);

    // newest 仍然要求存在该 fingerprint,应触发 repair。
    let newest = semantic_constraints([&c1]);
    let report = solver
        .reconcile_semantic_constraints(&mut last_observation, &newest)
        .expect("reconcile should succeed");

    assert!(report.did_update);
    assert!(report.removed_fingerprints.is_empty());
    assert!(report.added_fingerprints.is_empty());
    assert_eq!(report.repaired_fingerprints.len(), 1);

    let (repaired_fp, new_id) = &report.repaired_fingerprints[0];
    assert_eq!(repaired_fp, &fp1);
    assert_ne!(*new_id, old_id, "repair should allocate a new id");
    assert!(solver.has_constraint(*new_id));
    assert_eq!(last_observation.get(&fp1), Some(new_id));
}
