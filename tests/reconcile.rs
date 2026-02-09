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
    assert_eq!(last_observation.get(&fp1), Some(&c1_id));

    let c2_id = *last_observation.get(&fp2).expect("should insert new id");
    assert!(solver.has_constraint(c2_id));
    assert_ne!(c1_id, c2_id);
}
