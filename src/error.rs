use thiserror::Error;

use crate::InternalSolverError;

/// The possible error conditions that `Solver::add_constraint` can fail with.
#[derive(Debug, Copy, Clone, Error)]
pub enum AddConstraintError {
    /// The constraint specified has already been added to the solver.
    #[error("The constraint specified has already been added to the solver.")]
    DuplicateConstraint,

    /// The constraint is required, but it is unsatisfiable in conjunction with the existing
    /// constraints.
    #[error("The constraint is required, but it is unsatisfiable in conjunction with the existing constraints.")]
    UnsatisfiableConstraint,

    /// The solver entered an invalid state.
    #[error("The solver entered an invalid state. If this occurs please report the issue.")]
    InternalSolverError(#[from] InternalSolverError),
}

/// The possible error conditions that `Solver::remove_constraint` can fail with.
#[derive(Debug, Copy, Clone, Error)]
pub enum RemoveConstraintError {
    /// The constraint specified was not already in the solver, so cannot be removed.
    #[error("The constraint specified was not already in the solver, so cannot be removed.")]
    UnknownConstraint,

    /// The solver entered an invalid state. If this occurs please report the issue. This variant
    /// specifies additional details as a string.
    #[error("The solver entered an invalid state. If this occurs please report the issue.")]
    InternalSolverError(#[from] InternalSolverError),
}

/// The possible error conditions that `Solver::add_edit_variable` can fail with.
#[derive(Debug, Copy, Clone, Error)]
pub enum AddEditVariableError {
    /// The specified variable is already marked as an edit variable in the solver.
    #[error("The specified variable is already marked as an edit variable in the solver.")]
    DuplicateEditVariable,

    /// The specified strength was `REQUIRED`. This is illegal for edit variable strengths.
    #[error("The specified strength was `REQUIRED`. This is illegal for edit variable strengths.")]
    BadRequiredStrength,

    /// The constraint is unsatisfiable in conjunction with the existing constraints.
    ///
    /// 说明：
    /// - edit variable 的约束理论上不应触发该错误（因为 edit strength 禁止 REQUIRED）；
    /// - 但为了避免内部 unwrap/panic，这里保留该分支用于“防御式”传播错误。
    #[error("The constraint is unsatisfiable in conjunction with the existing constraints.")]
    UnsatisfiableConstraint,

    /// The solver entered an invalid state. If this occurs please report the issue.
    #[error("The solver entered an invalid state. If this occurs please report the issue.")]
    InternalSolverError(#[from] InternalSolverError),
}

/// The possible error conditions that `Solver::remove_edit_variable` can fail with.
#[derive(Debug, Copy, Clone, Error)]
pub enum RemoveEditVariableError {
    /// The specified variable was not an edit variable in the solver, so cannot be removed.
    #[error(
        "The specified variable was not an edit variable in the solver, so cannot be removed."
    )]
    UnknownEditVariable,

    /// The solver entered an invalid state. If this occurs please report the issue. This variant
    /// specifies additional details as a string.
    #[error("The solver entered an invalid state. If this occurs please report the issue.")]
    InternalSolverError(#[from] InternalSolverError),
}

/// The possible error conditions that `Solver::suggest_value` can fail with.
#[derive(Debug, Copy, Clone, Error)]
pub enum SuggestValueError {
    /// The specified variable was not an edit variable in the solver, so cannot have its value
    /// suggested.
    #[error(
        "The specified variable was not an edit variable in the solver, so cannot have its value suggested."
    )]
    UnknownEditVariable,

    /// The solver entered an invalid state. If this occurs please report the issue. This variant
    /// specifies additional details as a string.
    #[error("The solver entered an invalid state. If this occurs please report the issue.")]
    InternalSolverError(#[from] InternalSolverError),
}
