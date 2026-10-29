//! Pure, versioned contracts for Ryu's recursive-improvement loop.
//!
//! Learning and Research remain separate owners: Learning captures experience
//! and proposes reusable capabilities, while Research runs bounded candidate
//! comparisons. This crate only owns the vocabulary that lets both surfaces
//! point at the same improvement run without sharing storage or policy logic.
//!
//! The contract deliberately carries references and digests rather than raw
//! prompts, conversations, credentials, or evaluator transcripts. Core owns
//! execution, Gateway owns policy and measurement, and the owning app decides
//! whether a verified candidate may be promoted.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The public protocol version for improvement-run projections.
pub const IMPROVEMENT_PROTOCOL_VERSION: &str = "ryu.improvement.v1";
/// Maximum length of an opaque run or artifact id.
pub const MAX_ID_LEN: usize = 128;
/// Maximum length of the human-readable improvement objective.
pub const MAX_OBJECTIVE_LEN: usize = 4_000;

/// The thing whose behavior or quality is being improved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Skill,
    Prompt,
    MemoryRule,
    Workflow,
    Tool,
    App,
    Sidecar,
    ProviderRoute,
    ModelAdapter,
}

/// Monotonic promotion vocabulary shared by Learning and Research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImprovementStatus {
    Draft,
    Baseline,
    Candidate,
    Verified,
    Approved,
    Canary,
    Promoted,
    Rejected,
    RolledBack,
}

impl ImprovementStatus {
    /// Stable wire/storage spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Baseline => "baseline",
            Self::Candidate => "candidate",
            Self::Verified => "verified",
            Self::Approved => "approved",
            Self::Canary => "canary",
            Self::Promoted => "promoted",
            Self::Rejected => "rejected",
            Self::RolledBack => "rolled_back",
        }
    }

    /// Whether the status cannot change without creating a new run.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Rejected | Self::RolledBack)
    }
}

/// An explicit decision recorded separately from the lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionAction {
    Promote,
    Reject,
    Rollback,
}

/// A stable reference to an artifact version. The artifact body lives with its
/// owning app, package, workspace, or model registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactRef {
    pub kind: ArtifactKind,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

/// References the source that caused a run to exist. It is attribution, not a
/// permission decision.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementProvenance {
    /// `learning`, `research`, `manual`, or another registered source.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub experience_ids: Vec<String>,
}

/// One bounded comparison dimension. Raw cases remain in the evaluator store;
/// the improvement projection carries only the measured values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricDelta {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<f64>,
    pub higher_is_better: bool,
}

/// A bounded pointer to tests, evals, policy checks, or rendered proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRef {
    pub kind: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

/// Quality and safety checks required before a candidate is verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardrailSummary {
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<String>,
}

/// The measured part of an improvement run. It is absent until the candidate
/// has been evaluated.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationSummary {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<MetricDelta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_passed: Option<bool>,
}

/// A human- or policy-owned decision. The proposer must not silently become
/// the promoter by omitting this record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementDecision {
    pub action: DecisionAction,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    pub decided_at: String,
}

/// The small correlation projection embedded in Learning and Research records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementRunRef {
    pub protocol_version: String,
    pub id: String,
}

impl ImprovementRunRef {
    /// Create a reference after applying the shared opaque-id boundary.
    pub fn new(id: impl Into<String>) -> Result<Self, ContractError> {
        let id = id.into();
        validate_id("improvement run", &id)?;
        Ok(Self {
            protocol_version: IMPROVEMENT_PROTOCOL_VERSION.to_owned(),
            id,
        })
    }
}

/// Canonical durable record for a single bounded improvement attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementRun {
    pub protocol_version: String,
    pub id: String,
    pub objective: String,
    pub artifact_kind: ArtifactKind,
    pub baseline: ArtifactRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<ArtifactRef>,
    pub status: ImprovementStatus,
    pub provenance: ImprovementProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<EvaluationSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guardrails: Option<GuardrailSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<ImprovementDecision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback: Option<ArtifactRef>,
    pub created_at: String,
    pub updated_at: String,
}

impl ImprovementRun {
    /// Start a new run at the draft state. No candidate or decision is active.
    pub fn new(
        id: impl Into<String>,
        objective: impl Into<String>,
        artifact_kind: ArtifactKind,
        baseline: ArtifactRef,
        provenance: ImprovementProvenance,
        created_at: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let id = id.into();
        let objective = objective.into();
        validate_id("improvement run", &id)?;
        validate_text("objective", &objective, MAX_OBJECTIVE_LEN)?;
        validate_artifact(&baseline)?;
        let created_at = created_at.into();
        validate_text("createdAt", &created_at, 128)?;
        Ok(Self {
            protocol_version: IMPROVEMENT_PROTOCOL_VERSION.to_owned(),
            id,
            objective,
            artifact_kind,
            baseline,
            candidate: None,
            status: ImprovementStatus::Draft,
            provenance,
            evaluation: None,
            guardrails: None,
            decision: None,
            rollback: None,
            created_at: created_at.clone(),
            updated_at: created_at,
        })
    }

    /// Attach a candidate artifact before moving to `candidate`.
    pub fn set_candidate(&mut self, candidate: ArtifactRef) -> Result<(), ContractError> {
        validate_artifact(&candidate)?;
        if !matches!(
            self.status,
            ImprovementStatus::Draft | ImprovementStatus::Baseline | ImprovementStatus::Candidate
        ) {
            return Err(ContractError::InvalidState {
                status: self.status,
                operation: "set_candidate",
            });
        }
        self.candidate = Some(candidate);
        Ok(())
    }

    /// Record bounded evaluation evidence before verification.
    pub fn set_evaluation(
        &mut self,
        evaluation: EvaluationSummary,
        guardrails: GuardrailSummary,
    ) -> Result<(), ContractError> {
        if self.candidate.is_none() || self.status != ImprovementStatus::Candidate {
            return Err(ContractError::InvalidState {
                status: self.status,
                operation: "set_evaluation",
            });
        }
        self.evaluation = Some(evaluation);
        self.guardrails = Some(guardrails);
        Ok(())
    }

    /// Record a promotion, rejection, or rollback decision.
    pub fn set_decision(&mut self, decision: ImprovementDecision) -> Result<(), ContractError> {
        validate_text("decision.reason", &decision.reason, MAX_OBJECTIVE_LEN)?;
        if self.status.is_terminal() {
            return Err(ContractError::InvalidState {
                status: self.status,
                operation: "set_decision",
            });
        }
        self.decision = Some(decision);
        Ok(())
    }

    /// Attach the last known-good artifact used for rollback.
    pub fn set_rollback(&mut self, rollback: ArtifactRef) -> Result<(), ContractError> {
        validate_artifact(&rollback)?;
        if !matches!(
            self.status,
            ImprovementStatus::Approved | ImprovementStatus::Canary | ImprovementStatus::Promoted
        ) {
            return Err(ContractError::InvalidState {
                status: self.status,
                operation: "set_rollback",
            });
        }
        self.rollback = Some(rollback);
        Ok(())
    }

    /// Apply one lifecycle transition. Invalid or skipped transitions fail
    /// closed, while repeating the current state is idempotent.
    pub fn transition_to(&mut self, next: ImprovementStatus) -> Result<(), ContractError> {
        if self.status == next {
            return Ok(());
        }
        if !allowed_transition(self.status, next) {
            return Err(ContractError::Transition {
                from: self.status,
                to: next,
            });
        }
        match next {
            ImprovementStatus::Baseline => {}
            ImprovementStatus::Candidate => {
                if self.candidate.is_none() {
                    return Err(ContractError::Missing("candidate"));
                }
            }
            ImprovementStatus::Verified => {
                let Some(evaluation) = self.evaluation.as_ref() else {
                    return Err(ContractError::Missing("evaluation"));
                };
                let Some(guardrails) = self.guardrails.as_ref() else {
                    return Err(ContractError::Missing("guardrails"));
                };
                if evaluation.metrics.is_empty() || evaluation.evidence.is_empty() {
                    return Err(ContractError::Missing("evaluation evidence"));
                }
                if !guardrails.passed {
                    return Err(ContractError::GuardrailsFailed);
                }
            }
            ImprovementStatus::Approved | ImprovementStatus::Canary => {
                if self.candidate.is_none() {
                    return Err(ContractError::Missing("candidate"));
                }
            }
            ImprovementStatus::Promoted => {
                if self.rollback.is_none() {
                    return Err(ContractError::Missing("rollback"));
                }
                if !has_decision(self, DecisionAction::Promote) {
                    return Err(ContractError::Missing("promote decision"));
                }
            }
            ImprovementStatus::Rejected => {
                if !has_decision(self, DecisionAction::Reject) {
                    return Err(ContractError::Missing("reject decision"));
                }
            }
            ImprovementStatus::RolledBack => {
                if self.rollback.is_none() || !has_decision(self, DecisionAction::Rollback) {
                    return Err(ContractError::Missing("rollback decision"));
                }
            }
            ImprovementStatus::Draft => unreachable!("draft has no inbound transition"),
        }
        self.status = next;
        Ok(())
    }

    /// Validate all invariants, including the invariants implied by the
    /// current lifecycle state. Consumers should call this after deserializing
    /// an untrusted projection.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.protocol_version != IMPROVEMENT_PROTOCOL_VERSION {
            return Err(ContractError::ProtocolVersion(
                self.protocol_version.clone(),
            ));
        }
        validate_id("improvement run", &self.id)?;
        validate_text("objective", &self.objective, MAX_OBJECTIVE_LEN)?;
        validate_artifact(&self.baseline)?;
        if let Some(candidate) = &self.candidate {
            validate_artifact(candidate)?;
        }
        if let Some(rollback) = &self.rollback {
            validate_artifact(rollback)?;
        }
        if matches!(
            self.status,
            ImprovementStatus::Candidate
                | ImprovementStatus::Verified
                | ImprovementStatus::Approved
                | ImprovementStatus::Canary
                | ImprovementStatus::Promoted
        ) && self.candidate.is_none()
        {
            return Err(ContractError::Missing("candidate"));
        }
        if matches!(
            self.status,
            ImprovementStatus::Verified
                | ImprovementStatus::Approved
                | ImprovementStatus::Canary
                | ImprovementStatus::Promoted
        ) {
            let Some(evaluation) = self.evaluation.as_ref() else {
                return Err(ContractError::Missing("evaluation"));
            };
            let Some(guardrails) = self.guardrails.as_ref() else {
                return Err(ContractError::Missing("guardrails"));
            };
            if evaluation.metrics.is_empty() || evaluation.evidence.is_empty() {
                return Err(ContractError::Missing("evaluation evidence"));
            }
            if !guardrails.passed {
                return Err(ContractError::GuardrailsFailed);
            }
        }
        if matches!(
            self.status,
            ImprovementStatus::Approved | ImprovementStatus::Canary | ImprovementStatus::Promoted
        ) && !has_decision(self, DecisionAction::Promote)
        {
            return Err(ContractError::Missing("promote decision"));
        }
        if self.status == ImprovementStatus::Rejected && !has_decision(self, DecisionAction::Reject)
        {
            return Err(ContractError::Missing("reject decision"));
        }
        if self.status == ImprovementStatus::Promoted {
            if self.rollback.is_none() || !has_decision(self, DecisionAction::Promote) {
                return Err(ContractError::Missing("promotion rollback/decision"));
            }
        }
        if self.status == ImprovementStatus::RolledBack
            && (self.rollback.is_none() || !has_decision(self, DecisionAction::Rollback))
        {
            return Err(ContractError::Missing("rollback decision"));
        }
        Ok(())
    }
}

fn has_decision(run: &ImprovementRun, action: DecisionAction) -> bool {
    run.decision
        .as_ref()
        .is_some_and(|decision| decision.action == action)
}

fn allowed_transition(from: ImprovementStatus, to: ImprovementStatus) -> bool {
    matches!(
        (from, to),
        (ImprovementStatus::Draft, ImprovementStatus::Baseline)
            | (ImprovementStatus::Baseline, ImprovementStatus::Candidate)
            | (ImprovementStatus::Candidate, ImprovementStatus::Verified)
            | (ImprovementStatus::Candidate, ImprovementStatus::Rejected)
            | (ImprovementStatus::Verified, ImprovementStatus::Approved)
            | (ImprovementStatus::Verified, ImprovementStatus::Rejected)
            | (ImprovementStatus::Approved, ImprovementStatus::Canary)
            | (ImprovementStatus::Approved, ImprovementStatus::Rejected)
            | (ImprovementStatus::Canary, ImprovementStatus::Promoted)
            | (ImprovementStatus::Canary, ImprovementStatus::Rejected)
            | (ImprovementStatus::Promoted, ImprovementStatus::RolledBack)
    )
}

fn validate_artifact(artifact: &ArtifactRef) -> Result<(), ContractError> {
    validate_id("artifact", &artifact.id)
}

fn validate_id(label: &'static str, value: &str) -> Result<(), ContractError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
    {
        return Err(ContractError::InvalidId {
            label,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_text(label: &'static str, value: &str, max_len: usize) -> Result<(), ContractError> {
    if value.trim().is_empty() || value.chars().count() > max_len {
        return Err(ContractError::InvalidText { label, max_len });
    }
    Ok(())
}

/// Contract validation and lifecycle failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    InvalidId {
        label: &'static str,
        value: String,
    },
    InvalidText {
        label: &'static str,
        max_len: usize,
    },
    ProtocolVersion(String),
    InvalidState {
        status: ImprovementStatus,
        operation: &'static str,
    },
    Transition {
        from: ImprovementStatus,
        to: ImprovementStatus,
    },
    Missing(&'static str),
    GuardrailsFailed,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId { label, value } => write!(formatter, "invalid {label} id: {value}"),
            Self::InvalidText { label, max_len } => {
                write!(
                    formatter,
                    "invalid {label}; must be non-empty and at most {max_len} characters"
                )
            }
            Self::ProtocolVersion(version) => write!(
                formatter,
                "unsupported improvement protocol version: {version}"
            ),
            Self::InvalidState { status, operation } => {
                write!(
                    formatter,
                    "cannot {operation} while improvement run is {status:?}"
                )
            }
            Self::Transition { from, to } => {
                write!(
                    formatter,
                    "invalid improvement transition {from:?} -> {to:?}"
                )
            }
            Self::Missing(field) => write!(formatter, "missing improvement field: {field}"),
            Self::GuardrailsFailed => write!(formatter, "improvement guardrails did not pass"),
        }
    }
}

impl std::error::Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> ArtifactRef {
        ArtifactRef {
            kind: ArtifactKind::Skill,
            id: "skill-baseline".to_owned(),
            version: Some("1.0.0".to_owned()),
            digest: Some("sha256-baseline".to_owned()),
        }
    }

    fn candidate() -> ArtifactRef {
        ArtifactRef {
            kind: ArtifactKind::Skill,
            id: "skill-candidate".to_owned(),
            version: Some("1.1.0".to_owned()),
            digest: Some("sha256-candidate".to_owned()),
        }
    }

    fn run() -> ImprovementRun {
        ImprovementRun::new(
            "improvement-1",
            "Reduce corrections on app documentation",
            ArtifactKind::Skill,
            baseline(),
            ImprovementProvenance {
                source: "learning".to_owned(),
                ..Default::default()
            },
            "2026-09-11T00:00:00Z",
        )
        .unwrap()
    }

    #[test]
    fn wire_contract_uses_versioned_camel_case_fields() {
        let reference = ImprovementRunRef::new("improvement-1").unwrap();
        let value = serde_json::to_value(reference).unwrap();
        assert_eq!(value["protocolVersion"], IMPROVEMENT_PROTOCOL_VERSION);
        assert_eq!(value["id"], "improvement-1");
        assert_eq!(ImprovementStatus::RolledBack.as_str(), "rolled_back");
    }

    #[test]
    fn lifecycle_requires_evidence_and_rollback() {
        let mut run = run();
        run.transition_to(ImprovementStatus::Baseline).unwrap();
        run.set_candidate(candidate()).unwrap();
        run.transition_to(ImprovementStatus::Candidate).unwrap();
        assert_eq!(
            run.transition_to(ImprovementStatus::Verified),
            Err(ContractError::Missing("evaluation"))
        );
        run.set_evaluation(
            EvaluationSummary {
                metrics: vec![MetricDelta {
                    name: "task_completion".to_owned(),
                    baseline: Some(0.5),
                    candidate: Some(0.8),
                    higher_is_better: true,
                }],
                evidence: vec![EvidenceRef {
                    kind: "eval".to_owned(),
                    id: "eval-1".to_owned(),
                    digest: None,
                }],
                holdout_passed: Some(true),
            },
            GuardrailSummary {
                passed: true,
                failures: vec![],
            },
        )
        .unwrap();
        run.transition_to(ImprovementStatus::Verified).unwrap();
        run.set_decision(ImprovementDecision {
            action: DecisionAction::Promote,
            reason: "holdout improved without a guardrail regression".to_owned(),
            actor_id: Some("owner-1".to_owned()),
            decided_at: "2026-09-11T00:01:00Z".to_owned(),
        })
        .unwrap();
        run.transition_to(ImprovementStatus::Approved).unwrap();
        assert_eq!(
            run.transition_to(ImprovementStatus::Promoted),
            Err(ContractError::Transition {
                from: ImprovementStatus::Approved,
                to: ImprovementStatus::Promoted,
            })
        );
        run.set_rollback(baseline()).unwrap();
        run.transition_to(ImprovementStatus::Canary).unwrap();
        run.transition_to(ImprovementStatus::Promoted).unwrap();
        run.set_decision(ImprovementDecision {
            action: DecisionAction::Rollback,
            reason: "holdout regression detected".to_owned(),
            actor_id: Some("owner-1".to_owned()),
            decided_at: "2026-09-11T00:02:00Z".to_owned(),
        })
        .unwrap();
        run.transition_to(ImprovementStatus::RolledBack).unwrap();
        run.validate().unwrap();
    }

    #[test]
    fn invalid_ids_fail_closed() {
        assert!(ImprovementRunRef::new("../escape").is_err());
        assert!(ImprovementRunRef::new("").is_err());
        assert!(ImprovementRunRef::new("a/b").is_err());
    }
}
