/** Shared graph-host services. Host adapters own editor APIs and permission UI. */
export { GraphOpeningController } from './graphOpening';
export type { GraphOpeningDependencies, GraphSetupPreparation } from './graphOpening';
export { TraceWorkerCore } from './traceWorkerCore';
export type { TraceWorkerHost, TraceRequest, TraceOutcome, WorkerEnvelope } from './traceWorkerCore';
export { readHostSnapshot, disposeSnapshotHelpers } from './hostSnapshot';
export { lookupProfile } from './profileValidation';
export { multifileProjectionBody, sha256Text, SOURCE_SET_LIMITS, ENUMERATION_ASK } from './workspaceSourceCore';
export type { SourceIO } from './workspaceSourceCore';
export { gitSnapshot, indexSnapshot, workingSnapshotCore, discoverReview, reviewCoverage } from './reviewSnapshotsCore';
export { projectReviewSnapshot, emptyReviewProjection, reviewAnalysisError } from './graphReviewCore';
