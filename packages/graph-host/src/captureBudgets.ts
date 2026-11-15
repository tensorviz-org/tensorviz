/** Recorder accounting excludes the surrounding worker graph, plan and evidence.
 * Keep their serialized envelope bounded without discarding a supported capture. */
export const CAPTURE_ENVELOPE_MAX_BYTES = 64 * 1024 * 1024;
/** The service response also contains the graph projection and display metadata. */
export const CAPTURE_SERVICE_MAX_BYTES = 128 * 1024 * 1024;
