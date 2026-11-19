/** Feedback from the plugin, from the Canvas's button or the send_feedback tool: the
 * report names the plugin's version and this computer's operating system, and hides the
 * open workspace's path. See @tensorviz/graph-host/feedback. */
import { feedbackReport, operatingSystem, type FeedbackInput, type FeedbackReport } from '@tensorviz/graph-host/feedback';

declare const TENSORVIZ_VERSION: string;

export function pluginFeedbackReport(input: FeedbackInput, source: FeedbackReport['source'], workspace: string | undefined): FeedbackReport {
  return feedbackReport(input, source, { product: 'plugin', version: TENSORVIZ_VERSION, os: operatingSystem() }, [workspace]);
}
