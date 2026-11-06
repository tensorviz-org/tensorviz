// Expected public plugin tools, shared by discovery and client acceptance checks.
module.exports = [
  'list_models', 'get_graph', 'inspect_nodes', 'get_selection',
  'focus_view', 'open_graph_view', 'open_graph', 'get_execution_environment', 'list_captures', 'inspect_capture',
  'get_model_context', 'find_computation', 'query_computation', 'analyze_computation_change',
  'get_graph_setup', 'suggest_graph_inputs', 'configure_graph_setup', 'configure_graph_environment',
  'validate_model', 'get_run', 'inspect_run', 'cancel_run', 'compare_runs', 'open_graph_diff',
  'open_graph_panel', 'graph_panel_read', 'graph_panel_exchange', 'detach_graph_panel', 'close_graph_panel',
];
