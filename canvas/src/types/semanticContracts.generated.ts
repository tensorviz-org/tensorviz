/* GENERATED from backend/services/semantic/contracts.py; npm run semantic:generate from the repository root. */
export type paths = Record<string, never>;
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** ActivationEditCapability */
        ActivationEditCapability: {
            binding: components["schemas"]["SourceBinding"];
            /**
             * Command
             * @constant
             */
            command: "replace_activation";
            /** Compatibleactivations */
            compatibleActivations?: ("relu" | "gelu" | "silu" | "sigmoid" | "tanh")[];
            /**
             * Expected
             * @enum {string}
             */
            expected: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            /** @default null */
            importPlan?: components["schemas"]["ActivationImportPlan"] | null;
            /** Namespace */
            namespace: string;
            options?: components["schemas"]["ActivationSourceOptions"];
            /**
             * Recipe
             * @enum {string}
             */
            recipe: "functional_activation_v1" | "module_activation_v1";
            scope: components["schemas"]["SourceEditScope"];
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ActivationImportPlan */
        ActivationImportPlan: {
            /** Line */
            line: number;
            /**
             * Lineending
             * @enum {string}
             */
            lineEnding: "\n" | "\r\n";
            /**
             * Module
             * @enum {string}
             */
            module: "torch.nn" | "torch.nn.functional";
        };
        /** ActivationSourceOptions */
        ActivationSourceOptions: {
            /**
             * Approximate
             * @default null
             */
            approximate?: ("none" | "tanh") | null;
            /**
             * Inplace
             * @default null
             */
            inplace?: boolean | null;
        };
        /** Arguments */
        Arguments: {
            /** Args */
            args: (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"])[];
            /** Kwargs */
            kwargs: {
                [key: string]: (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"]) | (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"]);
            };
        };
        /**
         * AutomaticCaptureRequest
         * @description Existing automatic/declared-input request, not a resolved input recipe.
         */
        AutomaticCaptureRequest: {
            /** Input */
            input: string | null;
            /** Overrides */
            overrides: {
                [key: string]: components["schemas"]["JsonValue"];
            } | null;
            /** Tensorinputs */
            tensorInputs: {
                [key: string]: components["schemas"]["JsonValue"];
            }[] | null;
        };
        /** AutomaticCaptureSelection */
        AutomaticCaptureSelection: {
            /** Candidateindex */
            candidateIndex: number;
            /** Candidatestried */
            candidatesTried: number;
            /**
             * Provenance
             * @enum {string}
             */
            provenance: "explicit" | "declared" | "seed-from-first-layer" | "search-candidate";
        };
        /** BoolLiteral */
        BoolLiteral: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "bool";
            /** Value */
            value: boolean;
        };
        /**
         * CandidateExecution
         * @description Private replay provenance; never a replacement for the live model capture.
         */
        CandidateExecution: {
            /** Afterdigest */
            afterDigest: string;
            /** Baseattemptid */
            baseAttemptId: string;
            /** Basecaptureid */
            baseCaptureId: string;
            /** Baseplandigest */
            basePlanDigest: string;
            /** Basesourceid */
            baseSourceId: string;
            /** Baseworkspaceid */
            baseWorkspaceId: string;
            /** Beforedigest */
            beforeDigest: string;
            /** File */
            file: string;
            /** Id */
            id: string;
            /** Intentdigest */
            intentDigest: string;
            plan: components["schemas"]["ResolvedExecutionPlan"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /**
         * CandidatePostconditions
         * @description Evidence for this captured expression only, never program equivalence.
         */
        CandidatePostconditions: {
            /** Bindingid */
            bindingId: string | null;
            /** Captureid */
            captureId: string | null;
            /**
             * Configurationuses
             * @default []
             */
            configurationUses?: components["schemas"]["ConfigurationArgumentUse"][];
            /** Executionid */
            executionId: string;
            /**
             * Observedinitializers
             * @default 0
             */
            observedInitializers?: number;
            /** Observedoperations */
            observedOperations: number;
            /**
             * Reason
             * @enum {string}
             */
            reason: "observed_requested_literal" | "observed_literal_mismatch" | "observed_requested_activation" | "observed_activation_mismatch" | "unsupported_recipe" | "observed_requested_argument" | "observed_argument_mismatch" | "observed_requested_insertion" | "observed_insertion_mismatch" | "observed_requested_deletion" | "observed_deletion_mismatch" | "observed_requested_constructor" | "observed_constructor_mismatch" | "initializer_arguments_unavailable" | "observed_requested_configuration" | "observed_configuration_mismatch" | "configuration_reference_unavailable" | "capture_unavailable" | "capture_incomplete" | "capture_identity_mismatch" | "source_mapping_unavailable" | "expression_not_observed" | "ambiguous_observation";
            /** Recipe */
            recipe: ("binary_scalar_v1" | "functional_activation_v1" | "module_activation_v1" | "straight_line_tensor_v1" | "straight_line_unary_insert_v1" | "straight_line_unary_delete_v1" | "constructor_scalar_v1" | "configuration_scalar_v1") | null;
            /**
             * Scope
             * @enum {string}
             */
            scope: "observed_expression" | "observed_initializer_entries" | "observed_configuration_arguments";
            /**
             * Status
             * @enum {string}
             */
            status: "verified" | "failed" | "unavailable";
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /**
         * CandidateRequest
         * @description Private host payload; a closed recipe plus the full reviewed artifact.
         */
        CandidateRequest: {
            capture: components["schemas"]["CaptureSnapshot"];
            /** Id */
            id: string;
            /** Intent */
            intent: components["schemas"]["SetLiteralEdit"] | components["schemas"]["ReplaceActivationEdit"] | components["schemas"]["SetConstructorLiteralEdit"] | components["schemas"]["SetConfigurationLiteralEdit"] | components["schemas"]["ReconnectArgumentEdit"] | components["schemas"]["InsertUnaryEdit"] | components["schemas"]["DeleteUnaryEdit"];
            plan: components["schemas"]["ResolvedExecutionPlan"];
            /** Reviewedfile */
            reviewedFile: string;
            /** Reviewedsource */
            reviewedSource: string;
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** CaptureCoverage */
        CaptureCoverage: {
            /** Invocationsseen */
            invocationsSeen: number;
            /** Modules */
            modules: components["schemas"]["CoverageEntry"][];
            /** Omissions */
            omissions: ("operation_limit" | "invocation_limit" | "value_limit" | "byte_limit" | "depth_limit" | "item_limit" | "unsupported_type" | "opaque_kernel" | "recorder_error" | "timeout" | "cancelled")[];
            /** Operationsseen */
            operationsSeen: number;
            /**
             * Scope
             * @enum {string}
             */
            scope: "executed_path" | "traced_program";
        };
        /** CaptureLimits */
        CaptureLimits: {
            /** Bytes */
            bytes: number;
            /** Invocations */
            invocations: number;
            /** Operations */
            operations: number;
            /** Values */
            values: number;
        };
        /**
         * CapturePreparation
         * @description Saved facts before automatic inputs resolve; cannot impersonate a run plan.
         */
        CapturePreparation: {
            captureLimits: components["schemas"]["CaptureLimits"];
            /**
             * Configuration
             * @enum {string}
             */
            configuration: "absent" | "no_match";
            environment: components["schemas"]["EnvironmentFact"];
            /** Id */
            id: string;
            interpreter: components["schemas"]["InterpreterIdentity"];
            /**
             * Policy
             * @constant
             */
            policy: "legacy-input-search-v1";
            request: components["schemas"]["AutomaticCaptureRequest"];
            source: components["schemas"]["SourceSnapshot"];
            /** Timeoutms */
            timeoutMs: number;
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** CaptureProjection */
        CaptureProjection: {
            /** Captureid */
            captureId: string;
            /** Collapsedinvocationids */
            collapsedInvocationIds: string[];
            /** Edges */
            edges: components["schemas"]["ProjectionEdge"][];
            /** Nodes */
            nodes: components["schemas"]["ProjectionNode"][];
            /** Ports */
            ports: components["schemas"]["ProjectionPort"][];
            /** Sourceid */
            sourceId: string;
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /**
         * CaptureReconciliation
         * @description Observed source round trip, never freshness certification of an editor buffer.
         */
        CaptureReconciliation: {
            /** Basecaptureid */
            baseCaptureId: string;
            /** Captureid */
            captureId: string;
            /**
             * Dependencyscope
             * @enum {string}
             */
            dependencyScope: "complete" | "partial";
            /**
             * Inputs
             * @enum {string}
             */
            inputs: "same" | "different" | "unavailable";
            postconditions: components["schemas"]["CandidatePostconditions"] | null;
            /**
             * Reason
             * @enum {string}
             */
            reason: "reviewed_source" | "source_reverted" | "source_changed" | "dependencies_changed" | "environment_changed" | "different_model" | "capture_not_new" | "source_unavailable";
            /** Revieweddigest */
            reviewedDigest: string;
            /** Sourceid */
            sourceId: string;
            /**
             * State
             * @enum {string}
             */
            state: "same" | "different" | "unavailable";
            /**
             * Status
             * @enum {string}
             */
            status: "checked" | "unavailable";
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /**
         * CaptureReconciliationRequest
         * @description Private static check of a later live capture against the approved edit.
         */
        CaptureReconciliationRequest: {
            base: components["schemas"]["CaptureSnapshot"];
            current: components["schemas"]["CaptureSnapshot"];
            /** Intent */
            intent: components["schemas"]["SetLiteralEdit"] | components["schemas"]["ReplaceActivationEdit"] | components["schemas"]["SetConstructorLiteralEdit"] | components["schemas"]["SetConfigurationLiteralEdit"] | components["schemas"]["ReconnectArgumentEdit"] | components["schemas"]["InsertUnaryEdit"] | components["schemas"]["DeleteUnaryEdit"];
            /** Reviewedfile */
            reviewedFile: string;
            /** Reviewedsource */
            reviewedSource: string;
            /** Sources */
            sources: {
                [key: string]: string;
            };
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** CaptureReference */
        CaptureReference: {
            /** Captureid */
            captureId: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "operation" | "invocation" | "input" | "output" | "value";
            /** Semanticid */
            semanticId: string;
            /** Sourceid */
            sourceId: string;
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** CaptureSnapshot */
        CaptureSnapshot: {
            /** Attemptid */
            attemptId: string;
            automatic: components["schemas"]["AutomaticCaptureSelection"] | null;
            /**
             * Backend
             * @enum {string}
             */
            backend: "runtime" | "export" | "jaxpr";
            /** Backendversion */
            backendVersion: string;
            coverage: components["schemas"]["CaptureCoverage"];
            /** Definitions */
            definitions: components["schemas"]["ModuleDefinition"][];
            /** Diagnostics */
            diagnostics: components["schemas"]["Diagnostic"][];
            /** Id */
            id: string;
            /** Inputdigest */
            inputDigest: string | null;
            inputs: components["schemas"]["Arguments"];
            /** Invocations */
            invocations: components["schemas"]["Invocation"][];
            /**
             * Jaxversion
             * @default null
             */
            jaxVersion?: string | null;
            limits: components["schemas"]["CaptureLimits"];
            /** Modules */
            modules: components["schemas"]["ModuleInstance"][];
            /** Normalizations */
            normalizations: string[];
            /** Operations */
            operations: components["schemas"]["Operation"][];
            /** Output */
            output: (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"]) | null;
            /** Plandigest */
            planDigest: string | null;
            /** Preparationdigest */
            preparationDigest: string | null;
            /** Rootinvocationid */
            rootInvocationId: string | null;
            source: components["schemas"]["SourceSnapshot"];
            /** Sourcebindings */
            sourceBindings: components["schemas"]["SourceBinding"][];
            /** Statedigest */
            stateDigest: string | null;
            /**
             * Status
             * @enum {string}
             */
            status: "complete" | "partial" | "failed";
            /** Torchversion */
            torchVersion: string | null;
            /** Values */
            values: components["schemas"]["Value"][];
            /** Valuesunavailable */
            valuesUnavailable?: "fake_tensor" | null;
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** CaptureState */
        CaptureState: {
            /** Displayedcaptureid */
            displayedCaptureId: string | null;
            /** Lastcompletecaptureid */
            lastCompleteCaptureId: string | null;
            /** Lastpartialcaptureid */
            lastPartialCaptureId: string | null;
            latestAttempt: components["schemas"]["RunAttempt"] | null;
        };
        /** ConfigurationArgumentUse */
        ConfigurationArgumentUse: {
            /** Bindingid */
            bindingId: string;
            /** Sourceargumentpath */
            sourceArgumentPath: (number | string)[];
        };
        /** ConfigurationEditScope */
        ConfigurationEditScope: {
            /**
             * Affects
             * @constant
             */
            affects: "all_references";
            /**
             * Includesunobservedreferences
             * @constant
             */
            includesUnobservedReferences: true;
            /**
             * Kind
             * @constant
             */
            kind: "configuration_definition";
            /** Knownsourcereferences */
            knownSourceReferences: number;
            /** Observedconstructorsites */
            observedConstructorSites: number;
            /** Observedinvocations */
            observedInvocations: number;
            /** Observedmoduleinstances */
            observedModuleInstances: number;
            /** Observedoperations */
            observedOperations: number;
        };
        /** ConfigurationLiteralCapability */
        ConfigurationLiteralCapability: {
            binding: components["schemas"]["SourceBinding"];
            /**
             * Command
             * @constant
             */
            command: "set_configuration_literal";
            /** Expected */
            expected: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            /**
             * Recipe
             * @constant
             */
            recipe: "configuration_scalar_v1";
            reference: components["schemas"]["SourceBinding"];
            scope: components["schemas"]["ConfigurationEditScope"];
            /** Sourceargumentpath */
            sourceArgumentPath: (number | string)[];
            /** Symbol */
            symbol: string;
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ConstructorEditScope */
        ConstructorEditScope: {
            /**
             * Affects
             * @constant
             */
            affects: "all_executions";
            /**
             * Includesunobservedexecutions
             * @constant
             */
            includesUnobservedExecutions: true;
            /**
             * Kind
             * @constant
             */
            kind: "source_expression";
            /** Observedinvocations */
            observedInvocations: number;
            /** Observedmoduleinstances */
            observedModuleInstances: number;
            /** Observedoperations */
            observedOperations: number;
        };
        /** ConstructorLiteralCapability */
        ConstructorLiteralCapability: {
            binding: components["schemas"]["SourceBinding"];
            /**
             * Command
             * @constant
             */
            command: "set_constructor_literal";
            /** Constructor */
            constructor: string;
            /** Expected */
            expected: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            /**
             * Recipe
             * @constant
             */
            recipe: "constructor_scalar_v1";
            scope: components["schemas"]["ConstructorEditScope"];
            /** Sourceargumentpath */
            sourceArgumentPath: (number | string)[];
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** CoverageEntry */
        CoverageEntry: {
            /** Invocationids */
            invocationIds: string[];
            /** Modulepath */
            modulePath: string;
            /** Reason */
            reason: ("operation_limit" | "invocation_limit" | "value_limit" | "byte_limit" | "depth_limit" | "item_limit" | "unsupported_type" | "opaque_kernel" | "recorder_error" | "timeout" | "cancelled") | null;
            /**
             * Status
             * @enum {string}
             */
            status: "observed" | "failed" | "not_observed" | "capture_limited" | "static_only" | "unresolved" | "state";
        };
        /** DeleteUnaryCapability */
        DeleteUnaryCapability: {
            /**
             * Activation
             * @enum {string}
             */
            activation: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
            /** Argumentpath */
            argumentPath: (number | string)[];
            binding: components["schemas"]["SourceBinding"];
            /**
             * Command
             * @constant
             */
            command: "delete_unary_op";
            expected: components["schemas"]["ValueRef"];
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            input: components["schemas"]["SourceValueChoice"];
            /**
             * Recipe
             * @constant
             */
            recipe: "straight_line_unary_delete_v1";
            scope: components["schemas"]["SourceEditScope"];
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** DeleteUnaryEdit */
        DeleteUnaryEdit: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            /** Capabilityid */
            capabilityId: string;
            expected: components["schemas"]["ValueRef"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "delete_unary_op";
            /**
             * Scope
             * @constant
             */
            scope: "source_expression";
            target: components["schemas"]["CaptureReference"];
            value: components["schemas"]["ValueRef"];
        };
        /** Diagnostic */
        Diagnostic: {
            /**
             * Actualdtype
             * @default null
             */
            actualDtype?: string | null;
            /**
             * Actualshape
             * @default null
             */
            actualShape?: number[] | null;
            /**
             * Code
             * @enum {string}
             */
            code: "invalid_source" | "unresolved_import" | "invalid_config" | "unsaved" | "missing_file" | "missing_dependency" | "unavailable_environment" | "unavailable_device" | "missing_argument" | "unexpected_argument" | "invalid_binding" | "unknown_signature" | "invalid_factory" | "invalid_shape" | "invalid_dtype" | "assertion_failed" | "execution_failed" | "capture_limited" | "timeout" | "cancelled" | "stale_result" | "projection_failed" | "serialization_failed" | "transport_failed";
            /**
             * Expecteddtype
             * @default null
             */
            expectedDtype?: string | null;
            /**
             * Expectedshape
             * @default null
             */
            expectedShape?: (number | null)[] | null;
            /** Field */
            field: (number | string)[];
            /** Id */
            id: string;
            /** Names */
            names: string[];
            /**
             * Phase
             * @enum {string}
             */
            phase: "parse" | "index" | "configuration" | "environment" | "import" | "construction" | "binding" | "materialization" | "forward" | "capture" | "projection" | "serialization" | "transport";
            /**
             * Remedy
             * @enum {string}
             */
            remedy: "fix_source" | "save_files" | "edit_profile" | "select_environment" | "review_dependencies" | "provide_inputs" | "provide_factory" | "adjust_limits" | "retry" | "inspect_failure" | "none";
            /** Sourcebindingid */
            sourceBindingId: string | null;
        };
        /** DisplayPort */
        DisplayPort: {
            /**
             * Argumentkind
             * @enum {string}
             */
            argumentKind: "value" | "none" | "bool" | "int" | "float" | "str" | "dtype" | "device" | "layout" | "memory_format" | "symbolic" | "opaque" | "list" | "tuple" | "dict";
            /** Device */
            device: string | null;
            /**
             * Direction
             * @enum {string}
             */
            direction: "in" | "out";
            /**
             * Displayname
             * @default null
             */
            displayName?: string | null;
            /** Dtype */
            dtype: string | null;
            /** Id */
            id: string;
            /** Memberportid */
            memberPortId: string | null;
            /** Nodeid */
            nodeId: string;
            /** Paths */
            paths: (number | string)[][];
            /** Shape */
            shape: number[] | null;
            /** Valueid */
            valueId: string | null;
        };
        /**
         * EnvironmentFact
         * @description Already-probed machine-local environment, without credentials or env vars.
         */
        EnvironmentFact: {
            /** Executable */
            executable: string;
            /** Probe Digest */
            probe_digest: string;
            /** Python Version */
            python_version: string;
            /** Resolved Executable */
            resolved_executable: string;
            /** Selection Revision */
            selection_revision: number;
            /** Torch Version */
            torch_version: string | null;
        };
        /** FileIdentity */
        FileIdentity: {
            /** Digest */
            digest: string;
            /** Documentversion */
            documentVersion: number | null;
            /** Path */
            path: string;
            /**
             * Role
             * @enum {string}
             */
            role: "config" | "source" | "native";
        };
        /** FloatLiteral */
        FloatLiteral: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "float";
            /** Value */
            value: number | ("nan" | "inf" | "-inf" | "-0");
        };
        /**
         * InitializerArguments
         * @description Bound scalar parameters at entry, not authored call arguments or completion.
         */
        InitializerArguments: {
            /**
             * Keywordonly
             * @default []
             */
            keywordOnly?: components["schemas"]["InitializerParameter"][];
            /**
             * Phase
             * @constant
             */
            phase: "initializer_entry";
            /**
             * Positional
             * @default []
             */
            positional?: components["schemas"]["InitializerParameter"][];
            /**
             * Positionalonly
             * @default 0
             */
            positionalOnly?: number;
            /**
             * Reason
             * @default null
             */
            reason?: ("byte_limit" | "item_limit" | "unsupported_type") | null;
            /**
             * Status
             * @enum {string}
             */
            status: "observed" | "unavailable";
            /**
             * Varargs
             * @default null
             */
            varArgs?: (components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["OpaqueArgument"])[] | null;
            /**
             * Varkwargs
             * @default null
             */
            varKwargs?: {
                [key: string]: components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["OpaqueArgument"];
            } | null;
        };
        /** InitializerParameter */
        InitializerParameter: {
            /** Name */
            name: string;
            /** Value */
            value: components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["OpaqueArgument"];
        };
        /** InsertUnaryCapability */
        InsertUnaryCapability: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            binding: components["schemas"]["SourceBinding"];
            /**
             * Command
             * @constant
             */
            command: "insert_unary_op";
            expected: components["schemas"]["ValueRef"];
            /** Expectedname */
            expectedName: string;
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            /** @default null */
            importPlan?: components["schemas"]["ActivationImportPlan"] | null;
            /** Namespace */
            namespace: string;
            /**
             * Recipe
             * @constant
             */
            recipe: "straight_line_unary_insert_v1";
            scope: components["schemas"]["SourceEditScope"];
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** InsertUnaryEdit */
        InsertUnaryEdit: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            /** Capabilityid */
            capabilityId: string;
            expected: components["schemas"]["ValueRef"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "insert_unary_op";
            /**
             * Scope
             * @constant
             */
            scope: "source_expression";
            target: components["schemas"]["CaptureReference"];
            /**
             * Value
             * @enum {string}
             */
            value: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
        };
        /** IntLiteral */
        IntLiteral: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "int";
            /** Value */
            value: string;
        };
        /** InterpreterIdentity */
        InterpreterIdentity: {
            /** Environmentdigest */
            environmentDigest: string;
            /** Executable */
            executable: string;
            /** Version */
            version: string;
        };
        /** Invocation */
        Invocation: {
            arguments: components["schemas"]["Arguments"];
            /** Diagnosticid */
            diagnosticId: string | null;
            /** Endsequence */
            endSequence: number | null;
            /** Gradenabled */
            gradEnabled: boolean;
            /** Id */
            id: string;
            /** Moduleid */
            moduleId: string;
            /** Output */
            output: (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"]) | null;
            /**
             * Parametercount
             * @default null
             */
            parameterCount?: number | null;
            /**
             * Parametercountunavailable
             * @default null
             */
            parameterCountUnavailable?: ("item_limit" | "unsupported_type") | null;
            /** Parentid */
            parentId: string | null;
            /** Sequence */
            sequence: number;
            /** Sourcebindingids */
            sourceBindingIds: string[];
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "failed" | "interrupted";
            /** Training */
            training: boolean;
        };
        JsonValue: unknown;
        /** LiteralEditCapability */
        LiteralEditCapability: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            binding: components["schemas"]["SourceBinding"];
            /**
             * Command
             * @constant
             */
            command: "set_literal";
            /** Expected */
            expected: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            /**
             * Recipe
             * @constant
             */
            recipe: "binary_scalar_v1";
            scope: components["schemas"]["SourceEditScope"];
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** MappingArgument */
        MappingArgument: {
            /** Entries */
            entries: components["schemas"]["MappingEntry"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "dict";
        };
        /** MappingEntry */
        MappingEntry: {
            /** Key */
            key: components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["TextLiteral"];
            /** Value */
            value: components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"];
        };
        /** ModuleDefinition */
        ModuleDefinition: {
            /** Id */
            id: string;
            /** Qualifiedname */
            qualifiedName: string;
            /** Sourcebindingids */
            sourceBindingIds: string[];
        };
        /** ModuleInstance */
        ModuleInstance: {
            /**
             * Constructorbindingids
             * @default []
             */
            constructorBindingIds?: string[];
            /**
             * Constructorclassverified
             * @default false
             */
            constructorClassVerified?: boolean;
            /** Definitionid */
            definitionId: string;
            /** Id */
            id: string;
            /** @default null */
            initializerArguments?: components["schemas"]["InitializerArguments"] | null;
            /** Paths */
            paths: string[];
        };
        /** MutationEffect */
        MutationEffect: {
            /** After */
            after: components["schemas"]["ValueRef"] | components["schemas"]["OpaqueArgument"];
            /** Argumentpath */
            argumentPath: (number | string)[];
            /** Beforeid */
            beforeId: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "write" | "alias_update";
        };
        /** NoneLiteral */
        NoneLiteral: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "none";
        };
        /** OpaqueArgument */
        OpaqueArgument: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "opaque";
            /**
             * Reason
             * @enum {string}
             */
            reason: "operation_limit" | "invocation_limit" | "value_limit" | "byte_limit" | "depth_limit" | "item_limit" | "unsupported_type" | "opaque_kernel" | "recorder_error" | "timeout" | "cancelled";
            /** Typename */
            typeName: string | null;
        };
        /** Operation */
        Operation: {
            arguments: components["schemas"]["Arguments"];
            /** Diagnosticid */
            diagnosticId: string | null;
            /** Effects */
            effects: components["schemas"]["MutationEffect"][];
            /** Id */
            id: string;
            /** Invocationid */
            invocationId: string | null;
            /** Output */
            output: (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"]) | null;
            /** Schemasignature */
            schemaSignature: string | null;
            /** Sequence */
            sequence: number;
            /** Sourcebindingids */
            sourceBindingIds: string[];
            sourceFlattenCall?: components["schemas"]["SourceFlattenCall"] | null;
            sourceShapeCall?: components["schemas"]["SourceShapeCall"] | null;
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "failed" | "interrupted";
            /** Target */
            target: string;
        };
        /** OutputRef */
        OutputRef: {
            /**
             * Kind
             * @enum {string}
             */
            kind: "output" | "effect";
            /** Operationid */
            operationId: string;
            /** Path */
            path: (number | string)[];
        };
        /** PortableCapture */
        PortableCapture: {
            capture: components["schemas"]["CaptureSnapshot"];
            /**
             * Sourcenames
             * @constant
             */
            sourceNames: "anonymized";
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ProjectionEdge */
        ProjectionEdge: {
            /** Displaysourceportid */
            displaySourcePortId: string | null;
            /** Displaytargetportid */
            displayTargetPortId: string | null;
            /** Id */
            id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "argument" | "effect" | "result";
            /** Sourceportid */
            sourcePortId: string;
            /** Targetportid */
            targetPortId: string;
            /** Valueid */
            valueId: string;
        };
        /** ProjectionNode */
        ProjectionNode: {
            /** Collapsed */
            collapsed: boolean;
            /** Hiddenoperationcount */
            hiddenOperationCount: number;
            /** Id */
            id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "operation" | "invocation" | "input" | "output" | "value";
            /** Parentid */
            parentId: string | null;
            /** Semanticid */
            semanticId: string;
            /** Visible */
            visible: boolean;
        };
        /** ProjectionPort */
        ProjectionPort: {
            /**
             * Argumentkind
             * @enum {string}
             */
            argumentKind: "value" | "none" | "bool" | "int" | "float" | "str" | "dtype" | "device" | "layout" | "memory_format" | "symbolic" | "opaque" | "list" | "tuple" | "dict";
            /**
             * Direction
             * @enum {string}
             */
            direction: "in" | "out";
            /** Id */
            id: string;
            /** Memberportid */
            memberPortId: string | null;
            /** Nodeid */
            nodeId: string;
            /** Paths */
            paths: (number | string)[][];
            /** Valueid */
            valueId: string | null;
        };
        /** ReconnectArgumentCapability */
        ReconnectArgumentCapability: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            binding: components["schemas"]["SourceBinding"];
            /** Choices */
            choices: components["schemas"]["SourceValueChoice"][];
            /**
             * Command
             * @constant
             */
            command: "reconnect_argument";
            expected: components["schemas"]["ValueRef"];
            /** Expectedname */
            expectedName: string;
            expression: components["schemas"]["SourceBinding"];
            /** Id */
            id: string;
            /**
             * Recipe
             * @constant
             */
            recipe: "straight_line_tensor_v1";
            scope: components["schemas"]["SourceEditScope"];
            target: components["schemas"]["CaptureReference"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ReconnectArgumentEdit */
        ReconnectArgumentEdit: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            /** Capabilityid */
            capabilityId: string;
            expected: components["schemas"]["ValueRef"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "reconnect_argument";
            /**
             * Scope
             * @constant
             */
            scope: "source_expression";
            target: components["schemas"]["CaptureReference"];
            value: components["schemas"]["ValueRef"];
        };
        /** ReplaceActivationEdit */
        ReplaceActivationEdit: {
            /** Capabilityid */
            capabilityId: string;
            /**
             * Expected
             * @enum {string}
             */
            expected: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "replace_activation";
            /**
             * Scope
             * @constant
             */
            scope: "source_expression";
            target: components["schemas"]["CaptureReference"];
            /**
             * Value
             * @enum {string}
             */
            value: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
        };
        /**
         * ResolvedExecutionPlan
         * @description Host-resolved saved invocation. ExecutionSpec remains the recipe authority.
         *
         *     Automatic guesses are not resolved plans. The recipe's nested inputs retain
         *     validation.Binding or explicit tensor/fixture descriptions, never observations.
         */
        ResolvedExecutionPlan: {
            captureLimits: components["schemas"]["CaptureLimits"];
            /**
             * Configuration
             * @enum {string}
             */
            configuration: "absent" | "selected" | "overridden";
            environment: components["schemas"]["EnvironmentFact"];
            /** Execution */
            execution: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Id */
            id: string;
            interpreter: components["schemas"]["InterpreterIdentity"];
            /** Job */
            job: {
                [key: string]: components["schemas"]["JsonValue"];
            } | null;
            /**
             * Origin
             * @enum {string}
             */
            origin: "profile" | "explicit";
            /** Profiledigest */
            profileDigest: string | null;
            /** Profilename */
            profileName: string | null;
            source: components["schemas"]["SourceSnapshot"];
            /** Timeoutms */
            timeoutMs: number;
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ReviewedPythonSource */
        ReviewedPythonSource: {
            /** Beforedigest */
            beforeDigest: string;
            /** File */
            file: string;
            /** Source */
            source: string;
        };
        /** RunAttempt */
        RunAttempt: {
            /** Captureids */
            captureIds: string[];
            /** Diagnostics */
            diagnostics: components["schemas"]["Diagnostic"][];
            /** Finishedat */
            finishedAt: number | null;
            /** Id */
            id: string;
            /**
             * Origin
             * @enum {string}
             */
            origin: "manual" | "mcp" | "automatic";
            /**
             * Phase
             * @enum {string}
             */
            phase: "parse" | "index" | "configuration" | "environment" | "import" | "construction" | "binding" | "materialization" | "forward" | "capture" | "projection" | "serialization" | "transport";
            /** Plandigest */
            planDigest: string | null;
            /** Requestdigest */
            requestDigest: string;
            /**
             * Reusedcaptureid
             * @default null
             */
            reusedCaptureId?: string | null;
            /** Sourceid */
            sourceId: string | null;
            /** Startedat */
            startedAt: number;
            /**
             * Status
             * @enum {string}
             */
            status: "requested" | "running" | "needs_setup" | "succeeded" | "failed" | "cancelled" | "timed_out";
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** SemanticContracts */
        SemanticContracts: {
            activationEditCapability: components["schemas"]["ActivationEditCapability"];
            candidateExecution: components["schemas"]["CandidateExecution"];
            candidatePostconditions: components["schemas"]["CandidatePostconditions"];
            candidateRequest: components["schemas"]["CandidateRequest"];
            capture: components["schemas"]["CaptureSnapshot"];
            captureReconciliation: components["schemas"]["CaptureReconciliation"];
            captureReconciliationRequest: components["schemas"]["CaptureReconciliationRequest"];
            captureReference: components["schemas"]["CaptureReference"];
            configurationLiteralCapability: components["schemas"]["ConfigurationLiteralCapability"];
            constructorLiteralCapability: components["schemas"]["ConstructorLiteralCapability"];
            correspondence: components["schemas"]["SourceCorrespondence"];
            deleteUnaryCapability: components["schemas"]["DeleteUnaryCapability"];
            deleteUnaryEdit: components["schemas"]["DeleteUnaryEdit"];
            displayPort: components["schemas"]["DisplayPort"];
            insertUnaryCapability: components["schemas"]["InsertUnaryCapability"];
            insertUnaryEdit: components["schemas"]["InsertUnaryEdit"];
            literalEditCapability: components["schemas"]["LiteralEditCapability"];
            plan: components["schemas"]["ResolvedExecutionPlan"];
            portableCapture: components["schemas"]["PortableCapture"];
            preparation: components["schemas"]["CapturePreparation"];
            projection: components["schemas"]["CaptureProjection"];
            reconnectArgumentCapability: components["schemas"]["ReconnectArgumentCapability"];
            reconnectArgumentEdit: components["schemas"]["ReconnectArgumentEdit"];
            replaceActivationEdit: components["schemas"]["ReplaceActivationEdit"];
            setConfigurationLiteralEdit: components["schemas"]["SetConfigurationLiteralEdit"];
            setConstructorLiteralEdit: components["schemas"]["SetConstructorLiteralEdit"];
            setLiteralEdit: components["schemas"]["SetLiteralEdit"];
            sourceCandidateExecution: components["schemas"]["SourceCandidateExecution"];
            sourceCandidateRequest: components["schemas"]["SourceCandidateRequest"];
            state: components["schemas"]["CaptureState"];
        };
        /** SequenceArgument */
        SequenceArgument: {
            /** Items */
            items: (components["schemas"]["ValueRef"] | components["schemas"]["NoneLiteral"] | components["schemas"]["BoolLiteral"] | components["schemas"]["IntLiteral"] | components["schemas"]["FloatLiteral"] | components["schemas"]["TextLiteral"] | components["schemas"]["SymbolicArgument"] | components["schemas"]["OpaqueArgument"] | components["schemas"]["SequenceArgument"] | components["schemas"]["MappingArgument"])[];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "list" | "tuple";
        };
        /** SetConfigurationLiteralEdit */
        SetConfigurationLiteralEdit: {
            /** Capabilityid */
            capabilityId: string;
            /** Expected */
            expected: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "set_configuration_literal";
            /**
             * Scope
             * @constant
             */
            scope: "configuration_definition";
            /** Sourceargumentpath */
            sourceArgumentPath: (number | string)[];
            target: components["schemas"]["CaptureReference"];
            /** Value */
            value: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
        };
        /** SetConstructorLiteralEdit */
        SetConstructorLiteralEdit: {
            /** Capabilityid */
            capabilityId: string;
            /** Expected */
            expected: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "set_constructor_literal";
            /**
             * Scope
             * @constant
             */
            scope: "source_expression";
            /** Sourceargumentpath */
            sourceArgumentPath: (number | string)[];
            target: components["schemas"]["CaptureReference"];
            /** Value */
            value: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
        };
        /** SetLiteralEdit */
        SetLiteralEdit: {
            /** Argumentpath */
            argumentPath: (number | string)[];
            /** Capabilityid */
            capabilityId: string;
            /** Expected */
            expected: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "set_literal";
            /**
             * Scope
             * @constant
             */
            scope: "source_expression";
            target: components["schemas"]["CaptureReference"];
            /** Value */
            value: components["schemas"]["IntLiteral"] | components["schemas"]["SourceFloatLiteral"];
        };
        /** SourceAnchor */
        SourceAnchor: {
            /** Digest */
            digest: string;
            /** Endcolumn */
            endColumn: number;
            /** Endline */
            endLine: number;
            /** File */
            file: string;
            /** Scope */
            scope: string;
            /** Startcolumn */
            startColumn: number;
            /** Startline */
            startLine: number;
            /** Tokendigest */
            tokenDigest: string;
        };
        /** SourceBinding */
        SourceBinding: {
            /** Anchors */
            anchors: components["schemas"]["SourceAnchor"][];
            /** Id */
            id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "definition" | "constructor" | "call" | "argument" | "configuration";
            /**
             * Quality
             * @enum {string}
             */
            quality: "exact" | "candidate" | "ambiguous";
            /** Snapshotid */
            snapshotId: string;
        };
        /**
         * SourceCandidateExecution
         * @description Exact reviewed source set; successful replay is not a recipe verdict.
         */
        SourceCandidateExecution: {
            /** Baseattemptid */
            baseAttemptId: string;
            /** Basecaptureid */
            baseCaptureId: string;
            /** Baseplandigest */
            basePlanDigest: string;
            /** Basesourceid */
            baseSourceId: string;
            /** Baseworkspaceid */
            baseWorkspaceId: string;
            /** Changes */
            changes: components["schemas"]["SourceChangeIdentity"][];
            /** Id */
            id: string;
            plan: components["schemas"]["ResolvedExecutionPlan"];
            /**
             * Version
             * @constant
             */
            version: 2;
        };
        /**
         * SourceCandidateRequest
         * @description Private exact-file proposal; execution still requires the host Run action.
         */
        SourceCandidateRequest: {
            capture: components["schemas"]["CaptureSnapshot"];
            /** Id */
            id: string;
            plan: components["schemas"]["ResolvedExecutionPlan"];
            /** Reviewedsources */
            reviewedSources: components["schemas"]["ReviewedPythonSource"][];
            /**
             * Version
             * @constant
             */
            version: 2;
        };
        /** SourceChangeIdentity */
        SourceChangeIdentity: {
            /** Afterdigest */
            afterDigest: string;
            /** Beforedigest */
            beforeDigest: string;
            /** File */
            file: string;
        };
        /**
         * SourceCorrespondence
         * @description Revision-specific evidence for a source card, not proof that an expression ran.
         */
        SourceCorrespondence: {
            /**
             * Basis
             * @enum {string}
             */
            basis: "registered_path" | "shared_instance" | "state_reference" | "no_mapping";
            /** Captureid */
            captureId: string;
            /** Invocationids */
            invocationIds: string[];
            /** Moduleid */
            moduleId: string | null;
            /** Nodeid */
            nodeId: string;
            /** Reason */
            reason: ("operation_limit" | "invocation_limit" | "value_limit" | "byte_limit" | "depth_limit" | "item_limit" | "unsupported_type" | "opaque_kernel" | "recorder_error" | "timeout" | "cancelled") | null;
            /** Sourceid */
            sourceId: string;
            /** Statevalueids */
            stateValueIds: string[];
            /**
             * Status
             * @enum {string}
             */
            status: "observed" | "failed" | "not_observed" | "capture_limited" | "static_only" | "unresolved" | "state";
        };
        /** SourceEditScope */
        SourceEditScope: {
            /**
             * Affects
             * @constant
             */
            affects: "all_executions";
            /**
             * Includesunobservedexecutions
             * @constant
             */
            includesUnobservedExecutions: true;
            /**
             * Kind
             * @constant
             */
            kind: "source_expression";
            /** Observedinvocations */
            observedInvocations: number;
            /** Observedmoduleinstances */
            observedModuleInstances: number;
            /** Observedoperations */
            observedOperations: number;
        };
        /** SourceFlattenArgument */
        SourceFlattenArgument: {
            /**
             * Parameter
             * @enum {string}
             */
            parameter: "start_dim" | "end_dim";
            /** Sourcepath */
            sourcePath: (number | string)[];
            /** Value */
            value: number;
        };
        /** SourceFlattenCall */
        SourceFlattenCall: {
            /** Arguments */
            arguments: components["schemas"]["SourceFlattenArgument"][];
            /** Bindingid */
            bindingId: string;
            /**
             * Callable
             * @enum {string}
             */
            callable: "torch.flatten" | "torch.Tensor.flatten";
            /** Enddim */
            endDim: number;
            /** Startdim */
            startDim: number;
        };
        /** SourceFloatLiteral */
        SourceFloatLiteral: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "float";
            /** Value */
            value: number | "-0";
        };
        /** SourceShapeCall */
        SourceShapeCall: {
            /** Bindingid */
            bindingId: string;
            /**
             * Callable
             * @enum {string}
             */
            callable: "torch.reshape" | "torch.Tensor.reshape" | "torch.Tensor.view";
            /** Dimensions */
            dimensions: components["schemas"]["SourceShapeDimension"][];
        };
        /** SourceShapeDimension */
        SourceShapeDimension: {
            /** Dimension */
            dimension: number;
            /** Sourcepath */
            sourcePath: (number | string)[];
            tensorDimension?: components["schemas"]["SourceTensorDimension"] | null;
            value: components["schemas"]["IntLiteral"];
        };
        /** SourceSnapshot */
        SourceSnapshot: {
            /** Classname */
            className: string;
            /** Configdigest */
            configDigest: string | null;
            /**
             * Dependencyscope
             * @enum {string}
             */
            dependencyScope: "complete" | "partial";
            /** Environmentdigest */
            environmentDigest: string;
            /** Files */
            files: components["schemas"]["FileIdentity"][];
            /** Generation */
            generation: number;
            /** Id */
            id: string;
            /** Modelfile */
            modelFile: string;
            /** Modelid */
            modelId: string;
            /**
             * Sourcemode
             * @enum {string}
             */
            sourceMode: "saved" | "buffers";
            /**
             * Version
             * @constant
             */
            version: 1;
            /** Workspaceid */
            workspaceId: string;
            /** Workspacesessionid */
            workspaceSessionId: string;
        };
        /** SourceTensorDimension */
        SourceTensorDimension: {
            /** Axis */
            axis: number;
            /** Valueid */
            valueId: string;
        };
        /** SourceValueChoice */
        SourceValueChoice: {
            binding: components["schemas"]["SourceBinding"];
            /** Name */
            name: string;
            value: components["schemas"]["ValueRef"];
        };
        /** StateRef */
        StateRef: {
            /**
             * Kind
             * @enum {string}
             */
            kind: "parameter" | "buffer";
            /** Name */
            name: string;
        };
        /** SymbolicArgument */
        SymbolicArgument: {
            /** Expression */
            expression: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "symbolic";
            /**
             * Type
             * @enum {string}
             */
            type: "int" | "float" | "bool";
        };
        /** TextLiteral */
        TextLiteral: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "device" | "dtype" | "layout" | "memory_format" | "str";
            /** Value */
            value: string;
        };
        /** Value */
        Value: {
            /** Aliasof */
            aliasOf: string | null;
            /** Device */
            device: string;
            /** Dtype */
            dtype: string;
            /** Id */
            id: string;
            /**
             * Logicalbytes
             * @default null
             */
            logicalBytes?: number | null;
            /** Logicalversion */
            logicalVersion: number;
            /** Objectid */
            objectId: string;
            /**
             * Origin
             * @enum {string}
             */
            origin: "input" | "state" | "operation" | "external";
            /** Previousversion */
            previousVersion: string | null;
            producer: components["schemas"]["OutputRef"] | null;
            /** Requiresgrad */
            requiresGrad: boolean;
            /** Shape */
            shape: number[] | null;
            /** State */
            state: components["schemas"]["StateRef"][];
            /** Storageid */
            storageId: string | null;
            /** Storageoffset */
            storageOffset: number | null;
            /** Stride */
            stride: number[] | null;
            /** Version */
            version: number | null;
        };
        /** ValueRef */
        ValueRef: {
            /** Id */
            id: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "value";
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export type operations = Record<string, never>;
