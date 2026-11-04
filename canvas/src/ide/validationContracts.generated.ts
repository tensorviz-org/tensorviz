/* GENERATED from backend/services/validation/contracts.py; npm run validation:generate from the repository root. */
export type paths = Record<string, never>;
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** AutocastContext */
        AutocastContext: {
            /** Dtype */
            dtype: string | null;
            /** Enabled */
            enabled: boolean;
        };
        /** Binding */
        Binding: {
            /** Args */
            args: (boolean | number | string | (components["schemas"]["FloatValue"] | components["schemas"]["NoneValue"] | components["schemas"]["SequenceValue"] | components["schemas"]["DictValue"] | components["schemas"]["TensorValue"]))[];
            /** Kwargs */
            kwargs: {
                [key: string]: (boolean | number | string | (components["schemas"]["FloatValue"] | components["schemas"]["NoneValue"] | components["schemas"]["SequenceValue"] | components["schemas"]["DictValue"] | components["schemas"]["TensorValue"])) | (boolean | number | string | (components["schemas"]["FloatValue"] | components["schemas"]["NoneValue"] | components["schemas"]["SequenceValue"] | components["schemas"]["DictValue"] | components["schemas"]["TensorValue"]));
            };
        };
        /** CallContext */
        CallContext: {
            /** Autocast */
            autocast: {
                [key: string]: components["schemas"]["AutocastContext"] | components["schemas"]["AutocastContext"];
            };
            /** Gradenabled */
            gradEnabled: boolean;
            /** Inferencemode */
            inferenceMode: boolean;
            /** Training */
            training: boolean | null;
        };
        /** CaptureComplete */
        CaptureComplete: {
            /** Rootinvocationid */
            rootInvocationId: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "complete";
        };
        /** CaptureIncomplete */
        CaptureIncomplete: {
            /**
             * Reason
             * @enum {string}
             */
            reason: "target_not_observed" | "unsupported" | "event_limit" | "tensor_limit" | "byte_limit" | "timeout" | "cancelled" | "recorder_error" | "transport_error";
            /** Rootinvocationid */
            rootInvocationId: string | null;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "incomplete";
        };
        /** CaptureOpen */
        CaptureOpen: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "recording" | "waiting";
        };
        /** CaptureSelector */
        CaptureSelector: {
            /** Call */
            call: number;
            /** Model */
            model: string;
        };
        /** DictEntry */
        DictEntry: {
            /** Key */
            key: string | number;
            /** Value */
            value: components["schemas"]["TensorPort"] | components["schemas"]["ValuePort"] | components["schemas"]["OpaquePort"] | components["schemas"]["SequencePort"] | components["schemas"]["DictPort"];
        };
        /** DictPort */
        DictPort: {
            /** Entries */
            entries: components["schemas"]["DictEntry"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "dict";
        };
        /** DictValue */
        DictValue: {
            /** Items */
            items: {
                [key: string]: boolean | number | string | (components["schemas"]["FloatValue"] | components["schemas"]["NoneValue"] | components["schemas"]["SequenceValue"] | components["schemas"]["DictValue"] | components["schemas"]["TensorValue"]);
            };
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "dict";
        };
        /** ExecutionRecord */
        ExecutionRecord: {
            /** Capture */
            capture: components["schemas"]["CaptureOpen"] | components["schemas"]["CaptureComplete"] | components["schemas"]["CaptureIncomplete"];
            context: components["schemas"]["ObservedContext"] | null;
            /** Edges */
            edges: components["schemas"]["TensorEdge"][];
            identity: components["schemas"]["RunIdentity"];
            /** Invocations */
            invocations: components["schemas"]["Invocation"][];
            limits: components["schemas"]["Limits"];
            /** Modules */
            modules: components["schemas"]["ModuleInstance"][];
            /** Process */
            process: components["schemas"]["ProcessRunning"] | components["schemas"]["ProcessExited"] | components["schemas"]["ProcessStopped"];
            /** Requirements */
            requirements: components["schemas"]["Requirement"][];
            /** Sequence */
            sequence: number;
            /** Tensors */
            tensors: components["schemas"]["TensorFact"][];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ExecutionUpdate */
        ExecutionUpdate: {
            freshness: components["schemas"]["Freshness"];
            record: components["schemas"]["ExecutionRecord"];
            /**
             * Type
             * @constant
             */
            type: "executionUpdate";
            /**
             * Version
             * @constant
             */
            version: 1;
            /** Workspacesessionid */
            workspaceSessionId: string;
        };
        /** Failure */
        Failure: {
            /** Code */
            code: string;
            /**
             * Phase
             * @enum {string}
             */
            phase: "setup" | "construction" | "forward" | "script" | "recorder" | "transport" | "resource";
            source: components["schemas"]["SourceLocation"] | null;
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
        /** FloatValue */
        FloatValue: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "float";
            /** Value */
            value: number;
        };
        /** Freshness */
        Freshness: {
            /** Reasons */
            reasons: ("source" | "config" | "native_file" | "profile" | "environment" | "cancelled" | "session" | "dependency_scope")[];
            /**
             * State
             * @enum {string}
             */
            state: "current" | "stale";
        };
        /** Full */
        Full: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "full";
            /** Value */
            value: boolean | number;
        };
        /** InputFactory */
        InputFactory: {
            factory: components["schemas"]["ModelFactory"];
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
            context: components["schemas"]["CallContext"];
            failure: components["schemas"]["Failure"] | null;
            /** Id */
            id: string;
            /** Inputs */
            inputs: (components["schemas"]["TensorPort"] | components["schemas"]["ValuePort"] | components["schemas"]["OpaquePort"] | components["schemas"]["SequencePort"] | components["schemas"]["DictPort"])[];
            /**
             * Kind
             * @enum {string}
             */
            kind: "module" | "function" | "opaque";
            /** Kwargs */
            kwargs: {
                [key: string]: (components["schemas"]["TensorPort"] | components["schemas"]["ValuePort"] | components["schemas"]["OpaquePort"] | components["schemas"]["SequencePort"] | components["schemas"]["DictPort"]) | (components["schemas"]["TensorPort"] | components["schemas"]["ValuePort"] | components["schemas"]["OpaquePort"] | components["schemas"]["SequencePort"] | components["schemas"]["DictPort"]);
            };
            /** Moduleid */
            moduleId: string | null;
            /** Name */
            name: string;
            /**
             * Outcome
             * @enum {string}
             */
            outcome: "succeeded" | "failed" | "unsupported";
            /** Output */
            output: (components["schemas"]["TensorPort"] | components["schemas"]["ValuePort"] | components["schemas"]["OpaquePort"] | components["schemas"]["SequencePort"] | components["schemas"]["DictPort"]) | null;
            /** Parentid */
            parentId: string | null;
            source: components["schemas"]["SourceLocation"] | null;
        };
        JsonValue: unknown;
        /** Limits */
        Limits: {
            /**
             * Max Capture Bytes
             * @default null
             */
            max_capture_bytes?: number | null;
            /**
             * Max Events
             * @default 10000
             */
            max_events?: number;
            /**
             * Max Record Bytes
             * @default 4194304
             */
            max_record_bytes?: number;
            /**
             * Max Tensors
             * @default 10000
             */
            max_tensors?: number;
            /**
             * Timeout Ms
             * @default 60000
             */
            timeout_ms?: number;
        };
        /** ModelFactory */
        ModelFactory: {
            /** Callable */
            callable: string;
            /** File */
            file: string;
            /** Kwargs */
            kwargs?: {
                [key: string]: components["schemas"]["JsonValue"] | components["schemas"]["JsonValue"];
            };
        };
        /** ModelProfile */
        ModelProfile: {
            /**
             * Cwd
             * @default .
             */
            cwd?: "." | string;
            /**
             * Deterministic
             * @default true
             */
            deterministic?: boolean;
            /** Device */
            device: string;
            /** @default null */
            factory?: components["schemas"]["ModelFactory"] | null;
            /**
             * Framework
             * @default null
             */
            framework?: "jax" | null;
            /** Grad Enabled */
            grad_enabled: boolean;
            /** @default null */
            init?: components["schemas"]["Binding"] | null;
            /** Inputs */
            inputs: components["schemas"]["Binding"] | components["schemas"]["InputFactory"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "model";
            limits?: components["schemas"]["Limits"];
            /**
             * Mode
             * @enum {string}
             */
            mode: "train" | "eval";
            /** Model */
            model: string;
            /**
             * Seed
             * @default null
             */
            seed?: number | null;
            /** Tracked Files */
            tracked_files?: string[];
        } & ({
            factory?: null;
            init: unknown;
        } | {
            factory: unknown;
            init?: null;
        });
        /** ModuleInstance */
        ModuleInstance: {
            /** Classsymbol */
            classSymbol: string;
            /** Id */
            id: string;
            source: components["schemas"]["SourceLocation"] | null;
        };
        /** NeedsSetup */
        NeedsSetup: {
            /** Requirements */
            requirements: components["schemas"]["Requirement"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            status: "needs_setup";
        };
        /** NoneValue */
        NoneValue: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "none";
        };
        /** Normal */
        Normal: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "normal";
            /** Mean */
            mean: number;
            /** Seed */
            seed: number;
            /** Std */
            std: number;
        };
        /** ObservedContext */
        ObservedContext: {
            /** Autocast */
            autocast: {
                [key: string]: components["schemas"]["AutocastContext"] | components["schemas"]["AutocastContext"];
            };
            /** Devices */
            devices: string[];
            /** Gradenabled */
            gradEnabled: boolean;
            /** Inferencemode */
            inferenceMode: boolean;
            /** Torchversion */
            torchVersion: string;
            /** Training */
            training: boolean;
        };
        /** OpaquePort */
        OpaquePort: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "opaque";
            /**
             * Reason
             * @enum {string}
             */
            reason: "unsupported_type" | "depth_limit" | "item_limit";
        };
        /** PortRef */
        PortRef: {
            /** Invocationid */
            invocationId: string;
            /** Path */
            path: (number | string)[];
            /**
             * Slot
             * @enum {string}
             */
            slot: "args" | "kwargs" | "output";
        };
        /** ProcessExited */
        ProcessExited: {
            /** Exitcode */
            exitCode: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "exited";
        };
        /** ProcessRunning */
        ProcessRunning: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "running";
        };
        /** ProcessStopped */
        ProcessStopped: {
            failure: components["schemas"]["Failure"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "cancelled" | "failed" | "timed_out";
        };
        /** RandInt */
        RandInt: {
            /** High */
            high: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "randint";
            /** Low */
            low: number;
            /** Seed */
            seed: number;
        };
        /** Requirement */
        Requirement: {
            /**
             * Code
             * @enum {string}
             */
            code: "missing" | "invalid" | "invalid_source" | "unknown_key" | "ambiguous" | "unsupported" | "unsaved" | "missing_file" | "environment" | "signature" | "removed_format";
            /** Field */
            field: (string | number)[];
            /**
             * Phase
             * @enum {string}
             */
            phase: "static" | "runtime";
            /** Profile */
            profile: string | null;
            /** Reason */
            reason: string;
            /** Remedy */
            remedy: string;
        };
        /** Resolved */
        Resolved: {
            spec: components["schemas"]["RunSpec"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            status: "resolved";
        };
        /** ResultChannel */
        ResultChannel: {
            /**
             * Host
             * @constant
             */
            host: "127.0.0.1";
            /** Port */
            port: number;
            /** Token */
            token: string;
        };
        /** RunIdentity */
        RunIdentity: {
            /** Configdigest */
            configDigest: string;
            /** Environmentdigest */
            environmentDigest: string;
            /** Generation */
            generation: number;
            /** Profile */
            profile: string;
            /** Profiledigest */
            profileDigest: string;
            /** Runid */
            runId: string;
            /** Sourcedigest */
            sourceDigest: string;
            /** Workspaceid */
            workspaceId: string;
            /** Workspacesessionid */
            workspaceSessionId: string;
        };
        /** RunSpec */
        RunSpec: {
            /**
             * Dependencyscope
             * @enum {string}
             */
            dependencyScope: "complete" | "partial";
            /** Files */
            files: components["schemas"]["FileIdentity"][];
            identity: components["schemas"]["RunIdentity"];
            interpreter: components["schemas"]["InterpreterIdentity"];
            /** Profile */
            profile: components["schemas"]["ScriptProfile"] | components["schemas"]["ModelProfile"];
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ScriptProfile */
        ScriptProfile: {
            /** Args */
            args: string[];
            capture: components["schemas"]["CaptureSelector"];
            /**
             * Cwd
             * @default .
             */
            cwd?: "." | string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "script";
            limits?: components["schemas"]["Limits"];
            /** Script */
            script: string;
            /** Tracked Files */
            tracked_files?: string[];
        };
        /** SequencePort */
        SequencePort: {
            /** Items */
            items: (components["schemas"]["TensorPort"] | components["schemas"]["ValuePort"] | components["schemas"]["OpaquePort"] | components["schemas"]["SequencePort"] | components["schemas"]["DictPort"])[];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "list" | "tuple";
        };
        /** SequenceValue */
        SequenceValue: {
            /** Items */
            items: (boolean | number | string | (components["schemas"]["FloatValue"] | components["schemas"]["NoneValue"] | components["schemas"]["SequenceValue"] | components["schemas"]["DictValue"] | components["schemas"]["TensorValue"]))[];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "list" | "tuple";
        };
        /** SourceLocation */
        SourceLocation: {
            /** Column */
            column: number;
            /** File */
            file: string;
            /** Line */
            line: number;
        };
        /** TensorEdge */
        TensorEdge: {
            consumer: components["schemas"]["PortRef"];
            producer: components["schemas"]["PortRef"] | null;
            /** Tensorid */
            tensorId: string;
        };
        /** TensorFact */
        TensorFact: {
            /** Aliasof */
            aliasOf: string | null;
            /** Device */
            device: string;
            /** Dtype */
            dtype: string;
            /** Id */
            id: string;
            /** Objectid */
            objectId: string;
            /** Previousversion */
            previousVersion: string | null;
            /** Requiresgrad */
            requiresGrad: boolean;
            /** Shape */
            shape: number[] | null;
            /** Storageid */
            storageId: string | null;
            /** Version */
            version: number | null;
        };
        /** TensorPort */
        TensorPort: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "tensor";
            /** Tensorid */
            tensorId: string;
        };
        /** TensorValue */
        TensorValue: {
            /**
             * Dtype
             * @enum {string}
             */
            dtype: "bool" | "uint8" | "int8" | "int16" | "int32" | "int64" | "float16" | "bfloat16" | "float32" | "float64";
            /** Generator */
            generator: components["schemas"]["Normal"] | components["schemas"]["Uniform"] | components["schemas"]["RandInt"] | components["schemas"]["Full"] | components["schemas"]["Values"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "tensor";
            /**
             * Requires Grad
             * @default false
             */
            requires_grad?: boolean;
            /** Shape */
            shape: number[];
        };
        /** Uniform */
        Uniform: {
            /** High */
            high: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "uniform";
            /** Low */
            low: number;
            /** Seed */
            seed: number;
        };
        /** ValidationConfig */
        ValidationConfig: {
            /** Profiles */
            profiles: {
                [key: string]: (components["schemas"]["ScriptProfile"] | components["schemas"]["ModelProfile"]) | (components["schemas"]["ScriptProfile"] | components["schemas"]["ModelProfile"]);
            };
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** ValidationContracts */
        ValidationContracts: {
            bootstrap: components["schemas"]["WorkerBootstrap"];
            config: components["schemas"]["ValidationConfig"];
            execution: components["schemas"]["ExecutionUpdate"];
            /** Resolution */
            resolution: components["schemas"]["Resolved"] | components["schemas"]["NeedsSetup"];
            /**
             * State
             * @enum {string}
             */
            state: "not_run" | "needs_setup" | "running" | "completed_for_input" | "execution_failed" | "capture_incomplete" | "cancelled" | "stale";
            /** Worker */
            worker: components["schemas"]["WorkerHello"] | components["schemas"]["WorkerRecord"];
        };
        /** ValuePort */
        ValuePort: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "value";
            /**
             * Type
             * @enum {string}
             */
            type: "none" | "bool" | "int" | "float" | "str" | "dtype" | "device" | "layout" | "memory_format";
        };
        /** Values */
        Values: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "values";
            /** Values */
            values: (boolean | number)[];
        };
        /** WorkerBootstrap */
        WorkerBootstrap: {
            resultChannel: components["schemas"]["ResultChannel"];
            spec: components["schemas"]["RunSpec"];
            /**
             * Version
             * @constant
             */
            version: 1;
            /** Workspaceroot */
            workspaceRoot: string;
        };
        /** WorkerHello */
        WorkerHello: {
            /** Runid */
            runId: string;
            /** Token */
            token: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "hello";
            /**
             * Version
             * @constant
             */
            version: 1;
        };
        /** WorkerRecord */
        WorkerRecord: {
            /** Dependenciescurrent */
            dependenciesCurrent: boolean;
            record: components["schemas"]["ExecutionRecord"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "record";
            /**
             * Version
             * @constant
             */
            version: 1;
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
