/* GENERATED from services/semantic/contracts.py; npm run semantic:generate. */
const schema: object = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "additionalProperties": false,
  "definitions": {
    "ActivationEditCapability": {
      "additionalProperties": false,
      "properties": {
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "command": {
          "const": "replace_activation",
          "title": "Command",
          "type": "string"
        },
        "compatibleActivations": {
          "items": {
            "enum": [
              "relu",
              "gelu",
              "silu",
              "sigmoid",
              "tanh"
            ],
            "type": "string"
          },
          "maxItems": 5,
          "minItems": 1,
          "title": "Compatibleactivations",
          "type": "array"
        },
        "expected": {
          "enum": [
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh"
          ],
          "title": "Expected",
          "type": "string"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "importPlan": {
          "anyOf": [
            {
              "$ref": "#/definitions/ActivationImportPlan"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "namespace": {
          "maxLength": 256,
          "minLength": 1,
          "pattern": "^[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*)*(?![\\s\\S])",
          "title": "Namespace",
          "type": "string"
        },
        "options": {
          "$ref": "#/definitions/ActivationSourceOptions"
        },
        "recipe": {
          "enum": [
            "functional_activation_v1",
            "module_activation_v1"
          ],
          "title": "Recipe",
          "type": "string"
        },
        "scope": {
          "$ref": "#/definitions/SourceEditScope"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "binding",
        "namespace",
        "expected",
        "scope"
      ],
      "title": "ActivationEditCapability",
      "type": "object"
    },
    "ActivationImportPlan": {
      "additionalProperties": false,
      "properties": {
        "line": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Line",
          "type": "integer"
        },
        "lineEnding": {
          "enum": [
            "\n",
            "\r\n"
          ],
          "title": "Lineending",
          "type": "string"
        },
        "module": {
          "enum": [
            "torch.nn",
            "torch.nn.functional"
          ],
          "title": "Module",
          "type": "string"
        }
      },
      "required": [
        "module",
        "line",
        "lineEnding"
      ],
      "title": "ActivationImportPlan",
      "type": "object"
    },
    "ActivationSourceOptions": {
      "additionalProperties": false,
      "properties": {
        "approximate": {
          "anyOf": [
            {
              "enum": [
                "none",
                "tanh"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Approximate"
        },
        "inplace": {
          "anyOf": [
            {
              "type": "boolean"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Inplace"
        }
      },
      "title": "ActivationSourceOptions",
      "type": "object"
    },
    "Arguments": {
      "additionalProperties": false,
      "properties": {
        "args": {
          "items": {
            "discriminator": {
              "mapping": {
                "bool": "#/definitions/BoolLiteral",
                "device": "#/definitions/TextLiteral",
                "dict": "#/definitions/MappingArgument",
                "dtype": "#/definitions/TextLiteral",
                "float": "#/definitions/FloatLiteral",
                "int": "#/definitions/IntLiteral",
                "layout": "#/definitions/TextLiteral",
                "list": "#/definitions/SequenceArgument",
                "memory_format": "#/definitions/TextLiteral",
                "none": "#/definitions/NoneLiteral",
                "opaque": "#/definitions/OpaqueArgument",
                "str": "#/definitions/TextLiteral",
                "symbolic": "#/definitions/SymbolicArgument",
                "tuple": "#/definitions/SequenceArgument",
                "value": "#/definitions/ValueRef"
              },
              "propertyName": "kind"
            },
            "oneOf": [
              {
                "$ref": "#/definitions/ValueRef"
              },
              {
                "$ref": "#/definitions/NoneLiteral"
              },
              {
                "$ref": "#/definitions/BoolLiteral"
              },
              {
                "$ref": "#/definitions/IntLiteral"
              },
              {
                "$ref": "#/definitions/FloatLiteral"
              },
              {
                "$ref": "#/definitions/TextLiteral"
              },
              {
                "$ref": "#/definitions/SymbolicArgument"
              },
              {
                "$ref": "#/definitions/OpaqueArgument"
              },
              {
                "$ref": "#/definitions/SequenceArgument"
              },
              {
                "$ref": "#/definitions/MappingArgument"
              }
            ]
          },
          "maxItems": 256,
          "title": "Args",
          "type": "array"
        },
        "kwargs": {
          "maxProperties": 256,
          "patternProperties": {
            "^[A-Za-z_][A-Za-z0-9_]{0,63}(?![\\s\\S])": {
              "discriminator": {
                "mapping": {
                  "bool": "#/definitions/BoolLiteral",
                  "device": "#/definitions/TextLiteral",
                  "dict": "#/definitions/MappingArgument",
                  "dtype": "#/definitions/TextLiteral",
                  "float": "#/definitions/FloatLiteral",
                  "int": "#/definitions/IntLiteral",
                  "layout": "#/definitions/TextLiteral",
                  "list": "#/definitions/SequenceArgument",
                  "memory_format": "#/definitions/TextLiteral",
                  "none": "#/definitions/NoneLiteral",
                  "opaque": "#/definitions/OpaqueArgument",
                  "str": "#/definitions/TextLiteral",
                  "symbolic": "#/definitions/SymbolicArgument",
                  "tuple": "#/definitions/SequenceArgument",
                  "value": "#/definitions/ValueRef"
                },
                "propertyName": "kind"
              },
              "oneOf": [
                {
                  "$ref": "#/definitions/ValueRef"
                },
                {
                  "$ref": "#/definitions/NoneLiteral"
                },
                {
                  "$ref": "#/definitions/BoolLiteral"
                },
                {
                  "$ref": "#/definitions/IntLiteral"
                },
                {
                  "$ref": "#/definitions/FloatLiteral"
                },
                {
                  "$ref": "#/definitions/TextLiteral"
                },
                {
                  "$ref": "#/definitions/SymbolicArgument"
                },
                {
                  "$ref": "#/definitions/OpaqueArgument"
                },
                {
                  "$ref": "#/definitions/SequenceArgument"
                },
                {
                  "$ref": "#/definitions/MappingArgument"
                }
              ]
            }
          },
          "title": "Kwargs",
          "type": "object"
        }
      },
      "required": [
        "args",
        "kwargs"
      ],
      "title": "Arguments",
      "type": "object"
    },
    "AutomaticCaptureRequest": {
      "additionalProperties": false,
      "description": "Existing automatic/declared-input request, not a resolved input recipe.",
      "properties": {
        "input": {
          "anyOf": [
            {
              "maxLength": 4096,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Input"
        },
        "overrides": {
          "anyOf": [
            {
              "additionalProperties": {
                "$ref": "#/definitions/JsonValue"
              },
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "title": "Overrides"
        },
        "tensorInputs": {
          "anyOf": [
            {
              "items": {
                "additionalProperties": {
                  "$ref": "#/definitions/JsonValue"
                },
                "type": "object"
              },
              "maxItems": 8,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "title": "Tensorinputs"
        }
      },
      "required": [
        "input",
        "tensorInputs",
        "overrides"
      ],
      "title": "AutomaticCaptureRequest",
      "type": "object"
    },
    "AutomaticCaptureSelection": {
      "additionalProperties": false,
      "properties": {
        "candidateIndex": {
          "maximum": 80,
          "minimum": 1,
          "title": "Candidateindex",
          "type": "integer"
        },
        "candidatesTried": {
          "maximum": 80,
          "minimum": 1,
          "title": "Candidatestried",
          "type": "integer"
        },
        "provenance": {
          "enum": [
            "explicit",
            "declared",
            "seed-from-first-layer",
            "search-candidate"
          ],
          "title": "Provenance",
          "type": "string"
        }
      },
      "required": [
        "provenance",
        "candidateIndex",
        "candidatesTried"
      ],
      "title": "AutomaticCaptureSelection",
      "type": "object"
    },
    "BoolLiteral": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "const": "bool",
          "title": "Kind",
          "type": "string"
        },
        "value": {
          "title": "Value",
          "type": "boolean"
        }
      },
      "required": [
        "kind",
        "value"
      ],
      "title": "BoolLiteral",
      "type": "object"
    },
    "CandidateExecution": {
      "additionalProperties": false,
      "description": "Private replay provenance; never a replacement for the live model capture.",
      "properties": {
        "afterDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Afterdigest",
          "type": "string"
        },
        "baseAttemptId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Baseattemptid",
          "type": "string"
        },
        "baseCaptureId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Basecaptureid",
          "type": "string"
        },
        "basePlanDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Baseplandigest",
          "type": "string"
        },
        "baseSourceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Basesourceid",
          "type": "string"
        },
        "baseWorkspaceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Baseworkspaceid",
          "type": "string"
        },
        "beforeDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Beforedigest",
          "type": "string"
        },
        "file": {
          "maxLength": 1024,
          "minLength": 1,
          "pattern": "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])",
          "title": "File",
          "type": "string"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "intentDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Intentdigest",
          "type": "string"
        },
        "plan": {
          "$ref": "#/definitions/ResolvedExecutionPlan"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "baseCaptureId",
        "baseAttemptId",
        "baseSourceId",
        "baseWorkspaceId",
        "basePlanDigest",
        "intentDigest",
        "file",
        "beforeDigest",
        "afterDigest",
        "plan"
      ],
      "title": "CandidateExecution",
      "type": "object"
    },
    "CandidatePostconditions": {
      "additionalProperties": false,
      "description": "Evidence for this captured expression only, never program equivalence.",
      "properties": {
        "bindingId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Bindingid"
        },
        "captureId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Captureid"
        },
        "configurationUses": {
          "default": [],
          "items": {
            "$ref": "#/definitions/ConfigurationArgumentUse"
          },
          "maxItems": 256,
          "title": "Configurationuses",
          "type": "array"
        },
        "executionId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Executionid",
          "type": "string"
        },
        "observedInitializers": {
          "default": 0,
          "maximum": 100000,
          "minimum": 0,
          "title": "Observedinitializers",
          "type": "integer"
        },
        "observedOperations": {
          "maximum": 100000,
          "minimum": 0,
          "title": "Observedoperations",
          "type": "integer"
        },
        "reason": {
          "enum": [
            "observed_requested_literal",
            "observed_literal_mismatch",
            "observed_requested_activation",
            "observed_activation_mismatch",
            "unsupported_recipe",
            "observed_requested_argument",
            "observed_argument_mismatch",
            "observed_requested_insertion",
            "observed_insertion_mismatch",
            "observed_requested_deletion",
            "observed_deletion_mismatch",
            "observed_requested_constructor",
            "observed_constructor_mismatch",
            "initializer_arguments_unavailable",
            "observed_requested_configuration",
            "observed_configuration_mismatch",
            "configuration_reference_unavailable",
            "capture_unavailable",
            "capture_incomplete",
            "capture_identity_mismatch",
            "source_mapping_unavailable",
            "expression_not_observed",
            "ambiguous_observation"
          ],
          "title": "Reason",
          "type": "string"
        },
        "recipe": {
          "anyOf": [
            {
              "enum": [
                "binary_scalar_v1",
                "functional_activation_v1",
                "module_activation_v1",
                "straight_line_tensor_v1",
                "straight_line_unary_insert_v1",
                "straight_line_unary_delete_v1",
                "constructor_scalar_v1",
                "configuration_scalar_v1"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Recipe"
        },
        "scope": {
          "enum": [
            "observed_expression",
            "observed_initializer_entries",
            "observed_configuration_arguments"
          ],
          "title": "Scope",
          "type": "string"
        },
        "status": {
          "enum": [
            "verified",
            "failed",
            "unavailable"
          ],
          "title": "Status",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "executionId",
        "captureId",
        "recipe",
        "scope",
        "status",
        "reason",
        "bindingId",
        "observedOperations"
      ],
      "title": "CandidatePostconditions",
      "type": "object"
    },
    "CandidateRequest": {
      "additionalProperties": false,
      "description": "Private host payload; a closed recipe plus the full reviewed artifact.",
      "properties": {
        "capture": {
          "$ref": "#/definitions/CaptureSnapshot"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "intent": {
          "discriminator": {
            "mapping": {
              "delete_unary_op": "#/definitions/DeleteUnaryEdit",
              "insert_unary_op": "#/definitions/InsertUnaryEdit",
              "reconnect_argument": "#/definitions/ReconnectArgumentEdit",
              "replace_activation": "#/definitions/ReplaceActivationEdit",
              "set_configuration_literal": "#/definitions/SetConfigurationLiteralEdit",
              "set_constructor_literal": "#/definitions/SetConstructorLiteralEdit",
              "set_literal": "#/definitions/SetLiteralEdit"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/SetLiteralEdit"
            },
            {
              "$ref": "#/definitions/ReplaceActivationEdit"
            },
            {
              "$ref": "#/definitions/SetConstructorLiteralEdit"
            },
            {
              "$ref": "#/definitions/SetConfigurationLiteralEdit"
            },
            {
              "$ref": "#/definitions/ReconnectArgumentEdit"
            },
            {
              "$ref": "#/definitions/InsertUnaryEdit"
            },
            {
              "$ref": "#/definitions/DeleteUnaryEdit"
            }
          ],
          "title": "Intent"
        },
        "plan": {
          "$ref": "#/definitions/ResolvedExecutionPlan"
        },
        "reviewedFile": {
          "maxLength": 1024,
          "minLength": 1,
          "pattern": "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])",
          "title": "Reviewedfile",
          "type": "string"
        },
        "reviewedSource": {
          "maxLength": 1048576,
          "title": "Reviewedsource",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "plan",
        "capture",
        "intent",
        "reviewedFile",
        "reviewedSource"
      ],
      "title": "CandidateRequest",
      "type": "object"
    },
    "CaptureCoverage": {
      "additionalProperties": false,
      "properties": {
        "invocationsSeen": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Invocationsseen",
          "type": "integer"
        },
        "modules": {
          "items": {
            "$ref": "#/definitions/CoverageEntry"
          },
          "maxItems": 100000,
          "title": "Modules",
          "type": "array"
        },
        "omissions": {
          "items": {
            "enum": [
              "operation_limit",
              "invocation_limit",
              "value_limit",
              "byte_limit",
              "depth_limit",
              "item_limit",
              "unsupported_type",
              "opaque_kernel",
              "recorder_error",
              "timeout",
              "cancelled"
            ],
            "type": "string"
          },
          "maxItems": 16,
          "title": "Omissions",
          "type": "array"
        },
        "operationsSeen": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Operationsseen",
          "type": "integer"
        },
        "scope": {
          "enum": [
            "executed_path",
            "traced_program"
          ],
          "title": "Scope",
          "type": "string"
        }
      },
      "required": [
        "scope",
        "operationsSeen",
        "invocationsSeen",
        "omissions",
        "modules"
      ],
      "title": "CaptureCoverage",
      "type": "object"
    },
    "CaptureLimits": {
      "additionalProperties": false,
      "properties": {
        "bytes": {
          "maximum": 67108864,
          "minimum": 1024,
          "title": "Bytes",
          "type": "integer"
        },
        "invocations": {
          "maximum": 100000,
          "minimum": 1,
          "title": "Invocations",
          "type": "integer"
        },
        "operations": {
          "maximum": 100000,
          "minimum": 1,
          "title": "Operations",
          "type": "integer"
        },
        "values": {
          "maximum": 100000,
          "minimum": 1,
          "title": "Values",
          "type": "integer"
        }
      },
      "required": [
        "operations",
        "invocations",
        "values",
        "bytes"
      ],
      "title": "CaptureLimits",
      "type": "object"
    },
    "CapturePreparation": {
      "additionalProperties": false,
      "description": "Saved facts before automatic inputs resolve; cannot impersonate a run plan.",
      "properties": {
        "captureLimits": {
          "$ref": "#/definitions/CaptureLimits"
        },
        "configuration": {
          "enum": [
            "absent",
            "no_match"
          ],
          "title": "Configuration",
          "type": "string"
        },
        "environment": {
          "$ref": "#/definitions/EnvironmentFact"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "interpreter": {
          "$ref": "#/definitions/InterpreterIdentity"
        },
        "policy": {
          "const": "legacy-input-search-v1",
          "title": "Policy",
          "type": "string"
        },
        "request": {
          "$ref": "#/definitions/AutomaticCaptureRequest"
        },
        "source": {
          "$ref": "#/definitions/SourceSnapshot"
        },
        "timeoutMs": {
          "maximum": 86400000,
          "minimum": 100,
          "title": "Timeoutms",
          "type": "integer"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "source",
        "interpreter",
        "environment",
        "configuration",
        "policy",
        "request",
        "timeoutMs",
        "captureLimits"
      ],
      "title": "CapturePreparation",
      "type": "object"
    },
    "CaptureProjection": {
      "additionalProperties": false,
      "properties": {
        "captureId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Captureid",
          "type": "string"
        },
        "collapsedInvocationIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 100000,
          "title": "Collapsedinvocationids",
          "type": "array"
        },
        "edges": {
          "items": {
            "$ref": "#/definitions/ProjectionEdge"
          },
          "maxItems": 1000000,
          "title": "Edges",
          "type": "array"
        },
        "nodes": {
          "items": {
            "$ref": "#/definitions/ProjectionNode"
          },
          "maxItems": 300002,
          "title": "Nodes",
          "type": "array"
        },
        "ports": {
          "items": {
            "$ref": "#/definitions/ProjectionPort"
          },
          "maxItems": 1000000,
          "title": "Ports",
          "type": "array"
        },
        "sourceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Sourceid",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "captureId",
        "sourceId",
        "collapsedInvocationIds",
        "nodes",
        "ports",
        "edges"
      ],
      "title": "CaptureProjection",
      "type": "object"
    },
    "CaptureReconciliation": {
      "additionalProperties": false,
      "description": "Observed source round trip, never freshness certification of an editor buffer.",
      "properties": {
        "baseCaptureId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Basecaptureid",
          "type": "string"
        },
        "captureId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Captureid",
          "type": "string"
        },
        "dependencyScope": {
          "enum": [
            "complete",
            "partial"
          ],
          "title": "Dependencyscope",
          "type": "string"
        },
        "inputs": {
          "enum": [
            "same",
            "different",
            "unavailable"
          ],
          "title": "Inputs",
          "type": "string"
        },
        "postconditions": {
          "anyOf": [
            {
              "$ref": "#/definitions/CandidatePostconditions"
            },
            {
              "type": "null"
            }
          ]
        },
        "reason": {
          "enum": [
            "reviewed_source",
            "source_reverted",
            "source_changed",
            "dependencies_changed",
            "environment_changed",
            "different_model",
            "capture_not_new",
            "source_unavailable"
          ],
          "title": "Reason",
          "type": "string"
        },
        "reviewedDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Revieweddigest",
          "type": "string"
        },
        "sourceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Sourceid",
          "type": "string"
        },
        "state": {
          "enum": [
            "same",
            "different",
            "unavailable"
          ],
          "title": "State",
          "type": "string"
        },
        "status": {
          "enum": [
            "checked",
            "unavailable"
          ],
          "title": "Status",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "baseCaptureId",
        "captureId",
        "sourceId",
        "reviewedDigest",
        "dependencyScope",
        "status",
        "reason",
        "inputs",
        "state",
        "postconditions"
      ],
      "title": "CaptureReconciliation",
      "type": "object"
    },
    "CaptureReconciliationRequest": {
      "additionalProperties": false,
      "description": "Private static check of a later live capture against the approved edit.",
      "properties": {
        "base": {
          "$ref": "#/definitions/CaptureSnapshot"
        },
        "current": {
          "$ref": "#/definitions/CaptureSnapshot"
        },
        "intent": {
          "discriminator": {
            "mapping": {
              "delete_unary_op": "#/definitions/DeleteUnaryEdit",
              "insert_unary_op": "#/definitions/InsertUnaryEdit",
              "reconnect_argument": "#/definitions/ReconnectArgumentEdit",
              "replace_activation": "#/definitions/ReplaceActivationEdit",
              "set_configuration_literal": "#/definitions/SetConfigurationLiteralEdit",
              "set_constructor_literal": "#/definitions/SetConstructorLiteralEdit",
              "set_literal": "#/definitions/SetLiteralEdit"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/SetLiteralEdit"
            },
            {
              "$ref": "#/definitions/ReplaceActivationEdit"
            },
            {
              "$ref": "#/definitions/SetConstructorLiteralEdit"
            },
            {
              "$ref": "#/definitions/SetConfigurationLiteralEdit"
            },
            {
              "$ref": "#/definitions/ReconnectArgumentEdit"
            },
            {
              "$ref": "#/definitions/InsertUnaryEdit"
            },
            {
              "$ref": "#/definitions/DeleteUnaryEdit"
            }
          ],
          "title": "Intent"
        },
        "reviewedFile": {
          "maxLength": 1024,
          "minLength": 1,
          "pattern": "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])",
          "title": "Reviewedfile",
          "type": "string"
        },
        "reviewedSource": {
          "maxLength": 1048576,
          "title": "Reviewedsource",
          "type": "string"
        },
        "sources": {
          "maxProperties": 128,
          "patternProperties": {
            "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])": {
              "type": "string"
            }
          },
          "propertyNames": {
            "maxLength": 1024,
            "minLength": 1
          },
          "title": "Sources",
          "type": "object"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "base",
        "sources",
        "intent",
        "reviewedFile",
        "reviewedSource",
        "current"
      ],
      "title": "CaptureReconciliationRequest",
      "type": "object"
    },
    "CaptureReference": {
      "additionalProperties": false,
      "properties": {
        "captureId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Captureid",
          "type": "string"
        },
        "kind": {
          "enum": [
            "operation",
            "invocation",
            "input",
            "output",
            "value"
          ],
          "title": "Kind",
          "type": "string"
        },
        "semanticId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Semanticid",
          "type": "string"
        },
        "sourceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Sourceid",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "captureId",
        "sourceId",
        "kind",
        "semanticId"
      ],
      "title": "CaptureReference",
      "type": "object"
    },
    "CaptureSnapshot": {
      "additionalProperties": false,
      "properties": {
        "attemptId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Attemptid",
          "type": "string"
        },
        "automatic": {
          "anyOf": [
            {
              "$ref": "#/definitions/AutomaticCaptureSelection"
            },
            {
              "type": "null"
            }
          ]
        },
        "backend": {
          "enum": [
            "runtime",
            "export",
            "jaxpr"
          ],
          "title": "Backend",
          "type": "string"
        },
        "backendVersion": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Backendversion",
          "type": "string"
        },
        "coverage": {
          "$ref": "#/definitions/CaptureCoverage"
        },
        "definitions": {
          "items": {
            "$ref": "#/definitions/ModuleDefinition"
          },
          "maxItems": 100000,
          "title": "Definitions",
          "type": "array"
        },
        "diagnostics": {
          "items": {
            "$ref": "#/definitions/Diagnostic"
          },
          "maxItems": 4096,
          "title": "Diagnostics",
          "type": "array"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "inputDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Inputdigest"
        },
        "inputs": {
          "$ref": "#/definitions/Arguments"
        },
        "invocations": {
          "items": {
            "$ref": "#/definitions/Invocation"
          },
          "maxItems": 100000,
          "title": "Invocations",
          "type": "array"
        },
        "jaxVersion": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Jaxversion"
        },
        "limits": {
          "$ref": "#/definitions/CaptureLimits"
        },
        "modules": {
          "items": {
            "$ref": "#/definitions/ModuleInstance"
          },
          "maxItems": 100000,
          "title": "Modules",
          "type": "array"
        },
        "normalizations": {
          "items": {
            "maxLength": 4096,
            "type": "string"
          },
          "maxItems": 64,
          "title": "Normalizations",
          "type": "array"
        },
        "operations": {
          "items": {
            "$ref": "#/definitions/Operation"
          },
          "maxItems": 100000,
          "title": "Operations",
          "type": "array"
        },
        "output": {
          "anyOf": [
            {
              "discriminator": {
                "mapping": {
                  "bool": "#/definitions/BoolLiteral",
                  "device": "#/definitions/TextLiteral",
                  "dict": "#/definitions/MappingArgument",
                  "dtype": "#/definitions/TextLiteral",
                  "float": "#/definitions/FloatLiteral",
                  "int": "#/definitions/IntLiteral",
                  "layout": "#/definitions/TextLiteral",
                  "list": "#/definitions/SequenceArgument",
                  "memory_format": "#/definitions/TextLiteral",
                  "none": "#/definitions/NoneLiteral",
                  "opaque": "#/definitions/OpaqueArgument",
                  "str": "#/definitions/TextLiteral",
                  "symbolic": "#/definitions/SymbolicArgument",
                  "tuple": "#/definitions/SequenceArgument",
                  "value": "#/definitions/ValueRef"
                },
                "propertyName": "kind"
              },
              "oneOf": [
                {
                  "$ref": "#/definitions/ValueRef"
                },
                {
                  "$ref": "#/definitions/NoneLiteral"
                },
                {
                  "$ref": "#/definitions/BoolLiteral"
                },
                {
                  "$ref": "#/definitions/IntLiteral"
                },
                {
                  "$ref": "#/definitions/FloatLiteral"
                },
                {
                  "$ref": "#/definitions/TextLiteral"
                },
                {
                  "$ref": "#/definitions/SymbolicArgument"
                },
                {
                  "$ref": "#/definitions/OpaqueArgument"
                },
                {
                  "$ref": "#/definitions/SequenceArgument"
                },
                {
                  "$ref": "#/definitions/MappingArgument"
                }
              ]
            },
            {
              "type": "null"
            }
          ],
          "title": "Output"
        },
        "planDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Plandigest"
        },
        "preparationDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Preparationdigest"
        },
        "rootInvocationId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Rootinvocationid"
        },
        "source": {
          "$ref": "#/definitions/SourceSnapshot"
        },
        "sourceBindings": {
          "items": {
            "$ref": "#/definitions/SourceBinding"
          },
          "maxItems": 100000,
          "title": "Sourcebindings",
          "type": "array"
        },
        "stateDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Statedigest"
        },
        "status": {
          "enum": [
            "complete",
            "partial",
            "failed"
          ],
          "title": "Status",
          "type": "string"
        },
        "torchVersion": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Torchversion"
        },
        "values": {
          "items": {
            "$ref": "#/definitions/Value"
          },
          "maxItems": 100000,
          "title": "Values",
          "type": "array"
        },
        "valuesUnavailable": {
          "anyOf": [
            {
              "const": "fake_tensor",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Valuesunavailable"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "attemptId",
        "source",
        "planDigest",
        "preparationDigest",
        "automatic",
        "inputDigest",
        "stateDigest",
        "backend",
        "backendVersion",
        "torchVersion",
        "normalizations",
        "status",
        "rootInvocationId",
        "inputs",
        "output",
        "definitions",
        "modules",
        "invocations",
        "operations",
        "values",
        "sourceBindings",
        "diagnostics",
        "coverage",
        "limits"
      ],
      "title": "CaptureSnapshot",
      "type": "object"
    },
    "CaptureState": {
      "additionalProperties": false,
      "properties": {
        "displayedCaptureId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Displayedcaptureid"
        },
        "lastCompleteCaptureId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Lastcompletecaptureid"
        },
        "lastPartialCaptureId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Lastpartialcaptureid"
        },
        "latestAttempt": {
          "anyOf": [
            {
              "$ref": "#/definitions/RunAttempt"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "latestAttempt",
        "lastCompleteCaptureId",
        "lastPartialCaptureId",
        "displayedCaptureId"
      ],
      "title": "CaptureState",
      "type": "object"
    },
    "ConfigurationArgumentUse": {
      "additionalProperties": false,
      "properties": {
        "bindingId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Bindingid",
          "type": "string"
        },
        "sourceArgumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Sourceargumentpath",
          "type": "array"
        }
      },
      "required": [
        "bindingId",
        "sourceArgumentPath"
      ],
      "title": "ConfigurationArgumentUse",
      "type": "object"
    },
    "ConfigurationEditScope": {
      "additionalProperties": false,
      "properties": {
        "affects": {
          "const": "all_references",
          "title": "Affects",
          "type": "string"
        },
        "includesUnobservedReferences": {
          "const": true,
          "title": "Includesunobservedreferences",
          "type": "boolean"
        },
        "kind": {
          "const": "configuration_definition",
          "title": "Kind",
          "type": "string"
        },
        "knownSourceReferences": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Knownsourcereferences",
          "type": "integer"
        },
        "observedConstructorSites": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Observedconstructorsites",
          "type": "integer"
        },
        "observedInvocations": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Observedinvocations",
          "type": "integer"
        },
        "observedModuleInstances": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Observedmoduleinstances",
          "type": "integer"
        },
        "observedOperations": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Observedoperations",
          "type": "integer"
        }
      },
      "required": [
        "kind",
        "affects",
        "knownSourceReferences",
        "observedConstructorSites",
        "observedOperations",
        "observedInvocations",
        "observedModuleInstances",
        "includesUnobservedReferences"
      ],
      "title": "ConfigurationEditScope",
      "type": "object"
    },
    "ConfigurationLiteralCapability": {
      "additionalProperties": false,
      "properties": {
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "command": {
          "const": "set_configuration_literal",
          "title": "Command",
          "type": "string"
        },
        "expected": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Expected"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "recipe": {
          "const": "configuration_scalar_v1",
          "title": "Recipe",
          "type": "string"
        },
        "reference": {
          "$ref": "#/definitions/SourceBinding"
        },
        "scope": {
          "$ref": "#/definitions/ConfigurationEditScope"
        },
        "sourceArgumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Sourceargumentpath",
          "type": "array"
        },
        "symbol": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Symbol",
          "type": "string"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "reference",
        "binding",
        "sourceArgumentPath",
        "symbol",
        "expected",
        "scope"
      ],
      "title": "ConfigurationLiteralCapability",
      "type": "object"
    },
    "ConstructorEditScope": {
      "additionalProperties": false,
      "properties": {
        "affects": {
          "const": "all_executions",
          "title": "Affects",
          "type": "string"
        },
        "includesUnobservedExecutions": {
          "const": true,
          "title": "Includesunobservedexecutions",
          "type": "boolean"
        },
        "kind": {
          "const": "source_expression",
          "title": "Kind",
          "type": "string"
        },
        "observedInvocations": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Observedinvocations",
          "type": "integer"
        },
        "observedModuleInstances": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Observedmoduleinstances",
          "type": "integer"
        },
        "observedOperations": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Observedoperations",
          "type": "integer"
        }
      },
      "required": [
        "kind",
        "affects",
        "observedOperations",
        "observedInvocations",
        "observedModuleInstances",
        "includesUnobservedExecutions"
      ],
      "title": "ConstructorEditScope",
      "type": "object"
    },
    "ConstructorLiteralCapability": {
      "additionalProperties": false,
      "properties": {
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "command": {
          "const": "set_constructor_literal",
          "title": "Command",
          "type": "string"
        },
        "constructor": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Constructor",
          "type": "string"
        },
        "expected": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Expected"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "recipe": {
          "const": "constructor_scalar_v1",
          "title": "Recipe",
          "type": "string"
        },
        "scope": {
          "$ref": "#/definitions/ConstructorEditScope"
        },
        "sourceArgumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Sourceargumentpath",
          "type": "array"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "binding",
        "sourceArgumentPath",
        "constructor",
        "expected",
        "scope"
      ],
      "title": "ConstructorLiteralCapability",
      "type": "object"
    },
    "CoverageEntry": {
      "additionalProperties": false,
      "properties": {
        "invocationIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 100000,
          "title": "Invocationids",
          "type": "array"
        },
        "modulePath": {
          "maxLength": 4096,
          "title": "Modulepath",
          "type": "string"
        },
        "reason": {
          "anyOf": [
            {
              "enum": [
                "operation_limit",
                "invocation_limit",
                "value_limit",
                "byte_limit",
                "depth_limit",
                "item_limit",
                "unsupported_type",
                "opaque_kernel",
                "recorder_error",
                "timeout",
                "cancelled"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Reason"
        },
        "status": {
          "enum": [
            "observed",
            "failed",
            "not_observed",
            "capture_limited",
            "static_only",
            "unresolved",
            "state"
          ],
          "title": "Status",
          "type": "string"
        }
      },
      "required": [
        "modulePath",
        "status",
        "invocationIds",
        "reason"
      ],
      "title": "CoverageEntry",
      "type": "object"
    },
    "DeleteUnaryCapability": {
      "additionalProperties": false,
      "properties": {
        "activation": {
          "enum": [
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh"
          ],
          "title": "Activation",
          "type": "string"
        },
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "command": {
          "const": "delete_unary_op",
          "title": "Command",
          "type": "string"
        },
        "expected": {
          "$ref": "#/definitions/ValueRef"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "input": {
          "$ref": "#/definitions/SourceValueChoice"
        },
        "recipe": {
          "const": "straight_line_unary_delete_v1",
          "title": "Recipe",
          "type": "string"
        },
        "scope": {
          "$ref": "#/definitions/SourceEditScope"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "binding",
        "argumentPath",
        "expected",
        "activation",
        "input",
        "scope"
      ],
      "title": "DeleteUnaryCapability",
      "type": "object"
    },
    "DeleteUnaryEdit": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "$ref": "#/definitions/ValueRef"
        },
        "kind": {
          "const": "delete_unary_op",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "source_expression",
          "title": "Scope",
          "type": "string"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "$ref": "#/definitions/ValueRef"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "argumentPath",
        "expected",
        "value",
        "scope"
      ],
      "title": "DeleteUnaryEdit",
      "type": "object"
    },
    "Diagnostic": {
      "additionalProperties": false,
      "properties": {
        "actualDtype": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Actualdtype"
        },
        "actualShape": {
          "anyOf": [
            {
              "items": {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              "maxItems": 32,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Actualshape"
        },
        "code": {
          "enum": [
            "invalid_source",
            "unresolved_import",
            "invalid_config",
            "unsaved",
            "missing_file",
            "missing_dependency",
            "unavailable_environment",
            "unavailable_device",
            "missing_argument",
            "unexpected_argument",
            "invalid_binding",
            "unknown_signature",
            "invalid_factory",
            "invalid_shape",
            "invalid_dtype",
            "assertion_failed",
            "execution_failed",
            "capture_limited",
            "timeout",
            "cancelled",
            "stale_result",
            "projection_failed",
            "serialization_failed",
            "transport_failed"
          ],
          "title": "Code",
          "type": "string"
        },
        "expectedDtype": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Expecteddtype"
        },
        "expectedShape": {
          "anyOf": [
            {
              "items": {
                "anyOf": [
                  {
                    "maximum": 9007199254740991,
                    "minimum": 0,
                    "type": "integer"
                  },
                  {
                    "type": "null"
                  }
                ]
              },
              "maxItems": 32,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Expectedshape"
        },
        "field": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Field",
          "type": "array"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "names": {
          "items": {
            "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}(?![\\s\\S])",
            "type": "string"
          },
          "maxItems": 64,
          "title": "Names",
          "type": "array"
        },
        "phase": {
          "enum": [
            "parse",
            "index",
            "configuration",
            "environment",
            "import",
            "construction",
            "binding",
            "materialization",
            "forward",
            "capture",
            "projection",
            "serialization",
            "transport"
          ],
          "title": "Phase",
          "type": "string"
        },
        "remedy": {
          "enum": [
            "fix_source",
            "save_files",
            "edit_profile",
            "select_environment",
            "review_dependencies",
            "provide_inputs",
            "provide_factory",
            "adjust_limits",
            "retry",
            "inspect_failure",
            "none"
          ],
          "title": "Remedy",
          "type": "string"
        },
        "sourceBindingId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Sourcebindingid"
        }
      },
      "required": [
        "id",
        "phase",
        "code",
        "field",
        "names",
        "sourceBindingId",
        "remedy"
      ],
      "title": "Diagnostic",
      "type": "object"
    },
    "DisplayPort": {
      "additionalProperties": false,
      "properties": {
        "argumentKind": {
          "enum": [
            "value",
            "none",
            "bool",
            "int",
            "float",
            "str",
            "dtype",
            "device",
            "layout",
            "memory_format",
            "symbolic",
            "opaque",
            "list",
            "tuple",
            "dict"
          ],
          "title": "Argumentkind",
          "type": "string"
        },
        "device": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Device"
        },
        "direction": {
          "enum": [
            "in",
            "out"
          ],
          "title": "Direction",
          "type": "string"
        },
        "displayName": {
          "anyOf": [
            {
              "maxLength": 4096,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Displayname"
        },
        "dtype": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Dtype"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "memberPortId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Memberportid"
        },
        "nodeId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Nodeid",
          "type": "string"
        },
        "paths": {
          "items": {
            "items": {
              "anyOf": [
                {
                  "maximum": 9007199254740991,
                  "minimum": 0,
                  "type": "integer"
                },
                {
                  "maxLength": 128,
                  "type": "string"
                }
              ]
            },
            "maxItems": 100,
            "type": "array"
          },
          "maxItems": 16384,
          "title": "Paths",
          "type": "array"
        },
        "shape": {
          "anyOf": [
            {
              "items": {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              "maxItems": 32,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "title": "Shape"
        },
        "valueId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Valueid"
        }
      },
      "required": [
        "id",
        "nodeId",
        "direction",
        "paths",
        "argumentKind",
        "valueId",
        "memberPortId",
        "shape",
        "dtype",
        "device"
      ],
      "title": "DisplayPort",
      "type": "object"
    },
    "EnvironmentFact": {
      "additionalProperties": false,
      "description": "Already-probed machine-local environment, without credentials or env vars.",
      "properties": {
        "executable": {
          "maxLength": 4096,
          "minLength": 1,
          "title": "Executable",
          "type": "string"
        },
        "probe_digest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Probe Digest",
          "type": "string"
        },
        "python_version": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Python Version",
          "type": "string"
        },
        "resolved_executable": {
          "maxLength": 4096,
          "minLength": 1,
          "title": "Resolved Executable",
          "type": "string"
        },
        "selection_revision": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Selection Revision",
          "type": "integer"
        },
        "torch_version": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Torch Version"
        }
      },
      "required": [
        "executable",
        "resolved_executable",
        "python_version",
        "torch_version",
        "selection_revision",
        "probe_digest"
      ],
      "title": "EnvironmentFact",
      "type": "object"
    },
    "FileIdentity": {
      "additionalProperties": false,
      "properties": {
        "digest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Digest",
          "type": "string"
        },
        "documentVersion": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "title": "Documentversion"
        },
        "path": {
          "maxLength": 1024,
          "minLength": 1,
          "pattern": "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])",
          "title": "Path",
          "type": "string"
        },
        "role": {
          "enum": [
            "config",
            "source",
            "native"
          ],
          "title": "Role",
          "type": "string"
        }
      },
      "required": [
        "path",
        "digest",
        "documentVersion",
        "role"
      ],
      "title": "FileIdentity",
      "type": "object"
    },
    "FloatLiteral": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "const": "float",
          "title": "Kind",
          "type": "string"
        },
        "value": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "enum": [
                "nan",
                "inf",
                "-inf",
                "-0"
              ],
              "type": "string"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "kind",
        "value"
      ],
      "title": "FloatLiteral",
      "type": "object"
    },
    "InitializerArguments": {
      "additionalProperties": false,
      "description": "Bound scalar parameters at entry, not authored call arguments or completion.",
      "properties": {
        "keywordOnly": {
          "default": [],
          "items": {
            "$ref": "#/definitions/InitializerParameter"
          },
          "maxItems": 256,
          "title": "Keywordonly",
          "type": "array"
        },
        "phase": {
          "const": "initializer_entry",
          "title": "Phase",
          "type": "string"
        },
        "positional": {
          "default": [],
          "items": {
            "$ref": "#/definitions/InitializerParameter"
          },
          "maxItems": 256,
          "title": "Positional",
          "type": "array"
        },
        "positionalOnly": {
          "default": 0,
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Positionalonly",
          "type": "integer"
        },
        "reason": {
          "anyOf": [
            {
              "enum": [
                "byte_limit",
                "item_limit",
                "unsupported_type"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Reason"
        },
        "status": {
          "enum": [
            "observed",
            "unavailable"
          ],
          "title": "Status",
          "type": "string"
        },
        "varArgs": {
          "anyOf": [
            {
              "items": {
                "discriminator": {
                  "mapping": {
                    "bool": "#/definitions/BoolLiteral",
                    "float": "#/definitions/FloatLiteral",
                    "int": "#/definitions/IntLiteral",
                    "none": "#/definitions/NoneLiteral",
                    "opaque": "#/definitions/OpaqueArgument"
                  },
                  "propertyName": "kind"
                },
                "oneOf": [
                  {
                    "$ref": "#/definitions/NoneLiteral"
                  },
                  {
                    "$ref": "#/definitions/BoolLiteral"
                  },
                  {
                    "$ref": "#/definitions/IntLiteral"
                  },
                  {
                    "$ref": "#/definitions/FloatLiteral"
                  },
                  {
                    "$ref": "#/definitions/OpaqueArgument"
                  }
                ]
              },
              "maxItems": 256,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Varargs"
        },
        "varKwargs": {
          "anyOf": [
            {
              "additionalProperties": {
                "discriminator": {
                  "mapping": {
                    "bool": "#/definitions/BoolLiteral",
                    "float": "#/definitions/FloatLiteral",
                    "int": "#/definitions/IntLiteral",
                    "none": "#/definitions/NoneLiteral",
                    "opaque": "#/definitions/OpaqueArgument"
                  },
                  "propertyName": "kind"
                },
                "oneOf": [
                  {
                    "$ref": "#/definitions/NoneLiteral"
                  },
                  {
                    "$ref": "#/definitions/BoolLiteral"
                  },
                  {
                    "$ref": "#/definitions/IntLiteral"
                  },
                  {
                    "$ref": "#/definitions/FloatLiteral"
                  },
                  {
                    "$ref": "#/definitions/OpaqueArgument"
                  }
                ]
              },
              "maxProperties": 256,
              "propertyNames": {
                "maxLength": 128,
                "minLength": 1
              },
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Varkwargs"
        }
      },
      "required": [
        "phase",
        "status"
      ],
      "title": "InitializerArguments",
      "type": "object"
    },
    "InitializerParameter": {
      "additionalProperties": false,
      "properties": {
        "name": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Name",
          "type": "string"
        },
        "value": {
          "discriminator": {
            "mapping": {
              "bool": "#/definitions/BoolLiteral",
              "float": "#/definitions/FloatLiteral",
              "int": "#/definitions/IntLiteral",
              "none": "#/definitions/NoneLiteral",
              "opaque": "#/definitions/OpaqueArgument"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/NoneLiteral"
            },
            {
              "$ref": "#/definitions/BoolLiteral"
            },
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/FloatLiteral"
            },
            {
              "$ref": "#/definitions/OpaqueArgument"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "name",
        "value"
      ],
      "title": "InitializerParameter",
      "type": "object"
    },
    "InsertUnaryCapability": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "command": {
          "const": "insert_unary_op",
          "title": "Command",
          "type": "string"
        },
        "expected": {
          "$ref": "#/definitions/ValueRef"
        },
        "expectedName": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Expectedname",
          "type": "string"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "importPlan": {
          "anyOf": [
            {
              "$ref": "#/definitions/ActivationImportPlan"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "namespace": {
          "maxLength": 256,
          "minLength": 1,
          "pattern": "^[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*)*(?![\\s\\S])",
          "title": "Namespace",
          "type": "string"
        },
        "recipe": {
          "const": "straight_line_unary_insert_v1",
          "title": "Recipe",
          "type": "string"
        },
        "scope": {
          "$ref": "#/definitions/SourceEditScope"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "binding",
        "argumentPath",
        "expected",
        "expectedName",
        "namespace",
        "scope"
      ],
      "title": "InsertUnaryCapability",
      "type": "object"
    },
    "InsertUnaryEdit": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "$ref": "#/definitions/ValueRef"
        },
        "kind": {
          "const": "insert_unary_op",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "source_expression",
          "title": "Scope",
          "type": "string"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "enum": [
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh"
          ],
          "title": "Value",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "argumentPath",
        "expected",
        "value",
        "scope"
      ],
      "title": "InsertUnaryEdit",
      "type": "object"
    },
    "IntLiteral": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "const": "int",
          "title": "Kind",
          "type": "string"
        },
        "value": {
          "maxLength": 4096,
          "pattern": "^(?:0|-?[1-9][0-9]*)(?![\\s\\S])",
          "title": "Value",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "value"
      ],
      "title": "IntLiteral",
      "type": "object"
    },
    "InterpreterIdentity": {
      "additionalProperties": false,
      "properties": {
        "environmentDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Environmentdigest",
          "type": "string"
        },
        "executable": {
          "maxLength": 4096,
          "minLength": 1,
          "title": "Executable",
          "type": "string"
        },
        "version": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Version",
          "type": "string"
        }
      },
      "required": [
        "executable",
        "version",
        "environmentDigest"
      ],
      "title": "InterpreterIdentity",
      "type": "object"
    },
    "Invocation": {
      "additionalProperties": false,
      "properties": {
        "arguments": {
          "$ref": "#/definitions/Arguments"
        },
        "callSiteDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Callsitedigest"
        },
        "diagnosticId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Diagnosticid"
        },
        "endSequence": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "title": "Endsequence"
        },
        "gradEnabled": {
          "title": "Gradenabled",
          "type": "boolean"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "moduleId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Moduleid",
          "type": "string"
        },
        "output": {
          "anyOf": [
            {
              "discriminator": {
                "mapping": {
                  "bool": "#/definitions/BoolLiteral",
                  "device": "#/definitions/TextLiteral",
                  "dict": "#/definitions/MappingArgument",
                  "dtype": "#/definitions/TextLiteral",
                  "float": "#/definitions/FloatLiteral",
                  "int": "#/definitions/IntLiteral",
                  "layout": "#/definitions/TextLiteral",
                  "list": "#/definitions/SequenceArgument",
                  "memory_format": "#/definitions/TextLiteral",
                  "none": "#/definitions/NoneLiteral",
                  "opaque": "#/definitions/OpaqueArgument",
                  "str": "#/definitions/TextLiteral",
                  "symbolic": "#/definitions/SymbolicArgument",
                  "tuple": "#/definitions/SequenceArgument",
                  "value": "#/definitions/ValueRef"
                },
                "propertyName": "kind"
              },
              "oneOf": [
                {
                  "$ref": "#/definitions/ValueRef"
                },
                {
                  "$ref": "#/definitions/NoneLiteral"
                },
                {
                  "$ref": "#/definitions/BoolLiteral"
                },
                {
                  "$ref": "#/definitions/IntLiteral"
                },
                {
                  "$ref": "#/definitions/FloatLiteral"
                },
                {
                  "$ref": "#/definitions/TextLiteral"
                },
                {
                  "$ref": "#/definitions/SymbolicArgument"
                },
                {
                  "$ref": "#/definitions/OpaqueArgument"
                },
                {
                  "$ref": "#/definitions/SequenceArgument"
                },
                {
                  "$ref": "#/definitions/MappingArgument"
                }
              ]
            },
            {
              "type": "null"
            }
          ],
          "title": "Output"
        },
        "parameterCount": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Parametercount"
        },
        "parameterCountUnavailable": {
          "anyOf": [
            {
              "enum": [
                "item_limit",
                "unsupported_type"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Parametercountunavailable"
        },
        "parentId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Parentid"
        },
        "sequence": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Sequence",
          "type": "integer"
        },
        "sourceBindingIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 64,
          "title": "Sourcebindingids",
          "type": "array"
        },
        "status": {
          "enum": [
            "completed",
            "failed",
            "interrupted"
          ],
          "title": "Status",
          "type": "string"
        },
        "training": {
          "title": "Training",
          "type": "boolean"
        }
      },
      "required": [
        "id",
        "moduleId",
        "parentId",
        "sequence",
        "endSequence",
        "arguments",
        "output",
        "status",
        "diagnosticId",
        "training",
        "gradEnabled",
        "sourceBindingIds"
      ],
      "title": "Invocation",
      "type": "object"
    },
    "JsonValue": {},
    "LiteralEditCapability": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "command": {
          "const": "set_literal",
          "title": "Command",
          "type": "string"
        },
        "expected": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Expected"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "recipe": {
          "const": "binary_scalar_v1",
          "title": "Recipe",
          "type": "string"
        },
        "scope": {
          "$ref": "#/definitions/SourceEditScope"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "binding",
        "argumentPath",
        "expected",
        "scope"
      ],
      "title": "LiteralEditCapability",
      "type": "object"
    },
    "MappingArgument": {
      "additionalProperties": false,
      "properties": {
        "entries": {
          "items": {
            "$ref": "#/definitions/MappingEntry"
          },
          "maxItems": 4096,
          "title": "Entries",
          "type": "array"
        },
        "kind": {
          "const": "dict",
          "title": "Kind",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "entries"
      ],
      "title": "MappingArgument",
      "type": "object"
    },
    "MappingEntry": {
      "additionalProperties": false,
      "properties": {
        "key": {
          "discriminator": {
            "mapping": {
              "bool": "#/definitions/BoolLiteral",
              "device": "#/definitions/TextLiteral",
              "dtype": "#/definitions/TextLiteral",
              "int": "#/definitions/IntLiteral",
              "layout": "#/definitions/TextLiteral",
              "memory_format": "#/definitions/TextLiteral",
              "none": "#/definitions/NoneLiteral",
              "str": "#/definitions/TextLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/NoneLiteral"
            },
            {
              "$ref": "#/definitions/BoolLiteral"
            },
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/TextLiteral"
            }
          ],
          "title": "Key"
        },
        "value": {
          "discriminator": {
            "mapping": {
              "bool": "#/definitions/BoolLiteral",
              "device": "#/definitions/TextLiteral",
              "dict": "#/definitions/MappingArgument",
              "dtype": "#/definitions/TextLiteral",
              "float": "#/definitions/FloatLiteral",
              "int": "#/definitions/IntLiteral",
              "layout": "#/definitions/TextLiteral",
              "list": "#/definitions/SequenceArgument",
              "memory_format": "#/definitions/TextLiteral",
              "none": "#/definitions/NoneLiteral",
              "opaque": "#/definitions/OpaqueArgument",
              "str": "#/definitions/TextLiteral",
              "symbolic": "#/definitions/SymbolicArgument",
              "tuple": "#/definitions/SequenceArgument",
              "value": "#/definitions/ValueRef"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/ValueRef"
            },
            {
              "$ref": "#/definitions/NoneLiteral"
            },
            {
              "$ref": "#/definitions/BoolLiteral"
            },
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/FloatLiteral"
            },
            {
              "$ref": "#/definitions/TextLiteral"
            },
            {
              "$ref": "#/definitions/SymbolicArgument"
            },
            {
              "$ref": "#/definitions/OpaqueArgument"
            },
            {
              "$ref": "#/definitions/SequenceArgument"
            },
            {
              "$ref": "#/definitions/MappingArgument"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "key",
        "value"
      ],
      "title": "MappingEntry",
      "type": "object"
    },
    "ModuleDefinition": {
      "additionalProperties": false,
      "properties": {
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "qualifiedName": {
          "maxLength": 512,
          "minLength": 1,
          "title": "Qualifiedname",
          "type": "string"
        },
        "sourceBindingIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 64,
          "title": "Sourcebindingids",
          "type": "array"
        }
      },
      "required": [
        "id",
        "qualifiedName",
        "sourceBindingIds"
      ],
      "title": "ModuleDefinition",
      "type": "object"
    },
    "ModuleInstance": {
      "additionalProperties": false,
      "properties": {
        "constructorBindingIds": {
          "default": [],
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 64,
          "title": "Constructorbindingids",
          "type": "array"
        },
        "constructorClassVerified": {
          "default": false,
          "title": "Constructorclassverified",
          "type": "boolean"
        },
        "definitionId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Definitionid",
          "type": "string"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "initializerArguments": {
          "anyOf": [
            {
              "$ref": "#/definitions/InitializerArguments"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "paths": {
          "items": {
            "maxLength": 4096,
            "type": "string"
          },
          "maxItems": 4096,
          "title": "Paths",
          "type": "array"
        }
      },
      "required": [
        "id",
        "definitionId",
        "paths"
      ],
      "title": "ModuleInstance",
      "type": "object"
    },
    "MutationEffect": {
      "additionalProperties": false,
      "properties": {
        "after": {
          "discriminator": {
            "mapping": {
              "opaque": "#/definitions/OpaqueArgument",
              "value": "#/definitions/ValueRef"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/ValueRef"
            },
            {
              "$ref": "#/definitions/OpaqueArgument"
            }
          ],
          "title": "After"
        },
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "beforeId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Beforeid",
          "type": "string"
        },
        "kind": {
          "enum": [
            "write",
            "alias_update"
          ],
          "title": "Kind",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "argumentPath",
        "beforeId",
        "after"
      ],
      "title": "MutationEffect",
      "type": "object"
    },
    "NoneLiteral": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "const": "none",
          "title": "Kind",
          "type": "string"
        }
      },
      "required": [
        "kind"
      ],
      "title": "NoneLiteral",
      "type": "object"
    },
    "OpaqueArgument": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "const": "opaque",
          "title": "Kind",
          "type": "string"
        },
        "reason": {
          "enum": [
            "operation_limit",
            "invocation_limit",
            "value_limit",
            "byte_limit",
            "depth_limit",
            "item_limit",
            "unsupported_type",
            "opaque_kernel",
            "recorder_error",
            "timeout",
            "cancelled"
          ],
          "title": "Reason",
          "type": "string"
        },
        "typeName": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Typename"
        }
      },
      "required": [
        "kind",
        "reason",
        "typeName"
      ],
      "title": "OpaqueArgument",
      "type": "object"
    },
    "Operation": {
      "additionalProperties": false,
      "properties": {
        "arguments": {
          "$ref": "#/definitions/Arguments"
        },
        "diagnosticId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Diagnosticid"
        },
        "effects": {
          "items": {
            "$ref": "#/definitions/MutationEffect"
          },
          "maxItems": 16384,
          "title": "Effects",
          "type": "array"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "invocationId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Invocationid"
        },
        "output": {
          "anyOf": [
            {
              "discriminator": {
                "mapping": {
                  "bool": "#/definitions/BoolLiteral",
                  "device": "#/definitions/TextLiteral",
                  "dict": "#/definitions/MappingArgument",
                  "dtype": "#/definitions/TextLiteral",
                  "float": "#/definitions/FloatLiteral",
                  "int": "#/definitions/IntLiteral",
                  "layout": "#/definitions/TextLiteral",
                  "list": "#/definitions/SequenceArgument",
                  "memory_format": "#/definitions/TextLiteral",
                  "none": "#/definitions/NoneLiteral",
                  "opaque": "#/definitions/OpaqueArgument",
                  "str": "#/definitions/TextLiteral",
                  "symbolic": "#/definitions/SymbolicArgument",
                  "tuple": "#/definitions/SequenceArgument",
                  "value": "#/definitions/ValueRef"
                },
                "propertyName": "kind"
              },
              "oneOf": [
                {
                  "$ref": "#/definitions/ValueRef"
                },
                {
                  "$ref": "#/definitions/NoneLiteral"
                },
                {
                  "$ref": "#/definitions/BoolLiteral"
                },
                {
                  "$ref": "#/definitions/IntLiteral"
                },
                {
                  "$ref": "#/definitions/FloatLiteral"
                },
                {
                  "$ref": "#/definitions/TextLiteral"
                },
                {
                  "$ref": "#/definitions/SymbolicArgument"
                },
                {
                  "$ref": "#/definitions/OpaqueArgument"
                },
                {
                  "$ref": "#/definitions/SequenceArgument"
                },
                {
                  "$ref": "#/definitions/MappingArgument"
                }
              ]
            },
            {
              "type": "null"
            }
          ],
          "title": "Output"
        },
        "schemaSignature": {
          "anyOf": [
            {
              "maxLength": 4096,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Schemasignature"
        },
        "sequence": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Sequence",
          "type": "integer"
        },
        "sourceBindingIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 64,
          "title": "Sourcebindingids",
          "type": "array"
        },
        "status": {
          "enum": [
            "completed",
            "failed",
            "interrupted"
          ],
          "title": "Status",
          "type": "string"
        },
        "target": {
          "maxLength": 512,
          "minLength": 1,
          "title": "Target",
          "type": "string"
        }
      },
      "required": [
        "id",
        "sequence",
        "invocationId",
        "target",
        "schemaSignature",
        "arguments",
        "output",
        "effects",
        "status",
        "diagnosticId",
        "sourceBindingIds"
      ],
      "title": "Operation",
      "type": "object"
    },
    "OutputRef": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "enum": [
            "output",
            "effect"
          ],
          "title": "Kind",
          "type": "string"
        },
        "operationId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Operationid",
          "type": "string"
        },
        "path": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Path",
          "type": "array"
        }
      },
      "required": [
        "operationId",
        "kind",
        "path"
      ],
      "title": "OutputRef",
      "type": "object"
    },
    "PortableCapture": {
      "additionalProperties": false,
      "properties": {
        "capture": {
          "$ref": "#/definitions/CaptureSnapshot"
        },
        "sourceNames": {
          "const": "anonymized",
          "title": "Sourcenames",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "sourceNames",
        "capture"
      ],
      "title": "PortableCapture",
      "type": "object"
    },
    "ProjectionEdge": {
      "additionalProperties": false,
      "properties": {
        "displaySourcePortId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Displaysourceportid"
        },
        "displayTargetPortId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Displaytargetportid"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "kind": {
          "enum": [
            "argument",
            "effect",
            "result"
          ],
          "title": "Kind",
          "type": "string"
        },
        "sourcePortId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Sourceportid",
          "type": "string"
        },
        "targetPortId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Targetportid",
          "type": "string"
        },
        "valueId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Valueid",
          "type": "string"
        }
      },
      "required": [
        "id",
        "kind",
        "valueId",
        "sourcePortId",
        "targetPortId",
        "displaySourcePortId",
        "displayTargetPortId"
      ],
      "title": "ProjectionEdge",
      "type": "object"
    },
    "ProjectionNode": {
      "additionalProperties": false,
      "properties": {
        "collapsed": {
          "title": "Collapsed",
          "type": "boolean"
        },
        "hiddenOperationCount": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Hiddenoperationcount",
          "type": "integer"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "kind": {
          "enum": [
            "operation",
            "invocation",
            "input",
            "output",
            "value"
          ],
          "title": "Kind",
          "type": "string"
        },
        "parentId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Parentid"
        },
        "semanticId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Semanticid",
          "type": "string"
        },
        "visible": {
          "title": "Visible",
          "type": "boolean"
        }
      },
      "required": [
        "id",
        "kind",
        "semanticId",
        "parentId",
        "visible",
        "collapsed",
        "hiddenOperationCount"
      ],
      "title": "ProjectionNode",
      "type": "object"
    },
    "ProjectionPort": {
      "additionalProperties": false,
      "properties": {
        "argumentKind": {
          "enum": [
            "value",
            "none",
            "bool",
            "int",
            "float",
            "str",
            "dtype",
            "device",
            "layout",
            "memory_format",
            "symbolic",
            "opaque",
            "list",
            "tuple",
            "dict"
          ],
          "title": "Argumentkind",
          "type": "string"
        },
        "direction": {
          "enum": [
            "in",
            "out"
          ],
          "title": "Direction",
          "type": "string"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "memberPortId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Memberportid"
        },
        "nodeId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Nodeid",
          "type": "string"
        },
        "paths": {
          "items": {
            "items": {
              "anyOf": [
                {
                  "maximum": 9007199254740991,
                  "minimum": 0,
                  "type": "integer"
                },
                {
                  "maxLength": 128,
                  "type": "string"
                }
              ]
            },
            "maxItems": 100,
            "type": "array"
          },
          "maxItems": 16384,
          "title": "Paths",
          "type": "array"
        },
        "valueId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Valueid"
        }
      },
      "required": [
        "id",
        "nodeId",
        "direction",
        "paths",
        "argumentKind",
        "valueId",
        "memberPortId"
      ],
      "title": "ProjectionPort",
      "type": "object"
    },
    "ReconnectArgumentCapability": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "choices": {
          "items": {
            "$ref": "#/definitions/SourceValueChoice"
          },
          "maxItems": 128,
          "minItems": 1,
          "title": "Choices",
          "type": "array"
        },
        "command": {
          "const": "reconnect_argument",
          "title": "Command",
          "type": "string"
        },
        "expected": {
          "$ref": "#/definitions/ValueRef"
        },
        "expectedName": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Expectedname",
          "type": "string"
        },
        "expression": {
          "$ref": "#/definitions/SourceBinding"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "recipe": {
          "const": "straight_line_tensor_v1",
          "title": "Recipe",
          "type": "string"
        },
        "scope": {
          "$ref": "#/definitions/SourceEditScope"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "command",
        "recipe",
        "target",
        "expression",
        "binding",
        "argumentPath",
        "expected",
        "expectedName",
        "choices",
        "scope"
      ],
      "title": "ReconnectArgumentCapability",
      "type": "object"
    },
    "ReconnectArgumentEdit": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "$ref": "#/definitions/ValueRef"
        },
        "kind": {
          "const": "reconnect_argument",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "source_expression",
          "title": "Scope",
          "type": "string"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "$ref": "#/definitions/ValueRef"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "argumentPath",
        "expected",
        "value",
        "scope"
      ],
      "title": "ReconnectArgumentEdit",
      "type": "object"
    },
    "ReplaceActivationEdit": {
      "additionalProperties": false,
      "properties": {
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "enum": [
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh"
          ],
          "title": "Expected",
          "type": "string"
        },
        "kind": {
          "const": "replace_activation",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "source_expression",
          "title": "Scope",
          "type": "string"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "enum": [
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh"
          ],
          "title": "Value",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "expected",
        "value",
        "scope"
      ],
      "title": "ReplaceActivationEdit",
      "type": "object"
    },
    "ResolvedExecutionPlan": {
      "additionalProperties": false,
      "description": "Host-resolved saved invocation. ExecutionSpec remains the recipe authority.\n\nAutomatic guesses are not resolved plans. The recipe's nested inputs retain\nvalidation.Binding or explicit tensor/fixture descriptions, never observations.",
      "properties": {
        "captureLimits": {
          "$ref": "#/definitions/CaptureLimits"
        },
        "configuration": {
          "enum": [
            "absent",
            "selected",
            "overridden"
          ],
          "title": "Configuration",
          "type": "string"
        },
        "environment": {
          "$ref": "#/definitions/EnvironmentFact"
        },
        "execution": {
          "additionalProperties": {
            "$ref": "#/definitions/JsonValue"
          },
          "title": "Execution",
          "type": "object"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "interpreter": {
          "$ref": "#/definitions/InterpreterIdentity"
        },
        "job": {
          "anyOf": [
            {
              "additionalProperties": {
                "$ref": "#/definitions/JsonValue"
              },
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "title": "Job"
        },
        "origin": {
          "enum": [
            "profile",
            "explicit"
          ],
          "title": "Origin",
          "type": "string"
        },
        "profileDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Profiledigest"
        },
        "profileName": {
          "anyOf": [
            {
              "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Profilename"
        },
        "source": {
          "$ref": "#/definitions/SourceSnapshot"
        },
        "timeoutMs": {
          "maximum": 86400000,
          "minimum": 100,
          "title": "Timeoutms",
          "type": "integer"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "source",
        "interpreter",
        "environment",
        "origin",
        "configuration",
        "profileName",
        "profileDigest",
        "execution",
        "job",
        "timeoutMs",
        "captureLimits"
      ],
      "title": "ResolvedExecutionPlan",
      "type": "object"
    },
    "RunAttempt": {
      "additionalProperties": false,
      "properties": {
        "captureIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 8,
          "title": "Captureids",
          "type": "array"
        },
        "diagnostics": {
          "items": {
            "$ref": "#/definitions/Diagnostic"
          },
          "maxItems": 256,
          "title": "Diagnostics",
          "type": "array"
        },
        "finishedAt": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "title": "Finishedat"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "origin": {
          "enum": [
            "manual",
            "mcp",
            "automatic"
          ],
          "title": "Origin",
          "type": "string"
        },
        "phase": {
          "enum": [
            "parse",
            "index",
            "configuration",
            "environment",
            "import",
            "construction",
            "binding",
            "materialization",
            "forward",
            "capture",
            "projection",
            "serialization",
            "transport"
          ],
          "title": "Phase",
          "type": "string"
        },
        "planDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Plandigest"
        },
        "requestDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Requestdigest",
          "type": "string"
        },
        "reusedCaptureId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Reusedcaptureid"
        },
        "sourceId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Sourceid"
        },
        "startedAt": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Startedat",
          "type": "integer"
        },
        "status": {
          "enum": [
            "requested",
            "running",
            "needs_setup",
            "succeeded",
            "failed",
            "cancelled",
            "timed_out"
          ],
          "title": "Status",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        }
      },
      "required": [
        "version",
        "id",
        "origin",
        "requestDigest",
        "sourceId",
        "planDigest",
        "phase",
        "status",
        "startedAt",
        "finishedAt",
        "captureIds",
        "diagnostics"
      ],
      "title": "RunAttempt",
      "type": "object"
    },
    "SequenceArgument": {
      "additionalProperties": false,
      "properties": {
        "items": {
          "items": {
            "discriminator": {
              "mapping": {
                "bool": "#/definitions/BoolLiteral",
                "device": "#/definitions/TextLiteral",
                "dict": "#/definitions/MappingArgument",
                "dtype": "#/definitions/TextLiteral",
                "float": "#/definitions/FloatLiteral",
                "int": "#/definitions/IntLiteral",
                "layout": "#/definitions/TextLiteral",
                "list": "#/definitions/SequenceArgument",
                "memory_format": "#/definitions/TextLiteral",
                "none": "#/definitions/NoneLiteral",
                "opaque": "#/definitions/OpaqueArgument",
                "str": "#/definitions/TextLiteral",
                "symbolic": "#/definitions/SymbolicArgument",
                "tuple": "#/definitions/SequenceArgument",
                "value": "#/definitions/ValueRef"
              },
              "propertyName": "kind"
            },
            "oneOf": [
              {
                "$ref": "#/definitions/ValueRef"
              },
              {
                "$ref": "#/definitions/NoneLiteral"
              },
              {
                "$ref": "#/definitions/BoolLiteral"
              },
              {
                "$ref": "#/definitions/IntLiteral"
              },
              {
                "$ref": "#/definitions/FloatLiteral"
              },
              {
                "$ref": "#/definitions/TextLiteral"
              },
              {
                "$ref": "#/definitions/SymbolicArgument"
              },
              {
                "$ref": "#/definitions/OpaqueArgument"
              },
              {
                "$ref": "#/definitions/SequenceArgument"
              },
              {
                "$ref": "#/definitions/MappingArgument"
              }
            ]
          },
          "maxItems": 4096,
          "title": "Items",
          "type": "array"
        },
        "kind": {
          "enum": [
            "list",
            "tuple"
          ],
          "title": "Kind",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "items"
      ],
      "title": "SequenceArgument",
      "type": "object"
    },
    "SetConfigurationLiteralEdit": {
      "additionalProperties": false,
      "properties": {
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Expected"
        },
        "kind": {
          "const": "set_configuration_literal",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "configuration_definition",
          "title": "Scope",
          "type": "string"
        },
        "sourceArgumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Sourceargumentpath",
          "type": "array"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "sourceArgumentPath",
        "expected",
        "value",
        "scope"
      ],
      "title": "SetConfigurationLiteralEdit",
      "type": "object"
    },
    "SetConstructorLiteralEdit": {
      "additionalProperties": false,
      "properties": {
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Expected"
        },
        "kind": {
          "const": "set_constructor_literal",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "source_expression",
          "title": "Scope",
          "type": "string"
        },
        "sourceArgumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Sourceargumentpath",
          "type": "array"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "sourceArgumentPath",
        "expected",
        "value",
        "scope"
      ],
      "title": "SetConstructorLiteralEdit",
      "type": "object"
    },
    "SetLiteralEdit": {
      "additionalProperties": false,
      "properties": {
        "argumentPath": {
          "items": {
            "anyOf": [
              {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              {
                "maxLength": 128,
                "type": "string"
              }
            ]
          },
          "maxItems": 64,
          "title": "Argumentpath",
          "type": "array"
        },
        "capabilityId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Capabilityid",
          "type": "string"
        },
        "expected": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Expected"
        },
        "kind": {
          "const": "set_literal",
          "title": "Kind",
          "type": "string"
        },
        "scope": {
          "const": "source_expression",
          "title": "Scope",
          "type": "string"
        },
        "target": {
          "$ref": "#/definitions/CaptureReference"
        },
        "value": {
          "discriminator": {
            "mapping": {
              "float": "#/definitions/SourceFloatLiteral",
              "int": "#/definitions/IntLiteral"
            },
            "propertyName": "kind"
          },
          "oneOf": [
            {
              "$ref": "#/definitions/IntLiteral"
            },
            {
              "$ref": "#/definitions/SourceFloatLiteral"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "kind",
        "capabilityId",
        "target",
        "argumentPath",
        "expected",
        "value",
        "scope"
      ],
      "title": "SetLiteralEdit",
      "type": "object"
    },
    "SourceAnchor": {
      "additionalProperties": false,
      "properties": {
        "digest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Digest",
          "type": "string"
        },
        "endColumn": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Endcolumn",
          "type": "integer"
        },
        "endLine": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Endline",
          "type": "integer"
        },
        "file": {
          "maxLength": 1024,
          "minLength": 1,
          "pattern": "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])",
          "title": "File",
          "type": "string"
        },
        "scope": {
          "maxLength": 4096,
          "title": "Scope",
          "type": "string"
        },
        "startColumn": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Startcolumn",
          "type": "integer"
        },
        "startLine": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Startline",
          "type": "integer"
        },
        "tokenDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Tokendigest",
          "type": "string"
        }
      },
      "required": [
        "file",
        "digest",
        "startLine",
        "startColumn",
        "endLine",
        "endColumn",
        "scope",
        "tokenDigest"
      ],
      "title": "SourceAnchor",
      "type": "object"
    },
    "SourceBinding": {
      "additionalProperties": false,
      "properties": {
        "anchors": {
          "items": {
            "$ref": "#/definitions/SourceAnchor"
          },
          "maxItems": 64,
          "minItems": 1,
          "title": "Anchors",
          "type": "array"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "kind": {
          "enum": [
            "definition",
            "constructor",
            "call",
            "argument",
            "configuration"
          ],
          "title": "Kind",
          "type": "string"
        },
        "quality": {
          "enum": [
            "exact",
            "candidate",
            "ambiguous"
          ],
          "title": "Quality",
          "type": "string"
        },
        "snapshotId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Snapshotid",
          "type": "string"
        }
      },
      "required": [
        "id",
        "snapshotId",
        "kind",
        "quality",
        "anchors"
      ],
      "title": "SourceBinding",
      "type": "object"
    },
    "SourceCorrespondence": {
      "additionalProperties": false,
      "description": "Revision-specific evidence for a source card, not proof that an expression ran.",
      "properties": {
        "basis": {
          "enum": [
            "registered_path",
            "shared_instance",
            "state_reference",
            "no_mapping"
          ],
          "title": "Basis",
          "type": "string"
        },
        "captureId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Captureid",
          "type": "string"
        },
        "invocationIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 100000,
          "title": "Invocationids",
          "type": "array"
        },
        "moduleId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Moduleid"
        },
        "nodeId": {
          "maxLength": 4096,
          "title": "Nodeid",
          "type": "string"
        },
        "reason": {
          "anyOf": [
            {
              "enum": [
                "operation_limit",
                "invocation_limit",
                "value_limit",
                "byte_limit",
                "depth_limit",
                "item_limit",
                "unsupported_type",
                "opaque_kernel",
                "recorder_error",
                "timeout",
                "cancelled"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Reason"
        },
        "sourceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Sourceid",
          "type": "string"
        },
        "stateValueIds": {
          "items": {
            "maxLength": 128,
            "minLength": 1,
            "type": "string"
          },
          "maxItems": 100000,
          "title": "Statevalueids",
          "type": "array"
        },
        "status": {
          "enum": [
            "observed",
            "failed",
            "not_observed",
            "capture_limited",
            "static_only",
            "unresolved",
            "state"
          ],
          "title": "Status",
          "type": "string"
        }
      },
      "required": [
        "nodeId",
        "captureId",
        "sourceId",
        "status",
        "basis",
        "moduleId",
        "invocationIds",
        "stateValueIds",
        "reason"
      ],
      "title": "SourceCorrespondence",
      "type": "object"
    },
    "SourceEditScope": {
      "additionalProperties": false,
      "properties": {
        "affects": {
          "const": "all_executions",
          "title": "Affects",
          "type": "string"
        },
        "includesUnobservedExecutions": {
          "const": true,
          "title": "Includesunobservedexecutions",
          "type": "boolean"
        },
        "kind": {
          "const": "source_expression",
          "title": "Kind",
          "type": "string"
        },
        "observedInvocations": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Observedinvocations",
          "type": "integer"
        },
        "observedModuleInstances": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Observedmoduleinstances",
          "type": "integer"
        },
        "observedOperations": {
          "maximum": 9007199254740991,
          "minimum": 1,
          "title": "Observedoperations",
          "type": "integer"
        }
      },
      "required": [
        "kind",
        "affects",
        "observedOperations",
        "observedInvocations",
        "observedModuleInstances",
        "includesUnobservedExecutions"
      ],
      "title": "SourceEditScope",
      "type": "object"
    },
    "SourceFloatLiteral": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "const": "float",
          "title": "Kind",
          "type": "string"
        },
        "value": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "const": "-0",
              "type": "string"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "kind",
        "value"
      ],
      "title": "SourceFloatLiteral",
      "type": "object"
    },
    "SourceSnapshot": {
      "additionalProperties": false,
      "properties": {
        "className": {
          "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}(?![\\s\\S])",
          "title": "Classname",
          "type": "string"
        },
        "configDigest": {
          "anyOf": [
            {
              "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Configdigest"
        },
        "dependencyScope": {
          "enum": [
            "complete",
            "partial"
          ],
          "title": "Dependencyscope",
          "type": "string"
        },
        "environmentDigest": {
          "pattern": "^[0-9a-f]{64}(?![\\s\\S])",
          "title": "Environmentdigest",
          "type": "string"
        },
        "files": {
          "items": {
            "$ref": "#/definitions/FileIdentity"
          },
          "maxItems": 100000,
          "minItems": 1,
          "title": "Files",
          "type": "array"
        },
        "generation": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Generation",
          "type": "integer"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "modelFile": {
          "maxLength": 1024,
          "minLength": 1,
          "pattern": "^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^\\\\:\\x00-\\x1f]+(?![\\s\\S])",
          "title": "Modelfile",
          "type": "string"
        },
        "modelId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Modelid",
          "type": "string"
        },
        "sourceMode": {
          "enum": [
            "saved",
            "buffers"
          ],
          "title": "Sourcemode",
          "type": "string"
        },
        "version": {
          "const": 1,
          "title": "Version",
          "type": "integer"
        },
        "workspaceId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Workspaceid",
          "type": "string"
        },
        "workspaceSessionId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Workspacesessionid",
          "type": "string"
        }
      },
      "required": [
        "version",
        "id",
        "workspaceSessionId",
        "workspaceId",
        "modelId",
        "modelFile",
        "className",
        "generation",
        "files",
        "dependencyScope",
        "configDigest",
        "environmentDigest",
        "sourceMode"
      ],
      "title": "SourceSnapshot",
      "type": "object"
    },
    "SourceValueChoice": {
      "additionalProperties": false,
      "properties": {
        "binding": {
          "$ref": "#/definitions/SourceBinding"
        },
        "name": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Name",
          "type": "string"
        },
        "value": {
          "$ref": "#/definitions/ValueRef"
        }
      },
      "required": [
        "name",
        "value",
        "binding"
      ],
      "title": "SourceValueChoice",
      "type": "object"
    },
    "StateRef": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "enum": [
            "parameter",
            "buffer"
          ],
          "title": "Kind",
          "type": "string"
        },
        "name": {
          "maxLength": 4096,
          "title": "Name",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "name"
      ],
      "title": "StateRef",
      "type": "object"
    },
    "SymbolicArgument": {
      "additionalProperties": false,
      "properties": {
        "expression": {
          "maxLength": 4096,
          "title": "Expression",
          "type": "string"
        },
        "kind": {
          "const": "symbolic",
          "title": "Kind",
          "type": "string"
        },
        "type": {
          "enum": [
            "int",
            "float",
            "bool"
          ],
          "title": "Type",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "type",
        "expression"
      ],
      "title": "SymbolicArgument",
      "type": "object"
    },
    "TextLiteral": {
      "additionalProperties": false,
      "properties": {
        "kind": {
          "enum": [
            "str",
            "dtype",
            "device",
            "layout",
            "memory_format"
          ],
          "title": "Kind",
          "type": "string"
        },
        "value": {
          "maxLength": 4096,
          "title": "Value",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "value"
      ],
      "title": "TextLiteral",
      "type": "object"
    },
    "Value": {
      "additionalProperties": false,
      "properties": {
        "aliasOf": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Aliasof"
        },
        "device": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Device",
          "type": "string"
        },
        "dtype": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Dtype",
          "type": "string"
        },
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "logicalBytes": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Logicalbytes"
        },
        "logicalVersion": {
          "maximum": 9007199254740991,
          "minimum": 0,
          "title": "Logicalversion",
          "type": "integer"
        },
        "objectId": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Objectid",
          "type": "string"
        },
        "origin": {
          "enum": [
            "input",
            "state",
            "operation",
            "external"
          ],
          "title": "Origin",
          "type": "string"
        },
        "previousVersion": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Previousversion"
        },
        "producer": {
          "anyOf": [
            {
              "$ref": "#/definitions/OutputRef"
            },
            {
              "type": "null"
            }
          ]
        },
        "requiresGrad": {
          "title": "Requiresgrad",
          "type": "boolean"
        },
        "shape": {
          "anyOf": [
            {
              "items": {
                "maximum": 9007199254740991,
                "minimum": 0,
                "type": "integer"
              },
              "maxItems": 32,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "title": "Shape"
        },
        "state": {
          "items": {
            "$ref": "#/definitions/StateRef"
          },
          "maxItems": 4096,
          "title": "State",
          "type": "array"
        },
        "storageId": {
          "anyOf": [
            {
              "maxLength": 128,
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "title": "Storageid"
        },
        "storageOffset": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "title": "Storageoffset"
        },
        "stride": {
          "anyOf": [
            {
              "items": {
                "maximum": 9007199254740991,
                "minimum": -9007199254740991,
                "type": "integer"
              },
              "maxItems": 32,
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "title": "Stride"
        },
        "version": {
          "anyOf": [
            {
              "maximum": 9007199254740991,
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "title": "Version"
        }
      },
      "required": [
        "id",
        "objectId",
        "storageId",
        "version",
        "shape",
        "dtype",
        "device",
        "requiresGrad",
        "aliasOf",
        "previousVersion",
        "logicalVersion",
        "stride",
        "storageOffset",
        "producer",
        "origin",
        "state"
      ],
      "title": "Value",
      "type": "object"
    },
    "ValueRef": {
      "additionalProperties": false,
      "properties": {
        "id": {
          "maxLength": 128,
          "minLength": 1,
          "title": "Id",
          "type": "string"
        },
        "kind": {
          "const": "value",
          "title": "Kind",
          "type": "string"
        }
      },
      "required": [
        "kind",
        "id"
      ],
      "title": "ValueRef",
      "type": "object"
    }
  },
  "properties": {
    "activationEditCapability": {
      "$ref": "#/definitions/ActivationEditCapability"
    },
    "candidateExecution": {
      "$ref": "#/definitions/CandidateExecution"
    },
    "candidatePostconditions": {
      "$ref": "#/definitions/CandidatePostconditions"
    },
    "candidateRequest": {
      "$ref": "#/definitions/CandidateRequest"
    },
    "capture": {
      "$ref": "#/definitions/CaptureSnapshot"
    },
    "captureReconciliation": {
      "$ref": "#/definitions/CaptureReconciliation"
    },
    "captureReconciliationRequest": {
      "$ref": "#/definitions/CaptureReconciliationRequest"
    },
    "captureReference": {
      "$ref": "#/definitions/CaptureReference"
    },
    "configurationLiteralCapability": {
      "$ref": "#/definitions/ConfigurationLiteralCapability"
    },
    "constructorLiteralCapability": {
      "$ref": "#/definitions/ConstructorLiteralCapability"
    },
    "correspondence": {
      "$ref": "#/definitions/SourceCorrespondence"
    },
    "deleteUnaryCapability": {
      "$ref": "#/definitions/DeleteUnaryCapability"
    },
    "deleteUnaryEdit": {
      "$ref": "#/definitions/DeleteUnaryEdit"
    },
    "displayPort": {
      "$ref": "#/definitions/DisplayPort"
    },
    "insertUnaryCapability": {
      "$ref": "#/definitions/InsertUnaryCapability"
    },
    "insertUnaryEdit": {
      "$ref": "#/definitions/InsertUnaryEdit"
    },
    "literalEditCapability": {
      "$ref": "#/definitions/LiteralEditCapability"
    },
    "plan": {
      "$ref": "#/definitions/ResolvedExecutionPlan"
    },
    "portableCapture": {
      "$ref": "#/definitions/PortableCapture"
    },
    "preparation": {
      "$ref": "#/definitions/CapturePreparation"
    },
    "projection": {
      "$ref": "#/definitions/CaptureProjection"
    },
    "reconnectArgumentCapability": {
      "$ref": "#/definitions/ReconnectArgumentCapability"
    },
    "reconnectArgumentEdit": {
      "$ref": "#/definitions/ReconnectArgumentEdit"
    },
    "replaceActivationEdit": {
      "$ref": "#/definitions/ReplaceActivationEdit"
    },
    "setConfigurationLiteralEdit": {
      "$ref": "#/definitions/SetConfigurationLiteralEdit"
    },
    "setConstructorLiteralEdit": {
      "$ref": "#/definitions/SetConstructorLiteralEdit"
    },
    "setLiteralEdit": {
      "$ref": "#/definitions/SetLiteralEdit"
    },
    "state": {
      "$ref": "#/definitions/CaptureState"
    }
  },
  "required": [
    "capture",
    "portableCapture",
    "state",
    "plan",
    "candidateExecution",
    "captureReconciliationRequest",
    "captureReconciliation",
    "candidateRequest",
    "candidatePostconditions",
    "preparation",
    "correspondence",
    "projection",
    "captureReference",
    "displayPort",
    "literalEditCapability",
    "setLiteralEdit",
    "reconnectArgumentCapability",
    "reconnectArgumentEdit",
    "activationEditCapability",
    "replaceActivationEdit",
    "insertUnaryCapability",
    "insertUnaryEdit",
    "deleteUnaryCapability",
    "deleteUnaryEdit",
    "constructorLiteralCapability",
    "setConstructorLiteralEdit",
    "configurationLiteralCapability",
    "setConfigurationLiteralEdit"
  ],
  "title": "SemanticContracts",
  "type": "object"
};
export default schema;
