"""Wire checks for the host-owned opening workflow; profile semantics stay in Python."""
import json
import math
import re


def text(value, limit):
    return isinstance(value, str) and 0 < len(value) <= limit


def nat(value):
    return type(value) is int and value >= 0


def bounded(value, limit=131072):
    count = 0
    def visit(item, depth=0):
        nonlocal count
        count += 1
        if count > 16384 or depth > 16:
            return False
        if item is None or isinstance(item, (str, bool)):
            return True
        if type(item) in (int, float):
            return math.isfinite(item)
        if isinstance(item, list):
            return all(visit(child, depth + 1) for child in item)
        return isinstance(item, dict) and all(isinstance(key, str) and len(key) <= 1024 and visit(child, depth + 1) for key, child in item.items())
    return visit(value) and len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()) <= limit


def fields(items):
    return items is None or isinstance(items, list) and len(items) <= 64 and all(
        isinstance(item, dict) and text(item.get("name"), 128) and type(item.get("required")) is bool
        and item.get("kind") in ("positional", "keyword", "keyword_only", "varargs", "kwargs")
        and ("annotation" not in item or isinstance(item["annotation"], str))
        and ("defaultText" not in item or isinstance(item["defaultText"], str))
        and ("source" not in item or isinstance(item["source"], dict) and text(item["source"].get("file"), 4096)
             and nat(item["source"].get("line")) and item["source"]["line"] > 0) for item in items)


def discovery(value):
    if not isinstance(value, dict) or not bounded(value, 262144):
        return False
    def items(key, check):
        entries = value.get(key)
        return isinstance(entries, list) and len(entries) <= 64 and all(isinstance(item, dict) and check(item) for item in entries)
    source = lambda item: text(item.get("file"), 4096)
    digest = value.get("sourceDigest")
    return (text(value.get("model"), 512) and isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        and type(value.get("truncated")) is bool and "init" in value and fields(value["init"]) and "inputs" in value and fields(value["inputs"])
        and items("examples", lambda item: source(item) and nat(item.get("line")) and isinstance(item.get("init"), dict) and ("inputs" not in item or isinstance(item["inputs"], dict)))
        and items("factories", lambda item: source(item) and text(item.get("callable"), 128) and isinstance(item.get("fields"), list) and fields(item["fields"]))
        and items("configs", lambda item: source(item) and text(item.get("cls"), 128) and "fields" in item and fields(item["fields"])
                  and "parameter" in item and (item["parameter"] is None or text(item["parameter"], 128)) and item.get("construction") in ("call", "attributes"))
        and items("parents", lambda item: source(item) and text(item.get("cls"), 128) and text(item.get("attribute"), 128))
        and items("constraints", lambda item: source(item) and text(item.get("input"), 128) and type(item.get("axis")) is int
                  and ("size" not in item or nat(item["size"])) and ("initParameter" not in item or text(item["initParameter"], 128)))
        and ("resolution" not in value or resolution(value["resolution"])))


def evidence(value):
    return isinstance(value, list) and len(value) <= 64 and all(isinstance(item, dict)
        and text(item.get("field"), 256) and item.get("kind") in ("repository", "source", "example")
        and text(item.get("detail"), 512) and ("file" not in item or text(item["file"], 4096))
        and ("line" not in item or nat(item["line"])) for item in value)


def resolution(value):
    return (isinstance(value, dict) and value.get("status") in ("complete", "partial")
        and draft(value.get("draft")) and evidence(value.get("evidence"))
        and isinstance(value.get("unresolved"), list) and len(value["unresolved"]) <= 64
        and all(text(item, 1024) for item in value["unresolved"])
        and (value["status"] != "complete" or not value["unresolved"]))


def draft(value):
    if (not isinstance(value, dict) or not bounded(value) or not isinstance(value.get("profile"), dict)
            or set(value) - {"profile", "profileName", "configObject", "configDefaults"}
            or "profileName" in value and not text(value["profileName"], 64)):
        return False
    if 'configDefaults' in value and (value['configDefaults'] is not True
            or 'configObject' in value or 'factory' in value['profile']):
        return False
    if "configObject" not in value:
        return True
    config = value["configObject"]
    return (isinstance(config, dict) and set(config) == {"index", "parameter", "values"}
            and nat(config["index"]) and text(config["parameter"], 128) and isinstance(config["values"], dict))


def environment_name(value):
    return bool(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', value)
                and not re.fullmatch(r'con|prn|aux|nul|com[1-9]|lpt[1-9]', value, re.IGNORECASE))


def action(value):
    if not isinstance(value, dict):
        return False
    if value.get("kind") == "execution":
        return (text(value.get("reviewId"), 128) and execution_device(value.get("device"))
                and set(value) == {"kind", "reviewId", "action", "device"}
                and value.get("action") in ("run", "cancel", "refresh", "select"))
    if value.get("kind") == "edit_execution":
        field = value.get("field")
        return (set(value) == {"kind", "reviewId", "field"} and text(value["reviewId"], 128)
                and isinstance(field, list) and 0 < len(field) <= 12 and field[0] in ("inputs", "mode", "grad_enabled", "device")
                and (field[0] == "inputs" or len(field) == 1)
                and all(text(part, 128) or nat(part) and part < 64 for part in field))
    if value.get("kind") == "submit":
        return set(value) == {"kind", "draft"} and draft(value["draft"])
    if value.get("kind") == "reveal_setup_source":
        return set(value) == {"kind", "file", "line"} and text(value["file"], 4096) and nat(value["line"]) and value["line"] > 0
    if value.get("kind") in ("factory_template", "agent_help"):
        return set(value) <= {"kind", "draft"} and ("draft" not in value or draft(value["draft"]))
    if value.get("kind") == "parent":
        return set(value) == {"kind", "index"} and nat(value["index"]) and value["index"] < 64
    if value.get("kind") == "select_environment":
        return set(value) == {"kind", "candidateId"} and text(value["candidateId"], 64)
    if value.get("kind") == "create_environment":
        return set(value) <= {"kind", "name"} and ("name" not in value or environment_name(value["name"]))
    return set(value) == {"kind"} and value.get("kind") in ("retry", "change_inputs", "cancel", "source_only", "save_sources", "environment", "edit_config", "agent_help", "factory_template", "install_dependencies", "refresh_environments", "use_installed_versions", "replace_versions", "create_environment")


def environment_candidate(value):
    return (isinstance(value, dict) and text(value.get("id"), 64)
        and text(value.get("interpreter"), 4096) and text(value.get("source"), 256)
        and value.get("status") in ("libraries_present", "missing_libraries", "incompatible", "unavailable")
        and all(key not in value or value[key] is True for key in ("selected", "pythonSelected", "managed"))
        and all(key not in value or text(value[key], 256) for key in ("python", "implementation", "architecture"))
        and ("prefix" not in value or text(value["prefix"], 4096))
        and all(key not in value or value[key] is None or text(value[key], 256) for key in ("torch", "abi"))
        and ("fingerprint" not in value or isinstance(value["fingerprint"], str) and re.fullmatch(r"[a-f0-9]{64}", value["fingerprint"]))
        and all(key not in value or isinstance(value[key], list) and len(value[key]) <= 32
                and all(text(item, 1024) for item in value[key]) for key in ("missing", "incompatible", "modelMissing", "modelIncompatible"))
        and ("detail" not in value or text(value["detail"], 1024))
        and ("modelChecked" not in value or value["modelChecked"] is True)
        and (value.get("modelChecked") is True or ("modelMissing" not in value and "modelIncompatible" not in value))
        and not set(value) - {"id", "interpreter", "source", "status", "selected", "pythonSelected", "managed", "python", "torch",
                             "prefix", "implementation", "abi", "architecture", "fingerprint", "missing", "incompatible", "detail",
                             "modelChecked", "modelMissing", "modelIncompatible"})


def environment_recovery(value):
    if (not isinstance(value, dict) or value.get("reason") not in ("ready", "selection_required", "unavailable", "missing_libraries", "incompatible", "worker_failed")
            or not text(value.get("detail"), 1024) or not isinstance(value.get("candidates"), list)
            or len(value["candidates"]) > 16 or not all(environment_candidate(candidate) for candidate in value["candidates"])):
        return False
    ids = [candidate["id"] for candidate in value["candidates"]]
    return (len(set(ids)) == len(ids) and ("selectedId" not in value or value["selectedId"] in ids)
        and (value["reason"] != "ready" or "selectedId" in value)
        and all(key not in value or isinstance(value[key], list) and len(value[key]) <= 32
                and all(text(item, 1024) for item in value[key]) for key in ("requirements", "compatibilityPackages", "missingModules", "declaredPins"))
        and ("replaceable" not in value or value["replaceable"] is True and bool(value.get("declaredPins")))
        and ("canCreate" not in value or value["canCreate"] is True)
        and ("recommendedId" not in value or value["recommendedId"] in ids)
        and not set(value) - {"reason", "detail", "candidates", "selectedId", "requirements", "compatibilityPackages", "missingModules", "declaredPins", "replaceable", "canCreate", "recommendedId"})


def failure(value):
    return (isinstance(value, dict) and value.get("kind") in ("setup", "source", "environment", "execution", "capture")
        and ("code" not in value or value["kind"] == "environment" and isinstance(value["code"], str)
             and bool(re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", value["code"])))
        and ("fields" not in value or isinstance(value["fields"], list) and len(value["fields"]) <= 64
             and all(text(field, 256) for field in value["fields"]))
        and ("missingModules" not in value or value["kind"] == "environment"
             and isinstance(value["missingModules"], list) and len(value["missingModules"]) <= 32
             and all(isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) for name in value["missingModules"]))
        and ("importFailure" not in value or value["kind"] == "environment" and value["importFailure"] is True)
        and not set(value) - {"kind", "code", "fields", "missingModules", "importFailure"})


def execution_device(value):
    return isinstance(value, str) and re.fullmatch(r"(cpu|mps|cuda(?::[0-9]{1,3})?)", value) is not None


def execution_reuse(value):
    """An identical earlier result shown instead of running (execution cache)."""
    return (isinstance(value, dict) and set(value) == {"capturedAt"} and type(value["capturedAt"]) is int
            and 0 <= value["capturedAt"] <= 2**53 - 1)


def execution_review(value):
    def number(n):
        return type(n) in (int, float) and math.isfinite(n) and n >= 0
    def metric(n):
        return n is None or number(n)
    required = {"id", "phase", "kind", "device", "configuredDevice", "detail", "stages", "inputs", "mode", "gradients", "interpreter"}
    if (not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"resources", "profileName", "reused", "fresh"}
            or "reused" in value and (not execution_reuse(value["reused"]) or value.get("phase") != "succeeded")
            or "fresh" in value and (value["fresh"] is not True or "reused" in value)
            or "profileName" in value and not text(value["profileName"], 64)
            or not text(value["id"], 128) or not execution_device(value["device"]) or not execution_device(value["configuredDevice"])
            or value["phase"] not in ("review", "running", "cancelling", "succeeded", "failed", "cancelled")
            or value["kind"] not in ("prepare", "capture", "trace") or not text(value["detail"], 1024)
            or not text(value["inputs"], 4096) or not text(value["mode"], 128) or not text(value["interpreter"], 4096)
            or type(value["gradients"]) is not bool or not isinstance(value["stages"], list) or len(value["stages"]) > 8
            or not all(isinstance(stage, dict) and set(stage) == {"label", "calls"} and text(stage["label"], 128) and nat(stage["calls"]) for stage in value["stages"])):
        return False
    if "resources" not in value:
        return True
    r = value["resources"]
    return (isinstance(r, dict) and set(r) == {"sampledAt", "host", "remote", "cpu", "cores", "cpuPercent", "memoryFree", "memoryTotal", "devices", "detail"}
            and number(r["sampledAt"]) and text(r["host"], 512) and type(r["remote"]) is bool and text(r["cpu"], 512)
            and number(r["cores"]) and metric(r["cpuPercent"]) and number(r["memoryFree"]) and number(r["memoryTotal"])
            and text(r["detail"], 1024) and isinstance(r["devices"], list) and len(r["devices"]) <= 128
            and all(isinstance(d, dict) and set(d) == {"id", "name", "utilization", "memoryFree", "memoryTotal"}
                    and execution_device(d["id"]) and text(d["name"], 512) and metric(d["utilization"])
                    and metric(d["memoryFree"]) and metric(d["memoryTotal"]) for d in r["devices"]))


def opening(value, target_check):
    return (isinstance(value, dict) and text(value.get("id"), 128) and target_check(value.get("target"))
        and value.get("phase") in ("awaiting_execution", "resolving", "needs_setup", "saving", "capturing", "ready", "failed", "cancelled", "source_only")
        and text(value.get("detail"), 1024) and type(value.get("hasCapture")) is bool
        and ("execution" not in value or execution_review(value["execution"]))
        and ("cancellationReason" not in value or value["phase"] == "cancelled" and value["cancellationReason"] == "source_changed")
        and ("configFileExists" not in value or type(value["configFileExists"]) is bool)
        and ("discovery" not in value or discovery(value["discovery"])) and ("draft" not in value or draft(value["draft"]))
        and ("requirements" not in value or isinstance(value["requirements"], list) and len(value["requirements"]) <= 64 and all(text(item, 1024) for item in value["requirements"]))
        and ("evidence" not in value or evidence(value["evidence"]))
        and ("failure" not in value or failure(value["failure"]))
        and ("environment" not in value or environment_recovery(value["environment"]))
        and ("environmentName" not in value or environment_name(value["environmentName"]))
        and not set(value) - {"id", "target", "phase", "detail", "hasCapture", "execution", "cancellationReason", "configFileExists", "discovery", "draft", "requirements", "evidence", "failure", "environment", "environmentName"})
