#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
CHUNK_SIZE = 1024 * 1024

def safe_json_load(path: Path) -> Tuple[Any, Dict[str, Any]]:
    meta={"file":str(path),"status":"ok","errors":[],"bytes":0}
    if not path.exists(): meta["status"]="error"; meta["errors"].append("file_missing"); return None, meta
    try:
        meta["bytes"]=path.stat().st_size
        txt=path.read_text(encoding="utf-8").strip()
        if not txt: meta["status"]="empty"; return None, meta
        return json.loads(txt), meta
    except json.JSONDecodeError as exc:
        meta["status"]="error"; meta["errors"].append(f"malformed_json:{exc.msg}@{exc.lineno}:{exc.colno}"); return None, meta
    except OSError as exc:
        meta["status"]="error"; meta["errors"].append(f"io_error:{exc}"); return None, meta

def walk_json_paths(obj: Any, max_depth: int = 6, prefix: str = "$") -> List[Dict[str, Any]]:
    out=[]
    def walk(v: Any, p: str, d: int) -> None:
        if d>max_depth: return
        if isinstance(v, dict):
            out.append({"path":p,"type":"dict","keys":list(v.keys())[:30]})
            for k,val in v.items(): walk(val, f"{p}.{k}", d+1)
        elif isinstance(v, list):
            out.append({"path":p,"type":"list","length":len(v)})
            for i,val in enumerate(v[:3]): walk(val, f"{p}[{i}]", d+1)
        else:
            out.append({"path":p,"type":type(v).__name__,"preview":extract_scalar_preview(v)})
    walk(obj,prefix,0)
    return out

def find_dicts_with_keys(obj: Any, required_keys: List[str], max_depth: int = 6, prefix: str = "$") -> List[Dict[str, Any]]:
    found=[]
    req=set(required_keys)
    def walk(v: Any, p: str, d: int):
        if d>max_depth: return
        if isinstance(v, dict):
            ks=set(v.keys())
            if req.issubset(ks): found.append({"path":p,"keys":list(v.keys())[:30],"preview":{k:extract_scalar_preview(v.get(k)) for k in required_keys}})
            for k,val in v.items(): walk(val,f"{p}.{k}",d+1)
        elif isinstance(v,list):
            for i,val in enumerate(v[:10]): walk(val,f"{p}[{i}]",d+1)
    walk(obj,prefix,0)
    return found

def find_lists_of_dicts(obj: Any, max_depth: int = 6, prefix: str = "$") -> List[Dict[str, Any]]:
    out=[]
    def walk(v: Any,p: str,d: int):
        if d>max_depth: return
        if isinstance(v,list):
            first=v[0] if v else None
            if isinstance(first,dict): out.append({"path":p,"length":len(v),"first_item_keys":list(first.keys())[:30]})
            for i,val in enumerate(v[:5]): walk(val,f"{p}[{i}]",d+1)
        elif isinstance(v,dict):
            for k,val in v.items(): walk(val,f"{p}.{k}",d+1)
    walk(obj,prefix,0)
    return out

def extract_scalar_preview(value: Any) -> Any:
    if isinstance(value,(str,int,float,bool)) or value is None: return value if not isinstance(value,str) else value[:120]
    if isinstance(value,list): return f"list(len={len(value)})"
    if isinstance(value,dict): return f"dict(keys={list(value.keys())[:5]})"
    return str(type(value).__name__)

def _iter_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload,dict):
        yield payload
        for v in payload.values():
            if isinstance(v,(dict,list)): yield from _iter_records(v)
    elif isinstance(payload,list):
        for i in payload:
            if isinstance(i,dict): yield i
            elif isinstance(i,list): yield from _iter_records(i)

def parse_sample_file(path: Path, max_records: int = 10000) -> Dict[str, Any]:
    payload, meta = safe_json_load(path)
    res={"file":str(path),"meta":meta,"records":[],"record_count":0,"truncated":False}
    if meta["status"]!="ok": return res
    for i,r in enumerate(_iter_records(payload)):
        if i>=max_records: res["truncated"]=True; break
        res["records"].append(r)
    res["record_count"]=len(res["records"])
    return res
