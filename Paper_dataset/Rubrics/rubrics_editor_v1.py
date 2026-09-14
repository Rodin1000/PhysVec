from flask import Flask, render_template, request, jsonify, session
from pathlib import Path
import json, os, uuid
from collections import OrderedDict

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.urandom(24)  # Needed for session management

BASE_DIR = Path(__file__).parent.resolve()
RESERVED_FOLDERS = {"templates", "static", "__pycache__"}
DEFAULT_FILENAME = "rubrics.json"


def sanitize_folder(folder: str) -> str:
    """Limit folder input to prevent escaping BASE_DIR."""
    folder = (folder or "").strip().replace("\\", "/")
    if not folder:
        return ""
    candidate = Path(folder)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    # Disallow reserved folders anywhere in the path to keep templates/static clean
    if any(part in RESERVED_FOLDERS for part in candidate.parts):
        return ""
    return candidate.as_posix()


def set_active_folder(folder: str):
    session['active_folder'] = sanitize_folder(folder)


def get_active_folder() -> str:
    return sanitize_folder(session.get('active_folder', ''))


# Use a session to track the active file for each user
def get_active_file():
    """Gets the active data file path from the session, defaulting to 'rubrics.json'."""
    folder = get_active_folder()
    filename = Path(session.get('active_file', DEFAULT_FILENAME)).name or DEFAULT_FILENAME
    if folder:
        return BASE_DIR / folder / filename
    return BASE_DIR / filename

def set_active_file(filename: str, folder=None):
    """Sets the active data file path in the session."""
    safe_name = Path(filename).name or DEFAULT_FILENAME
    session['active_file'] = safe_name
    if folder is not None:
        set_active_folder(folder)

def load_data():
    data_file = get_active_file()
    if not data_file.exists():
        # If the session's active file doesn't exist, it might be a new one.
        # Create it with a default structure.
        # This also handles the very first run where 'rubrics.json' might not exist.
        initial_data = {
            "source": "",
            "guide_caption": "",
            "task": "",
            "weight": 1,
            "children": []
        }
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text(json.dumps(initial_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return initial_data

    data = json.loads(data_file.read_text(encoding="utf-8"))
    
    # --- Migration and Validation Logic ---
    # This part handles old data formats and ensures the structure is correct.
    migrated = False
    if isinstance(data, list):
        # Old format was a list, migrate to a dict with metadata
        meta = {"source": "", "guide_caption": "", "task": ""}
        statements = data
        data = {
            "source": meta["source"],
            "guide_caption": meta["guide_caption"], 
            "task": meta["task"],
            "weight": 1,
            "children": statements
        }
        migrated = True
    
    if not all(k in data for k in ("source", "guide_caption", "task", "weight", "children")):
        # Ensure all required top-level keys exist
        data = {
            "source": data.get("source", ""),
            "guide_caption": data.get("guide_caption", ""),
            "task": data.get("task", ""),
            "weight": data.get("weight", 1),
            "children": data.get("children", [])
        }
        migrated = True

    if migrated:
        save_data(data) # Save the corrected structure

    return data

def save_data(data: dict):
    data_file = get_active_file()
    data_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_file.with_suffix(data_file.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, data_file)

@app.route("/")
def index():
    # Clear session on new page load to start with the default 'rubrics.json'
    session.pop('active_file', None)
    session.pop('active_folder', None)
    return render_template("index.html")


@app.route("/api/files", methods=["GET"])
def api_files():
    """List all .json files under the specified Rubrics subfolder."""
    folder = sanitize_folder(request.args.get("folder", ""))
    target_dir = BASE_DIR / folder if folder else BASE_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify([])
    files = sorted(p.name for p in target_dir.glob("*.json") if p.is_file())
    return jsonify(files)


@app.route("/api/folders", methods=["GET"])
def api_folders():
    """Return available subfolders under Rubrics (excluding templates/static)."""
    folders = [{"label": "Rubrics (root)", "value": ""}]
    for path in sorted(BASE_DIR.rglob("*")):
        if not path.is_dir():
            continue
        rel = path.relative_to(BASE_DIR)
        if not rel.parts:
            continue
        if any(part in RESERVED_FOLDERS for part in rel.parts):
            continue
        folders.append({"label": rel.as_posix(), "value": rel.as_posix()})
    return jsonify(folders)


@app.route("/api/load_file", methods=["POST"]) 
def api_load_file():
    """Set the active file (in session) to the provided filename and return its content."""
    payload = request.get_json(force=True) or {}
    filename = (payload.get('filename') or '').strip()
    if not filename:
        return jsonify({"ok": False, "error": "filename is required"}), 400

    folder = sanitize_folder(payload.get('folder', ""))
    target_dir = BASE_DIR / folder if folder else BASE_DIR
    target = target_dir / filename
    if not target.exists() or not target.is_file():
        return jsonify({"ok": False, "error": "file not found"}), 404

    set_active_file(filename, folder=folder)
    data = load_data()
    return jsonify({
        "ok": True,
        "active_file": str(get_active_file()),
        "folder": get_active_folder(),
        "data": data
    })

# return the whole tree structure for jsonl preview
@app.route("/api/tree", methods=["GET"])
def api_tree():
    return jsonify(load_data())

# metadata endpoints: get and update the top-level metadata dict
@app.route("/api/meta", methods=["GET"]) 
def api_get_meta():
    data = load_data()
    return jsonify({
        "source": data.get("source", ""),
        "guide_caption": data.get("guide_caption", ""),
        "task": data.get("task", ""),
        "folder": get_active_folder()
    })

@app.route("/api/meta", methods=["POST"])
def api_update_meta():
    payload = request.get_json(force=True) or {}
    source = (payload.get("source") or "").strip()
    folder = sanitize_folder(payload.get("folder", get_active_folder()))
    set_active_folder(folder)
    
    # --- Generate filename from source and update session ---
    if source:
        parts = source.split(',', 1)
        paper_id = parts[0].strip()
        figure_table = ""
        if len(parts) > 1:
            figure_table = parts[1].strip().replace(' ', '_')
        
        if paper_id and figure_table:
            new_filename = f"{paper_id}_{figure_table}_rubrics.json"
            set_active_file(new_filename, folder=folder)

    # Load data (from old file if it's the first save, or new file if already set)
    data = load_data() 
    
    # Update data with new metadata
    data["source"] = source
    data["guide_caption"] = (payload.get("guide_caption") or "").strip()
    data["task"] = (payload.get("task") or "").strip()
    data["weight"] = 1 # Keep weight at 1

    # Save data (will save to the new file path set in the session)
    save_data(data)
    return jsonify({
        "ok": True,
        "active_file": session.get('active_file'),
        "folder": get_active_folder()
    })

# rubrics input area
@app.route("/api/statements", methods=["GET"])
def api_statements():
    data = load_data()
    return jsonify(data.get("children", []))

# add a statement or issue
@app.route("/api/statements", methods=["POST"])
def api_add_statement():
    payload = request.get_json(force=True) or {}
    item_type = payload.get("type", "statement")  # "statement" or "issue"
    content = (payload.get("content") or "").strip()
    weight = float(payload.get("weight") or 0)
    parent_id = payload.get("parent_id")

    if not content:
        return jsonify({"ok": False, "error": "content is required"}), 400

    if item_type == "issue":
        new_item = OrderedDict([
            ("issue", content),
            ("weight", weight),
            ("children", OrderedDict([
                ("judge", "TBD"),
                ("support", "TBD")
            ]))
        ])
    else:  # statement
        new_item = OrderedDict([
            ("id", str(uuid.uuid4())),
            ("statement", content),
            ("weight", weight),
            ("children", [])
        ])

    data = load_data()
    children = data.get("children", [])

    if parent_id:
        # 添加为子项，递归查找parent
        def add_child_to_parent(items_list, parent_id, child):
            for it in items_list:
                it_id = it.get("id") or it.get("issue")
                if it_id == parent_id:
                    if isinstance(it.get("children"), list):  # only statements have children lists
                        it["children"].append(child)
                        return True
                    return False
                # recurse into children if they are a list
                if isinstance(it.get("children"), list) and add_child_to_parent(it["children"], parent_id, child):
                    return True
            return False

        if not add_child_to_parent(children, parent_id, new_item):
            return jsonify({"ok": False, "error": "parent not found"}), 404
    else:
        # 添加为顶级项
        children.append(new_item)

    # update children in data
    data["children"] = children
    save_data(data)
    return jsonify({"ok": True})

# delete a statement or issue
@app.route("/api/statements/<item_id>", methods=["DELETE"])
def api_delete_statement(item_id):
    data = load_data()
    children = data.get("children", [])

    def remove_item(items_list, target_id):
        for i, it in enumerate(items_list):
            current_id = it.get("id") or it.get("issue")
            if current_id == target_id:
                items_list.pop(i)
                return True
            if isinstance(it.get("children"), list) and remove_item(it["children"], target_id):
                return True
        return False

    if not remove_item(children, item_id):
        return jsonify({"ok": False, "error": "not found"}), 404

    data["children"] = children
    save_data(data)
    return jsonify({"ok": True})


@app.route('/api/statements/<item_id>', methods=['PUT'])
def api_update_statement(item_id):
    """Update the content and/or weight of a statement or issue identified by item_id."""
    payload = request.get_json(force=True) or {}
    new_content = payload.get('content')
    new_weight = payload.get('weight')

    data = load_data()
    children = data.get('children', [])

    updated = None

    def update_in_list(items):
        nonlocal updated
        for it in items:
            current_id = it.get('id') or it.get('issue')
            if current_id == item_id:
                # only allow updating the textual content and weight
                if 'statement' in it and new_content is not None:
                    it['statement'] = new_content
                if 'issue' in it and new_content is not None:
                    it['issue'] = new_content
                if new_weight is not None:
                    try:
                        it['weight'] = float(new_weight)
                    except Exception:
                        pass
                updated = it
                return True
            # recurse into children lists (only statements have list children)
            if isinstance(it.get('children'), list) and update_in_list(it['children']):
                return True
        return False

    if not update_in_list(children):
        return jsonify({"ok": False, "error": "not found"}), 404

    data['children'] = children
    save_data(data)
    return jsonify({"ok": True, "updated": updated})

if __name__ == "__main__":
    # 直接 python app.py 即可启动。生产环境请用 gunicorn 等。
    app.run(debug=True)