"""LaDeCo GT vs Prediction viewer.

Run:
    streamlit run app/viewer.py
"""
import json
import pickle
import re
from pathlib import Path

import streamlit as st
from PIL import Image

GT_DIR = Path("/home/cvlab18/media/data2/datasets/crello_images")
PRED_DIR = Path("/home/cvlab18/project/jaeho/elem2design/output/test")
ROLE_PKL = Path("/home/cvlab18/project/jaeho/elem2design/dataset/dataset/role/crello_role.pkl")
TEST_JSON = Path("/home/cvlab18/project/jaeho/elem2design/dataset/dataset/json/ours/test.json")

ROLES = {0: "Background", 1: "Underlay", 2: "Logo/Image", 3: "Text", 4: "Embellishment"}
ROLE_COLORS = {
    0: "#FFE4B5",
    1: "#FFB6C1",
    2: "#B0E0E6",
    3: "#98FB98",
    4: "#DDA0DD",
}


@st.cache_data
def load_pred_jsonl():
    items = []
    with open(PRED_DIR / "pred.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@st.cache_data
def load_roles():
    with open(ROLE_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_test_json():
    with open(TEST_JSON) as f:
        return {d["id"]: d for d in json.load(f)}


def split_predictions(pred_str):
    """Predictions look like '  ##### {...} $$$$$  ##### {...} $$$$$ ...'.
    Return list of 5 per-layer prediction strings."""
    chunks = [c.strip() for c in pred_str.split("$$$$$") if c.strip()]
    layers = []
    for c in chunks:
        m = re.match(r"^#####\s*(.*)$", c, re.DOTALL)
        layers.append(m.group(1).strip() if m else c)
    while len(layers) < 5:
        layers.append("")
    return layers[:5]


def gt_layers_from_test(test_item):
    if not test_item:
        return [""] * 5
    return [test_item["conversations"][k]["value"] for k in (1, 3, 5, 7, 9)]


def pretty_layer_json(s):
    """Layer output is `{json} {json} ...` (multiple element JSONs joined by space).
    Pretty-print as one JSON per line."""
    if not s or s == "{}":
        return s or ""
    out_lines = []
    depth, buf = 0, []
    for ch in s:
        buf.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj_str = "".join(buf).strip()
                buf = []
                try:
                    obj = json.loads(obj_str)
                    out_lines.append(json.dumps(obj))
                except Exception:
                    out_lines.append(obj_str)
    tail = "".join(buf).strip()
    if tail:
        out_lines.append(tail)
    return "\n".join(out_lines)


def role_badge(role_idx):
    name = ROLES.get(role_idx, "?")
    color = ROLE_COLORS.get(role_idx, "#DDDDDD")
    return f'<span style="background:{color};padding:2px 8px;border-radius:6px;font-size:12px;color:#222;">{name}</span>'


def main():
    st.set_page_config(page_title="LaDeCo Viewer", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            background-image: linear-gradient(45deg, #606060 25%, transparent 25%),
                              linear-gradient(-45deg, #606060 25%, transparent 25%),
                              linear-gradient(45deg, transparent 75%, #606060 75%),
                              linear-gradient(-45deg, transparent 75%, #606060 75%);
            background-size: 16px 16px;
            background-position: 0 0, 0 8px, 8px -8px, -8px 0;
            background-color: #808080;
            border: 1px solid #303030;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    items = load_pred_jsonl()
    roles_dict = load_roles()
    test_by_id = load_test_json()

    by_num = {it["num"]: it for it in items}
    ids_in_order = [it["id"] for it in sorted(items, key=lambda x: x["num"])]
    id_to_num = {it["id"]: it["num"] for it in items}

    # ---- Sidebar ----
    st.sidebar.title("LaDeCo Viewer")
    st.sidebar.caption(f"{len(items)} designs · GT={GT_DIR.name} · Pred={PRED_DIR.name}")

    mode = st.sidebar.radio("Pick design by", ["Index", "Design ID"], horizontal=True)
    if mode == "Index":
        num = st.sidebar.number_input(
            "num (0-based)", min_value=0, max_value=len(items) - 1, value=0, step=1
        )
        design_id = by_num[num]["id"]
    else:
        design_id = st.sidebar.selectbox("design id", options=ids_in_order, index=0)
        num = id_to_num[design_id]

    item = by_num[num]
    test_item = test_by_id.get(design_id)
    canvas_w, canvas_h = item["canvas_width"], item["canvas_height"]

    cprev, cnext = st.sidebar.columns(2)
    if cprev.button("◀ prev", use_container_width=True) and num > 0:
        st.query_params.update(num=str(num - 1))
        st.rerun()
    if cnext.button("next ▶", use_container_width=True) and num < len(items) - 1:
        st.query_params.update(num=str(num + 1))
        st.rerun()

    show_ele = st.sidebar.checkbox("show element thumbnails", value=True)
    show_raw = st.sidebar.checkbox("show raw layer JSON", value=True)
    show_diff = st.sidebar.checkbox("side-by-side per-layer", value=False)

    # ---- Header ----
    st.title(f"Design `{design_id}`")
    st.caption(f"num={num} · canvas {canvas_w}×{canvas_h}")

    # ---- Elements row ----
    if show_ele:
        st.subheader("Elements")
        elem_roles = roles_dict.get(design_id, [])
        elem_texts = item.get("render_text", [])
        n_elem = len(elem_roles)
        if n_elem == 0:
            st.info("no element roles found for this id")
        else:
            per_row = 8
            for row_start in range(0, n_elem, per_row):
                cols = st.columns(per_row)
                for k in range(per_row):
                    i = row_start + k
                    if i >= n_elem:
                        break
                    with cols[k]:
                        st.markdown(
                            f"**ele_{i}** {role_badge(elem_roles[i])}",
                            unsafe_allow_html=True,
                        )
                        ele_path = GT_DIR / design_id / f"ele_{i}.png"
                        if ele_path.exists():
                            st.image(str(ele_path), use_column_width=True)
                        else:
                            st.warning("missing png")
                        if i < len(elem_texts) and elem_texts[i]:
                            txt = elem_texts[i]
                            if len(txt) > 80:
                                txt = txt[:80] + "…"
                            st.caption(f'"{txt}"')

    # ---- Cumulative renders ----
    st.subheader("Cumulative renders — Ground truth vs Prediction")
    gt_cols = st.columns(5)
    pred_cols = st.columns(5)

    for i in range(5):
        # GT layer
        with gt_cols[i]:
            st.caption(f"**GT** layer_{i} ({ROLES[i]})")
            p = GT_DIR / design_id / f"layer_{i}.png"
            if p.exists():
                st.image(str(p), use_column_width=True)
            else:
                st.warning("missing")

        # Pred turn
        with pred_cols[i]:
            st.caption(f"**Pred** after turn {i} ({ROLES[i]})")
            p = PRED_DIR / "render" / f"{num}_{design_id}_{i}.png"
            if p.exists():
                st.image(str(p), use_column_width=True)
            else:
                st.warning("missing")

    # ---- Raw layer JSON ----
    if show_raw:
        st.subheader("Per-layer outputs (raw)")
        pred_layers = split_predictions(item["predictions"][0])
        gt_layers = gt_layers_from_test(test_item)

        if show_diff:
            for i in range(5):
                st.markdown(f"#### Layer {i} — {ROLES[i]}")
                c1, c2 = st.columns(2)
                c1.markdown("**Prediction**")
                c1.code(pretty_layer_json(pred_layers[i]) or "—", language="json")
                c2.markdown("**Ground truth**")
                c2.code(pretty_layer_json(gt_layers[i]) or "—", language="json")
        else:
            tabs = st.tabs([f"Layer {i} ({ROLES[i]})" for i in range(5)])
            for i, tab in enumerate(tabs):
                with tab:
                    c1, c2 = st.columns(2)
                    c1.markdown("**Prediction**")
                    c1.code(pretty_layer_json(pred_layers[i]) or "—", language="json")
                    c2.markdown("**Ground truth**")
                    c2.code(pretty_layer_json(gt_layers[i]) or "—", language="json")

    # ---- Footer: prompt context ----
    with st.expander("Human prompt (first turn)"):
        if test_item:
            st.code(test_item["conversations"][0]["value"])
        else:
            st.warning("test.json entry not found")


if __name__ == "__main__":
    main()
