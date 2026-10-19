import html

import streamlit as st

from dynamor.services.recommender import group_phases, load_context, suggest, validate_draft
from dynamor.settings import settings

PHASES = (
    ("early", "Early"),
    ("mid", "Mid"),
    ("late", "Late"),
)
BORDERS = {
    "weapon": "#E0863A",
    "vitality": "#3DDC84",
    "spirit": "#B46CFF",
    "skill": "#E6E6E6",
    "other": "#8A8A8A",
}


def _phase_html(items, *, show_time: bool) -> str:
    cards = []
    for item in items:
        border = BORDERS.get(item.category, BORDERS["other"])
        picture = ""
        if item.image:
            picture = (
                f'<img src="{html.escape(item.image)}" alt="" width="64" height="64" '
                'style="object-fit:contain">'
            )
        clock = ""
        if show_time and item.purchase_s is not None:
            clock = f"<div>{item.purchase_s // 60}:{item.purchase_s % 60:02d}</div>"
        meta = [str(item.cost)] if item.cost is not None else []
        meta.append(f"{item.confidence_pct}%")
        cards.append(
            f'<div style="border:2px solid {border};border-radius:8px;padding:8px;'
            f'width:120px;text-align:center">'
            f"{picture}"
            f'<div style="font-weight:600">{html.escape(item.name)}</div>'
            f"{clock}"
            f'<div style="opacity:0.75;font-size:0.85em">{html.escape(" · ".join(meta))}</div>'
            f"</div>"
        )
    return '<div style="display:flex;gap:8px;flex-wrap:wrap">' + "".join(cards) + "</div>"


st.set_page_config(page_title="DynamoR", layout="wide")
st.title("DynamoR")


@st.cache_resource
def cached_context():
    return load_context(settings)


try:
    context = cached_context()
except FileNotFoundError:
    st.error("Run python scripts/build_index.py first")
    st.stop()

heroes = sorted(context.heroes, key=lambda hero: (hero["name"], hero["id"]))
labels = {hero["id"]: hero["name"] for hero in heroes}
hero_ids = [hero["id"] for hero in heroes]
own_id = st.selectbox("Your hero", hero_ids, format_func=labels.get)
enemy_ids = []
for number in range(1, 7):
    picked = st.selectbox(
        f"Enemy {number}",
        [None, *hero_ids],
        format_func=lambda hero_id: "" if hero_id is None else labels[hero_id],
        key=f"enemy-{number}",
    )
    enemy_ids.append(picked)

message = validate_draft(own_id, enemy_ids, hero_ids)
if message is not None:
    st.warning(message)
    st.stop()

build = suggest(
    own_id,
    enemy_ids,
    context.index,
    context.targets,
    context.hero_ids,
    context.hero_catalog,
    context.item_catalog,
    context.names,
    context.item_times,
    settings.top_k,
    settings.top_items,
    context.skills.get(own_id, []),
)
if not build.items and not build.skills:
    st.info("No winning drafts for this hero")
    st.stop()

st.subheader("Skills")
if not build.skills:
    st.caption("No items")
else:
    st.markdown(_phase_html(build.skills, show_time=False), unsafe_allow_html=True)

phases = group_phases(build.items)
for key, heading in PHASES:
    st.subheader(heading)
    rows = phases[key]
    if not rows:
        st.caption("No items")
    else:
        st.markdown(_phase_html(rows, show_time=True), unsafe_allow_html=True)
